#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_inference.py

Inferencia con el olmOCR fine-tuneado usando docLayout-YOLO para clasificar el
tipo de página. Trabaja en dos fases: primero docLayout clasifica todas las
imágenes y se descarga de memoria; después se carga olmOCR y transcribe con el
prompt de cada tipo, así que los dos modelos nunca coinciden en RAM. Guarda un
TXT por imagen, usa checkpoint para reanudar la fase 2 y, con --evaluar,
calcula el CER sobre el test set.

Uso:
    python scripts/05_inference.py --evaluar --jsonl
    python scripts/05_inference.py --imagen corpus/fémina/jpg/femina_1922_p03.jpg
    python scripts/05_inference.py --dir corpus/fémina/jpg/ --jsonl
    python scripts/05_inference.py --evaluar --sin-layout

Opciones:
    --imagen / --dir / --evaluar   origen de las imágenes (una, carpeta o test set)
    --modelo-ocr RUTA              modelo olmOCR fusionado (config.INFERENCIA_MODELO_DIR)
    --modelo-layout RUTA           best.pt de docLayout
    --conf-layout X                umbral de confianza de docLayout (0.25)
    --max-tokens N                 tokens máximos de generación (4096)
    --max-ancho N                  reescala las imágenes a ese ancho (reduce si hay OOM)
    --jsonl                        guarda también la salida en JSONL
    --mostrar                      imprime las transcripciones
    --sin-layout                   usa el clasificador XML en lugar de docLayout
    --salida RUTA                  directorio de salida (config.OUTPUTS_DIR)
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "parse_pagexml",
    Path(__file__).parent / "01_parse_pagexml.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
clasificar_tipo_pagina = _mod.clasificar_tipo_pagina

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN — ruta al modelo docLayout (proyecto separado)
# ══════════════════════════════════════════════════════════════════════════════

DOCLAYOUT_MODEL_DEFAULT = (
    config.ROOT.parent
    / "docLayout_fine-tuned" / "models" / "layout-model"
    / "train" / "weights" / "best.pt"
)

CONF_LAYOUT_DEFAULT = 0.25


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CLASIFICADOR DE LAYOUT (docLayout)
# ══════════════════════════════════════════════════════════════════════════════

class ClasificadorLayout:

    def __init__(self, ruta_modelo: Path, conf: float = CONF_LAYOUT_DEFAULT):
        try:
            from doclayout_yolo import YOLOv10
        except ImportError:
            print("ERROR: doclayout-yolo no instalado.\n"
                  "  pip install doclayout-yolo")
            sys.exit(1)

        if not ruta_modelo.exists():
            print(f"ERROR: Modelo docLayout no encontrado: {ruta_modelo}")
            sys.exit(1)

        print(f"  Cargando docLayout desde «{ruta_modelo}» …")
        self._modelo = YOLOv10(str(ruta_modelo))
        self._conf   = conf
        print("  docLayout listo.")

    # ── API pública ───────────────────────────────────────────────────────────

    def clasificar_lote(self, imagenes: list[Path]) -> dict[str, str]:
        total    = len(imagenes)
        resultado: dict[str, str] = {}

        print(f"\nFase 1 — clasificando {total} páginas con docLayout …")
        for i, ruta in enumerate(imagenes, 1):
            try:
                tipo = self._clasificar_una(ruta)
            except Exception as e:
                print(f"  [{i:>3}/{total}] {ruta.name}  ERROR: {e} → mixta")
                tipo = "mixta"
            else:
                print(f"  [{i:>3}/{total}] {ruta.name}  → {tipo}")
            resultado[ruta.stem] = tipo

        print(f"Fase 1 completada.\n")
        return resultado

    def descargar(self) -> None:
        import torch
        del self._modelo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  docLayout descargado de memoria.")

    # ── Métodos internos ──────────────────────────────────────────────────────

    def _clasificar_una(self, ruta_jpg: Path) -> str:
        if _es_portada(ruta_jpg.stem):
            return "portada"

        ancho = self._ancho_imagen(ruta_jpg)
        cajas = self._predecir(ruta_jpg)

        if not cajas:
            return "texto_corrido"

        CLASES_NOTEX = {"no-text", "graphic", "image", "table"}
        cajas_texto = [c for c in cajas if c["clase"] == "text"]
        cajas_notex = [c for c in cajas if c["clase"] in CLASES_NOTEX]

        total = len(cajas_texto) + len(cajas_notex)
        fraccion_imagen = len(cajas_notex) / total if total else 0.0

        if fraccion_imagen >= 0.50:
            return "imagen_dominante"

        num_columnas = self._detectar_columnas(cajas_texto, ancho)

        if num_columnas <= 1:
            return "texto_corrido"

        tiene_cabecera = any(c["ancho"] > ancho * 0.60 for c in cajas_texto)
        return "mixta" if tiene_cabecera else "columnar"

    def _predecir(self, ruta_jpg: Path) -> list[dict]:
        resultados = self._modelo.predict(
            source=str(ruta_jpg),
            conf=self._conf,
            verbose=False,
        )
        cajas = []
        for r in resultados:
            nombres = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cajas.append({
                    "clase": nombres[cls_id],
                    "conf":  float(box.conf[0]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx":    (x1 + x2) / 2,
                    "cy":    (y1 + y2) / 2,
                    "ancho": x2 - x1,
                    "alto":  y2 - y1,
                })
        return cajas

    @staticmethod
    def _ancho_imagen(ruta_jpg: Path) -> float:
        from PIL import Image
        with Image.open(ruta_jpg) as im:
            return float(im.width)

    @staticmethod
    def _detectar_columnas(cajas_texto: list[dict], ancho_pagina: float) -> int:
        cxs = [
            c["cx"] for c in cajas_texto
            if c["ancho"] <= ancho_pagina * 0.60
        ]
        if len(cxs) < 2:
            return 1
        cxs_sorted = sorted(set(round(cx) for cx in cxs))
        umbral = ancho_pagina * 0.12
        cortes = sum(
            1 for i in range(len(cxs_sorted) - 1)
            if cxs_sorted[i + 1] - cxs_sorted[i] > umbral
        )
        return cortes + 1


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — SELECCIÓN DE PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def _es_portada(stem: str) -> bool:
    s = stem.lower()
    return s.endswith("_p001") or s.endswith("_001")


def seleccionar_prompt(
    ruta_imagen:  Path,
    tipos_layout: dict[str, str],
) -> tuple[str, str]:
    stem = ruta_imagen.stem

    if _es_portada(stem):
        return config.PROMPTS["portada"], "portada"

    if stem in tipos_layout:
        tipo = tipos_layout[stem]
        return config.PROMPTS[tipo], tipo

    # ── Fallback: XML paralelo ────────────────────────────────────────────────
    posibles_xml = [
        ruta_imagen.parent.parent / "page" / (stem + ".xml"),
        ruta_imagen.parent / "page" / (stem + ".xml"),
        ruta_imagen.with_suffix(".xml"),
    ]
    for ruta_xml in posibles_xml:
        if ruta_xml.exists():
            _, meta = _mod.xml_a_transcripcion(ruta_xml)
            if "error" not in meta:
                meta["es_portada"] = False
                tipo = clasificar_tipo_pagina(meta)
                return config.PROMPTS[tipo], f"{tipo} (xml)"

    return config.PROMPTS["mixta"], "mixta (fallback)"


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def cargar_checkpoint(ruta: Path) -> set[str]:
    if not ruta.exists():
        return set()
    try:
        data = json.loads(ruta.read_text(encoding="utf-8"))
        procesadas = set(data.get("procesadas", []))
        print(f"  Checkpoint cargado: {len(procesadas)} imágenes ya procesadas.")
        return procesadas
    except Exception:
        return set()


def guardar_checkpoint(ruta: Path, procesadas: set[str]) -> None:
    ruta.write_text(
        json.dumps({"procesadas": sorted(procesadas)}, ensure_ascii=False),
        encoding="utf-8",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — MODELO olmOCR
# ══════════════════════════════════════════════════════════════════════════════

class ModeloOCR:

    def __init__(self, ruta_modelo: Path, max_ancho: int | None = None):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        print(f"  Cargando olmOCR desde «{ruta_modelo}» …")
        if not ruta_modelo.exists():
            raise FileNotFoundError(
                f"{ruta_modelo} no existe.\n"
                "Ejecuta primero 04_train_lora.py para generar el modelo fusionado."
            )

        self._processor = AutoProcessor.from_pretrained(
            str(ruta_modelo), trust_remote_code=True
        )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(ruta_modelo),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()
        self._torch    = torch
        self._max_ancho = max_ancho
        print("  olmOCR listo.")

    def transcribir(
        self,
        ruta_imagen: Path,
        prompt: str,
        max_new_tokens: int = config.INFERENCIA_MAX_NEW_TOKENS,
    ) -> tuple[str, float]:
        from PIL import Image

        try:
            img_pil = Image.open(ruta_imagen).convert("RGB")
        except Exception as e:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}") from e

        if self._max_ancho is not None and img_pil.width > self._max_ancho:
            ratio   = self._max_ancho / img_pil.width
            img_pil = img_pil.resize(
                (self._max_ancho, int(img_pil.height * ratio)),
                Image.LANCZOS,
            )

        texto_instruccion = prompt.replace("<image>\n", "")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_pil},
                    {"type": "text",  "text":  texto_instruccion},
                ],
            }
        ]

        try:
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            text = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = self._processor(
                text=[text], images=[img_pil], return_tensors="pt"
            )

        if self._torch.cuda.is_available():
            inputs = {
                k: v.cuda() if hasattr(v, "cuda") else v
                for k, v in inputs.items()
            }

        t0 = time.time()
        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=config.INFERENCIA_DO_SAMPLE,
            )
        duracion = time.time() - t0

        input_len     = inputs["input_ids"].shape[1]
        gen_tokens    = output_ids[0][input_len:]
        transcripcion = self._processor.decode(
            gen_tokens, skip_special_tokens=True
        ).strip()

        # ── Liberar memoria ───────────────────────────────────────────────────
        del output_ids, gen_tokens, inputs, img_pil
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

        return transcripcion, duracion


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — GUARDADO DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def guardar_resultado(
    ruta_imagen:   Path,
    transcripcion: str,
    duracion:      float,
    tipo_pagina:   str,
    dir_salida:    Path,
    guardar_jsonl: bool,
) -> None:
    dir_salida.mkdir(parents=True, exist_ok=True)

    (dir_salida / (ruta_imagen.stem + ".txt")).write_text(
        transcripcion, encoding="utf-8"
    )

    if guardar_jsonl:
        registro = {
            "imagen":        str(ruta_imagen.resolve()),
            "tipo_pagina":   tipo_pagina,
            "transcripcion": transcripcion,
            "duracion_s":    round(duracion, 2),
            "chars":         len(transcripcion),
        }
        with open(dir_salida / "inferencias.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — BUCLE DE INFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

def procesar_imagenes(
    imagenes:       list[Path],
    modelo_ocr:     ModeloOCR,
    tipos_layout:   dict[str, str],
    dir_salida:     Path,
    max_new_tokens: int,
    guardar_jsonl:  bool,
    mostrar:        bool,
    gt_index:       dict[str, str] | None = None,
) -> list[dict]:
    ruta_checkpoint = dir_salida / "checkpoint.json"
    procesadas = cargar_checkpoint(ruta_checkpoint)

    total      = len(imagenes)
    resultados = []

    print(f"Fase 2 — transcribiendo {total} páginas con olmOCR …\n")

    for i, ruta_img in enumerate(imagenes, 1):
        stem = ruta_img.stem

        # Saltar imágenes ya procesadas
        if stem in procesadas:
            print(f"  [{i:>3}/{total}] {ruta_img.name} — ya procesada, saltando")
            continue

        prompt, tipo = seleccionar_prompt(ruta_img, tipos_layout)
        print(f"  [{i:>3}/{total}] {ruta_img.name}  [{tipo}] … ",
              end="", flush=True)

        try:
            transcripcion, duracion = modelo_ocr.transcribir(
                ruta_img, prompt, max_new_tokens
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # CER aproximado si tenemos GT
        cer = None
        if gt_index and stem in gt_index:
            cer = _cer_aproximado(gt_index[stem], transcripcion)
            print(f"CER={cer:.3f}  {duracion:.1f}s")
        else:
            print(f"{len(transcripcion)} chars — {duracion:.1f}s")

        if mostrar:
            print(f"\n{'─'*60}")
            print(transcripcion)
            print(f"{'─'*60}\n")

        guardar_resultado(
            ruta_img, transcripcion, duracion, tipo,
            dir_salida, guardar_jsonl,
        )

        resultado = {
            "imagen":      str(ruta_img),
            "tipo_pagina": tipo,
            "duracion_s":  round(duracion, 2),
        }
        if cer is not None:
            resultado["cer"] = cer
        resultados.append(resultado)

        # Actualizar checkpoint
        procesadas.add(stem)
        guardar_checkpoint(ruta_checkpoint, procesadas)

        # Liberar referencia a la transcripción
        del transcripcion

    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — EVALUACIÓN SOBRE EL TEST SET
# ══════════════════════════════════════════════════════════════════════════════

def evaluar_test_set(
    modelo_ocr:     ModeloOCR,
    tipos_layout:   dict[str, str],
    dir_salida:     Path,
    mostrar:        bool,
    max_new_tokens: int,
    guardar_jsonl:  bool,
) -> None:
    if not config.JSONL_TEST.exists():
        print(f"ERROR: {config.JSONL_TEST} no existe. "
              f"Ejecuta 03_build_dataset.py.")
        sys.exit(1)

    registros: list[dict] = []
    with open(config.JSONL_TEST, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                registros.append(json.loads(linea))

    imagenes = [Path(r["image"]) for r in registros]
    gt_index = {
        Path(r["image"]).stem: r["conversations"][1]["content"]
        for r in registros
    }

    print(f"  {len(imagenes)} páginas en el test set")

    resultados = procesar_imagenes(
        imagenes       = imagenes,
        modelo_ocr     = modelo_ocr,
        tipos_layout   = tipos_layout,
        dir_salida     = dir_salida,
        max_new_tokens = max_new_tokens,
        guardar_jsonl  = guardar_jsonl,
        mostrar        = mostrar,
        gt_index       = gt_index,
    )

    cers = [r["cer"] for r in resultados if "cer" in r]
    if cers:
        cer_medio = sum(cers) / len(cers)
        print(f"\n{'─'*55}")
        print(f"CER medio sobre {len(cers)} páginas: {cer_medio:.4f}")

        tipos: dict[str, list] = {}
        for r in resultados:
            if "cer" in r:
                tipos.setdefault(r["tipo_pagina"], []).append(r["cer"])
        print(f"\nCER por tipo de página:")
        for tipo, vals in sorted(tipos.items()):
            print(f"  {tipo:<28} {sum(vals)/len(vals):.4f}  (n={len(vals)})")

        ruta_resumen = dir_salida / "evaluacion_test.json"
        ruta_resumen.write_text(
            json.dumps({
                "cer_medio":  cer_medio,
                "n_paginas":  len(cers),
                "resultados": resultados,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nResumen: {ruta_resumen}")

    print(f"Resultados en: {dir_salida}")


def _cer_aproximado(referencia: str, prediccion: str) -> float:
    ref  = referencia.strip()[:5000]
    pred = prediccion.strip()[:5000]
    if not ref:
        return 0.0 if not pred else 1.0
    m, n = len(ref), len(pred)
    dp   = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if ref[i-1] == pred[j-1] else \
                    1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n] / max(m, n)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — CLASIFICACIÓN FALLBACK SIN docLayout
# ══════════════════════════════════════════════════════════════════════════════

def clasificar_sin_layout(imagenes: list[Path]) -> dict[str, str]:
    total     = len(imagenes)
    resultado: dict[str, str] = {}

    print(f"\nFase 1 — clasificando {total} páginas sin docLayout (fallback) …")
    for i, ruta in enumerate(imagenes, 1):
        stem = ruta.stem
        if _es_portada(stem):
            tipo = "portada"
        else:
            tipo = "mixta"
            posibles_xml = [
                ruta.parent.parent / "page" / (stem + ".xml"),
                ruta.parent / "page" / (stem + ".xml"),
                ruta.with_suffix(".xml"),
            ]
            for ruta_xml in posibles_xml:
                if ruta_xml.exists():
                    _, meta = _mod.xml_a_transcripcion(ruta_xml)
                    if "error" not in meta:
                        meta["es_portada"] = False
                        tipo = clasificar_tipo_pagina(meta)
                    break
        print(f"  [{i:>3}/{total}] {ruta.name}  → {tipo}")
        resultado[stem] = tipo

    print("Fase 1 completada.\n")
    return resultado


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inferencia olmOCR con clasificación de layout por docLayout-YOLO.\n"
            "Fase 1: docLayout clasifica todas las páginas y se descarga.\n"
            "Fase 2: olmOCR transcribe. Los dos modelos nunca coinciden en RAM."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Evaluar sobre el test set (reanuda si se interrumpe)
  python scripts/04_inference.py --evaluar --jsonl

  # Imagen individual
  python scripts/04_inference.py --imagen corpus/fémina/jpg/femina_1922_p03.jpg

  # Directorio completo
  python scripts/04_inference.py --dir corpus/fémina/jpg/ --jsonl

  # Sin docLayout (fallback XML, para comparar/depurar)
  python scripts/04_inference.py --evaluar --sin-layout

  # Reducir ancho máximo si hay OOM
  python scripts/04_inference.py --evaluar --max-ancho 1200
        """,
    )
    parser.add_argument("--imagen",    default=None,
                        help="Ruta a un JPG individual")
    parser.add_argument("--dir",       default=None,
                        help="Directorio con JPGs")
    parser.add_argument("--evaluar",   action="store_true",
                        help="Evaluar sobre el test set del dataset")
    parser.add_argument(
        "--modelo-ocr", default=None,
        help=f"Ruta al modelo olmOCR fusionado "
             f"(default: {config.INFERENCIA_MODELO_DIR})",
    )
    parser.add_argument(
        "--modelo-layout", default=None,
        help=f"Ruta al best.pt de docLayout "
             f"(default: {DOCLAYOUT_MODEL_DEFAULT})",
    )
    parser.add_argument(
        "--conf-layout", type=float, default=CONF_LAYOUT_DEFAULT,
        help=f"Umbral de confianza docLayout (default: {CONF_LAYOUT_DEFAULT})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=config.INFERENCIA_MAX_NEW_TOKENS,
        help=f"Tokens máximos de generación "
             f"(default: {config.INFERENCIA_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--max-ancho", type=int, default=None,
        help="Ancho máximo de imagen en px antes de escalar "
             "(por defecto: resolución original; reduce si hay OOM)",
    )
    parser.add_argument("--jsonl",    action="store_true",
                        help="Guardar salida también como JSONL")
    parser.add_argument("--mostrar",  action="store_true",
                        help="Imprimir transcripción en consola")
    parser.add_argument(
        "--sin-layout", action="store_true",
        help="No usar docLayout; clasificar desde XML o usar prompt 'mixta'",
    )
    parser.add_argument(
        "--salida", default=None,
        help=f"Directorio de salida (default: {config.OUTPUTS_DIR})",
    )
    args = parser.parse_args()

    if not any([args.imagen, args.dir, args.evaluar]):
        parser.print_help()
        sys.exit(1)

    ruta_modelo_ocr = (
        Path(args.modelo_ocr) if args.modelo_ocr
        else config.INFERENCIA_MODELO_DIR
    )
    ruta_modelo_layout = (
        Path(args.modelo_layout) if args.modelo_layout
        else DOCLAYOUT_MODEL_DEFAULT
    )
    dir_salida = Path(args.salida) if args.salida else config.OUTPUTS_DIR

    # ── Recopilar imágenes ────────────────────────────────────────────────────
    imagenes: list[Path] = []
    if args.imagen:
        p = Path(args.imagen)
        if not p.exists():
            print(f"ERROR: {p} no existe.")
            sys.exit(1)
        imagenes = [p]
    elif args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            print(f"ERROR: {d} no es un directorio.")
            sys.exit(1)
        imagenes = sorted(
            {p.resolve(): p for p in d.glob("*")
             if p.suffix.lower() in (".jpg", ".jpeg")}.values()
        )
        if not imagenes:
            print(f"ERROR: No se encontraron JPGs en {d}")
            sys.exit(1)
        print(f"  {len(imagenes)} imágenes encontradas en {d}")
    elif args.evaluar:
        if not config.JSONL_TEST.exists():
            print(f"ERROR: {config.JSONL_TEST} no existe. "
                  f"Ejecuta 03_build_dataset.py.")
            sys.exit(1)
        with open(config.JSONL_TEST, encoding="utf-8") as f:
            registros = [json.loads(l) for l in f if l.strip()]
        imagenes = [Path(r["image"]) for r in registros]

    # ── Fase 1: clasificar con docLayout (o fallback) ─────────────────────────
    if args.sin_layout:
        tipos_layout = clasificar_sin_layout(imagenes)
    else:
        clasificador = ClasificadorLayout(ruta_modelo_layout, args.conf_layout)
        tipos_layout = clasificador.clasificar_lote(imagenes)
        clasificador.descargar()
        del clasificador

    # ── Fase 2: cargar olmOCR y transcribir ───────────────────────────────────
    print("Fase 2 — cargando olmOCR …")
    modelo_ocr = ModeloOCR(ruta_modelo_ocr, max_ancho=args.max_ancho)
    print()

    if args.evaluar:
        gt_index = {
            Path(r["image"]).stem: r["conversations"][1]["content"]
            for r in registros
        }
        evaluar_test_set(
            modelo_ocr     = modelo_ocr,
            tipos_layout   = tipos_layout,
            dir_salida     = dir_salida,
            mostrar        = args.mostrar,
            max_new_tokens = args.max_tokens,
            guardar_jsonl  = args.jsonl,
        )
    else:
        procesar_imagenes(
            imagenes       = imagenes,
            modelo_ocr     = modelo_ocr,
            tipos_layout   = tipos_layout,
            dir_salida     = dir_salida,
            max_new_tokens = args.max_tokens,
            guardar_jsonl  = args.jsonl,
            mostrar        = args.mostrar,
        )
        print(f"\nResultados guardados en: {dir_salida}")


if __name__ == "__main__":
    main()