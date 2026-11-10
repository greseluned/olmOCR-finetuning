#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_inference.py

Inferencia con el modelo olmOCR fine-tuneado sobre imágenes de prensa
histórica. Con --modo-prompt auto (por defecto) el modelo clasifica la página
y se transcribe con el prompt de ese tipo; con unico se usa config.PROMPT_UNICO;
xml usa el PAGE XML y solo sirve para reproducir resultados antiguos. Extrae
natural_text de la salida de olmOCR, guarda un TXT por imagen, usa checkpoint
para reanudar y, con --evaluar, calcula el CER sobre el test set.

Uso:
    python scripts/04_inference.py --evaluar --jsonl
    python scripts/04_inference.py --evaluar --modo-prompt unico --salida outputs/unico
    python scripts/04_inference.py --imagen corpus/fémina/jpg/femina_1922_p03.jpg
    python scripts/04_inference.py --dir corpus/fémina/jpg/ --jsonl

Opciones:
    --imagen / --dir / --evaluar   origen de las imágenes (una, carpeta o test set)
    --modo-prompt auto|unico|xml   estrategia de prompting (auto)
    --modelo RUTA                  modelo fusionado (config.INFERENCIA_MODELO_DIR)
    --max-tokens N                 tokens máximos de generación (4096)
    --max-ancho N                  reescala las imágenes a ese ancho (reduce si hay OOM)
    --jsonl                        guarda también la salida en JSONL
    --mostrar                      imprime las transcripciones
    --salida RUTA                  directorio de salida (config.OUTPUTS_DIR)

Requiere limpiar_salida.py y el modelo fusionado generado por 03_train_lora.py.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from limpiar_salida import extraer_natural_text

try:
    import editdistance as _ed
except ImportError:
    _ed = None

TIPO_FALLBACK = "mixta"   # si la salida del clasificador no contiene ninguna etiqueta válida
MAX_TOKENS_CLASIFICACION = 96   # la salida nativa de olmOCR antepone ~45 tokens de metadatos JSON


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — SELECCIÓN DE PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def parsear_tipo(salida: str) -> tuple[str, bool]:
    texto = salida.lower().replace(" ", "_").replace("-", "_")
    candidatas = [(texto.find(t), t) for t in config.TIPOS_PAGINA if t in texto]
    if not candidatas:
        return TIPO_FALLBACK, True
    return min(candidatas)[1], False


# ── Modo xml (solo para reproducir resultados antiguos) ──────────────────────

_PARSER_XML = None


def _parser_xml():
    global _PARSER_XML
    if _PARSER_XML is None:
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "parse_pagexml",
            Path(__file__).parent / "01_parse_pagexml.py",
        )
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PARSER_XML = mod
    return _PARSER_XML


def _es_portada(stem: str) -> bool:
    s = stem.lower()
    return s.endswith("_p001") or s.endswith("_001")


def seleccionar_prompt_xml(ruta_imagen: Path) -> tuple[str, str]:
    stem = ruta_imagen.stem

    if _es_portada(stem):
        return config.PROMPTS["portada"], "portada"

    mod = _parser_xml()
    posibles_xml = [
        ruta_imagen.parent.parent / "page" / (stem + ".xml"),
        ruta_imagen.parent / "page" / (stem + ".xml"),
        ruta_imagen.with_suffix(".xml"),
    ]
    for ruta_xml in posibles_xml:
        if ruta_xml.exists():
            _, meta = mod.xml_a_transcripcion(ruta_xml)
            if "error" not in meta:
                meta["es_portada"] = False
                tipo = mod.clasificar_tipo_pagina(meta)
                return config.PROMPTS[tipo], f"{tipo}"

    return config.PROMPTS["mixta"], "mixta (fallback)"


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def cargar_checkpoint(
    ruta: Path, modo_prompt: str
) -> tuple[set[str], dict[str, dict]]:
    if not ruta.exists():
        return set(), {}
    try:
        data = json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        print(f"  [AVISO] No se pudo leer {ruta}; se empieza desde cero.")
        return set(), {}

    modo_previo = data.get("modo_prompt")
    if modo_previo is not None and modo_previo != modo_prompt:
        print(
            f"ERROR: el checkpoint de {ruta.parent} se creó con "
            f"--modo-prompt {modo_previo}, no con {modo_prompt}.\n"
            f"  Usa otro --salida o borra {ruta.name}."
        )
        sys.exit(1)

    procesadas = set(data.get("procesadas", []))
    resultados = data.get("resultados", {})
    print(f"  Checkpoint cargado: {len(procesadas)} imágenes ya procesadas.")
    return procesadas, resultados


def guardar_checkpoint(
    ruta: Path,
    procesadas: set[str],
    resultados: dict[str, dict],
    modo_prompt: str,
) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(ruta.name + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "modo_prompt": modo_prompt,
                "procesadas":  sorted(procesadas),
                "resultados":  resultados,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, ruta)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — MODELO olmOCR
# ══════════════════════════════════════════════════════════════════════════════

class ModeloOCR:

    def __init__(self, ruta_modelo: Path, max_ancho: int | None = None):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        print(f"Cargando modelo desde «{ruta_modelo}» …")
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
        self._torch     = torch
        self._max_ancho = max_ancho
        print("  Modelo cargado correctamente.")

    def cargar_imagen(self, ruta_imagen: Path):
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
        return img_pil

    def _generar(
        self,
        img_pil,
        instruccion:    str,
        max_new_tokens: int,
    ) -> tuple[str, float]:
        texto_instruccion = instruccion.replace("<image>\n", "")

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

        input_len  = inputs["input_ids"].shape[1]
        gen_tokens = output_ids[0][input_len:]
        salida     = self._processor.decode(
            gen_tokens, skip_special_tokens=True
        ).strip()

        # ── Liberar memoria ───────────────────────────────────────────────────
        del output_ids, gen_tokens, inputs
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

        return salida, duracion

    def clasificar(self, img_pil) -> tuple[str, str, bool]:
        salida, _ = self._generar(
            img_pil, config.PROMPT_CLASIFICADOR, max_new_tokens=MAX_TOKENS_CLASIFICACION
        )
        tipo, fallback = parsear_tipo(salida)
        return tipo, salida, fallback

    def transcribir(
        self,
        img_pil,
        prompt:         str,
        max_new_tokens: int = config.INFERENCIA_MAX_NEW_TOKENS,
    ) -> tuple[str, str, str, float]:
        raw, duracion = self._generar(img_pil, prompt, max_new_tokens)
        texto, formato = extraer_natural_text(raw)
        return texto, raw, formato, duracion


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — GUARDADO DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def guardar_resultado(
    ruta_imagen:   Path,
    transcripcion: str,
    duracion:      float,
    tipo_pagina:   str,
    dir_salida:    Path,
    guardar_jsonl: bool,
    clasificacion_raw:      str | None = None,
    clasificacion_fallback: bool = False,
    transcripcion_raw: str | None = None,
    formato_salida: str | None = None,
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
        if transcripcion_raw is not None:
            registro["transcripcion_raw"] = transcripcion_raw
            registro["formato_salida"]    = formato_salida
        if clasificacion_raw is not None:
            registro["clasificacion_raw"]      = clasificacion_raw
            registro["clasificacion_fallback"] = clasificacion_fallback
        with open(dir_salida / "inferencias.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — BUCLE DE INFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

def procesar_imagenes(
    imagenes:       list[Path],
    modelo_ocr:     ModeloOCR,
    modo_prompt:    str,
    dir_salida:     Path,
    max_new_tokens: int,
    guardar_jsonl:  bool,
    mostrar:        bool,
    gt_index:       dict[str, str] | None = None,
) -> list[dict]:
    ruta_checkpoint = dir_salida / "checkpoint.json"
    procesadas, resultados = cargar_checkpoint(ruta_checkpoint, modo_prompt)

    total = len(imagenes)
    print(f"\nTranscribiendo {total} páginas con olmOCR "
          f"(modo prompt: {modo_prompt}) …\n")

    for i, ruta_img in enumerate(imagenes, 1):
        stem = ruta_img.stem

        if stem in procesadas:
            print(f"  [{i:>3}/{total}] {ruta_img.name} — ya procesada, saltando")
            continue

        print(f"  [{i:>3}/{total}] {ruta_img.name} … ", end="", flush=True)

        img = None
        clas_raw, clas_fallback = None, False
        try:
            img = modelo_ocr.cargar_imagen(ruta_img)

            # ── Selección de prompt ──────────────────────────────────────────
            if modo_prompt == "unico":
                prompt, tipo = config.PROMPT_UNICO, "unico"
            elif modo_prompt == "xml":
                prompt, tipo = seleccionar_prompt_xml(ruta_img)
            else:
                tipo, clas_raw, clas_fallback = modelo_ocr.clasificar(img)
                prompt = config.PROMPTS[tipo]
            marca = "*" if clas_fallback else ""
            print(f"[{tipo}{marca}] ", end="", flush=True)

            # ── Transcripción (misma imagen ya cargada) ──────────────────────
            transcripcion, raw, formato, duracion = modelo_ocr.transcribir(
                img, prompt, max_new_tokens
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        finally:
            del img

        aviso = "" if formato == "json" else f"  [{formato}]"
        cer = None
        if gt_index and stem in gt_index:
            cer = _cer_aproximado(gt_index[stem], transcripcion)
            print(f"CER={cer:.3f}  {duracion:.1f}s{aviso}")
        else:
            print(f"{len(transcripcion)} chars — {duracion:.1f}s{aviso}")

        if mostrar:
            print(f"\n{'─'*60}")
            print(transcripcion)
            print(f"{'─'*60}\n")

        guardar_resultado(
            ruta_img, transcripcion, duracion, tipo,
            dir_salida, guardar_jsonl,
            clasificacion_raw=clas_raw,
            clasificacion_fallback=clas_fallback,
            transcripcion_raw=raw,
            formato_salida=formato,
        )

        resultado = {
            "imagen":      str(ruta_img),
            "tipo_pagina": tipo,
            "duracion_s":  round(duracion, 2),
            "formato_salida": formato,
        }
        if clas_raw is not None:
            resultado["clasificacion_fallback"] = clas_fallback
        if cer is not None:
            resultado["cer"] = cer

        resultados[stem] = resultado
        procesadas.add(stem)
        guardar_checkpoint(ruta_checkpoint, procesadas, resultados, modo_prompt)

        del transcripcion

    return [resultados[p.stem] for p in imagenes if p.stem in resultados]


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — EVALUACIÓN SOBRE EL TEST SET
# ══════════════════════════════════════════════════════════════════════════════

def evaluar_test_set(
    modelo_ocr:     ModeloOCR,
    modo_prompt:    str,
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

    # Del JSONL solo se usan la imagen y el GT; el prompt guardado se ignora.
    imagenes = [Path(r["image"]) for r in registros]
    gt_index = {
        Path(r["image"]).stem: r["conversations"][1]["content"]
        for r in registros
    }

    print(f"  {len(imagenes)} páginas en el test set")

    resultados = procesar_imagenes(
        imagenes       = imagenes,
        modelo_ocr     = modelo_ocr,
        modo_prompt    = modo_prompt,
        dir_salida     = dir_salida,
        max_new_tokens = max_new_tokens,
        guardar_jsonl  = guardar_jsonl,
        mostrar        = mostrar,
        gt_index       = gt_index,
    )

    faltan = len(imagenes) - len(resultados)
    if faltan:
        print(f"\n  [AVISO] {faltan} páginas sin resultado "
              f"(errores o checkpoint antiguo sin métricas): el resumen las excluye.")

    antiguas = sum(1 for r in resultados if "formato_salida" not in r)
    if antiguas:
        print(f"\n  [AVISO] {antiguas} páginas del checkpoint se procesaron antes de "
              f"extraer natural_text: su CER no es comparable. Borra checkpoint.json "
              f"(y los TXT) o usa otro --salida.")

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

        n_fallback = sum(1 for r in resultados if r.get("clasificacion_fallback"))
        if modo_prompt == "auto":
            print(f"\nClasificaciones no reconocidas (→ {TIPO_FALLBACK}): "
                  f"{n_fallback}/{len(resultados)}")

        formatos: dict[str, int] = {}
        for r in resultados:
            k = r.get("formato_salida", "antiguo")
            formatos[k] = formatos.get(k, 0) + 1
        print(f"\nFormato de salida del modelo: {formatos}")

        ruta_resumen = dir_salida / "evaluacion_test.json"
        ruta_resumen.write_text(
            json.dumps({
                "modo_prompt":              modo_prompt,
                "cer_medio":                cer_medio,
                "n_paginas":                len(cers),
                "n_fallback_clasificacion": n_fallback,
                "formatos_salida":          formatos,
                "resultados":               resultados,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nResumen: {ruta_resumen}")

    print(f"Resultados en: {dir_salida}")


def _cer_aproximado(referencia: str, prediccion: str) -> float:
    ref = referencia.strip()
    pred = prediccion.strip()

    if not ref:
        return 0.0 if not pred else 1.0

    m, n = len(ref), len(pred)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i

        for j in range(1, n + 1):
            coste = 0 if ref[i - 1] == pred[j - 1] else 1
            dp[j] = min(
                prev[j] + 1,
                dp[j - 1] + 1,
                prev[j - 1] + coste,
            )

    return dp[n] / m



# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inferencia olmOCR fine-tuneado.\n"
            "Por defecto el propio modelo clasifica cada página viendo la "
            "imagen y aplica las reglas de ese tipo. Sin XML ni docLayout."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Evaluar sobre el test set (el modelo clasifica; reanuda si se interrumpe)
  python scripts/04_inference.py --evaluar --jsonl

  # Evaluar con un único prompt, en otro directorio
  python scripts/04_inference.py --evaluar --modo-prompt unico --salida outputs/unico

  # Imagen individual
  python scripts/04_inference.py --imagen corpus/fémina/jpg/femina_1922_p03.jpg

  # Directorio completo
  python scripts/04_inference.py --dir corpus/fémina/jpg/ --jsonl

  # Reducir ancho máximo si hay OOM
  python scripts/04_inference.py --evaluar --max-ancho 1200
        """,
    )
    parser.add_argument("--imagen",   default=None,
                        help="Ruta a un JPG individual")
    parser.add_argument("--dir",      default=None,
                        help="Directorio con JPGs")
    parser.add_argument("--evaluar",  action="store_true",
                        help="Evaluar sobre el test set del dataset")
    parser.add_argument(
        "--modo-prompt", choices=["auto", "unico", "xml"], default="auto",
        help="auto: el modelo clasifica la página y se aplican las reglas de ese "
             "tipo (default). unico: un solo prompt (config.PROMPT_UNICO). "
             "xml: tipo por nombre de fichero y PAGE XML, solo para reproducir "
             "resultados antiguos",
    )
    parser.add_argument(
        "--modelo", default=None,
        help=f"Ruta al modelo olmOCR fusionado "
             f"(default: {config.INFERENCIA_MODELO_DIR})",
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
    parser.add_argument("--jsonl",   action="store_true",
                        help="Guardar salida también como JSONL")
    parser.add_argument("--mostrar", action="store_true",
                        help="Imprimir transcripción en consola")
    parser.add_argument(
        "--salida", default=None,
        help=f"Directorio de salida (default: {config.OUTPUTS_DIR})",
    )
    args = parser.parse_args()

    if not any([args.imagen, args.dir, args.evaluar]):
        parser.print_help()
        sys.exit(1)

    ruta_modelo = Path(args.modelo) if args.modelo else config.INFERENCIA_MODELO_DIR
    dir_salida  = Path(args.salida) if args.salida  else config.OUTPUTS_DIR

    # ── Cargar modelo ─────────────────────────────────────────────────────────
    modelo_ocr = ModeloOCR(ruta_modelo, max_ancho=args.max_ancho)
    print()

    # ── Modo evaluación ───────────────────────────────────────────────────────
    if args.evaluar:
        evaluar_test_set(
            modelo_ocr     = modelo_ocr,
            modo_prompt    = args.modo_prompt,
            dir_salida     = dir_salida,
            mostrar        = args.mostrar,
            max_new_tokens = args.max_tokens,
            guardar_jsonl  = args.jsonl,
        )
        return

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

    # ── Procesar ──────────────────────────────────────────────────────────────
    procesar_imagenes(
        imagenes       = imagenes,
        modelo_ocr     = modelo_ocr,
        modo_prompt    = args.modo_prompt,
        dir_salida     = dir_salida,
        max_new_tokens = args.max_tokens,
        guardar_jsonl  = args.jsonl,
        mostrar        = args.mostrar,
    )
    print(f"\nResultados guardados en: {dir_salida}")


if __name__ == "__main__":
    main()