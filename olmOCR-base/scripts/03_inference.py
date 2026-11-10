#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_inference.py

Inferencia con olmOCR base (sin fine-tuning) sobre el corpus, como baseline.
Con --modo-prompt auto el modelo clasifica la página y se transcribe con el
prompt de ese tipo; con unico se usa config.PROMPT_UNICO. Guarda un TXT por
imagen, usa checkpoint para reanudar y, con --evaluar, calcula el CER sobre
el test set.

Uso:
    python scripts/03_inference.py --evaluar --jsonl
    python scripts/03_inference.py --evaluar --modo-prompt unico
    python scripts/03_inference.py --imagen ../corpus/fémina/jpg/femina_1922_p03.jpg
    python scripts/03_inference.py --dir ../corpus/fémina/jpg/ --jsonl

Opciones:
    --imagen / --dir / --evaluar   origen de las imágenes (una, carpeta o test set)
    --modo-prompt auto|unico       estrategia de prompting (auto)
    --max-tokens N                 tokens máximos de generación
    --max-ancho N                  reescala las imágenes a ese ancho
    --jsonl                        guarda también la salida en JSONL
    --mostrar                      imprime las transcripciones
    --salida RUTA                  directorio de salida

Salida:
    outputs/evaluacion_baseline/ con checkpoint.json, inferencias.jsonl,
    evaluacion_test_baseline.json y los TXT. Requiere limpiar_salida.py.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from limpiar_salida import extraer_natural_text


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# Directorio de salida por defecto
OUTPUTS_BASELINE_DIR = config.ROOT / "outputs"

# Tipo de página que se asigna si la salida del clasificador no contiene
# ninguna etiqueta válida
TIPO_FALLBACK = "mixta"

# Tokens máximos de la pasada de clasificación (la salida nativa de olmOCR
# antepone ~45 tokens de metadatos JSON)
MAX_TOKENS_CLASIFICACION = 96


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CHECKPOINT
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
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps({"procesadas": sorted(procesadas)}, ensure_ascii=False),
        encoding="utf-8",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — MODELO olmOCR BASE
# ══════════════════════════════════════════════════════════════════════════════

def parsear_tipo(salida: str) -> tuple[str, bool]:
    texto = salida.lower().replace(" ", "_").replace("-", "_")
    candidatas = [(texto.find(t), t) for t in config.TIPOS_PAGINA if t in texto]
    if not candidatas:
        return TIPO_FALLBACK, True
    return min(candidatas)[1], False


class ModeloOCR:

    def __init__(self, max_ancho: int | None = None):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        modelo_id = config.MODELO_BASE
        print(f"  Cargando olmOCR BASE: «{modelo_id}» …")

        self._processor = AutoProcessor.from_pretrained(
            modelo_id, trust_remote_code=True
        )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            modelo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()
        self._torch     = torch
        self._max_ancho = max_ancho
        print("  olmOCR BASE listo.")

    def _cargar_imagen(self, ruta_imagen: Path):
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
        instruccion: str,
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

        # Liberar memoria (CPU y GPU) entre páginas
        del output_ids, gen_tokens, inputs
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

        return salida, duracion

    def clasificar(self, ruta_imagen: Path) -> tuple[str, str, bool]:
        img_pil = self._cargar_imagen(ruta_imagen)
        try:
            salida, _ = self._generar(
                img_pil, config.PROMPT_CLASIFICADOR, max_new_tokens=MAX_TOKENS_CLASIFICACION
            )
        finally:
            del img_pil
        tipo, fallback = parsear_tipo(salida)
        return tipo, salida, fallback

    def transcribir(
        self,
        ruta_imagen: Path,
        prompt: str,
        max_new_tokens: int = config.INFERENCIA_MAX_NEW_TOKENS,
    ) -> tuple[str, str, str, float]:
        img_pil = self._cargar_imagen(ruta_imagen)
        try:
            raw, duracion = self._generar(img_pil, prompt, max_new_tokens)
        finally:
            del img_pil
        texto, formato = extraer_natural_text(raw)
        return texto, raw, formato, duracion


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — GUARDADO DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def guardar_resultado(
    ruta_imagen:            Path,
    transcripcion:          str,
    duracion:               float,
    tipo_pagina:            str,
    dir_salida:             Path,
    guardar_jsonl:          bool,
    clasificacion_raw:      str | None = None,
    clasificacion_fallback: bool = False,
    transcripcion_raw:      str | None = None,
    formato_salida:         str | None = None,
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
# SECCIÓN 4 — BUCLE DE INFERENCIA
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
    procesadas = cargar_checkpoint(ruta_checkpoint)

    total      = len(imagenes)
    resultados = []

    print(f"Transcribiendo {total} páginas con olmOCR BASE "
          f"(modo prompt: {modo_prompt}) …\n")

    for i, ruta_img in enumerate(imagenes, 1):
        stem = ruta_img.stem

        if stem in procesadas:
            print(f"  [{i:>3}/{total}] {ruta_img.name} — ya procesada, saltando")
            continue

        print(f"  [{i:>3}/{total}] {ruta_img.name} … ", end="", flush=True)

        # ── Selección del prompt ──────────────────────────────────────────────
        clas_raw, clas_fallback = None, False
        if modo_prompt == "unico":
            prompt, tipo = config.PROMPT_UNICO, "unico"
        else:
            try:
                tipo, clas_raw, clas_fallback = modelo_ocr.clasificar(ruta_img)
            except Exception as e:
                print(f"ERROR clasificando: {e}")
                continue
            prompt = config.PROMPTS[tipo]
            marca = "*" if clas_fallback else ""
            print(f"[{tipo}{marca}] ", end="", flush=True)

        # ── Transcripción ─────────────────────────────────────────────────────
        try:
            transcripcion, raw, formato, duracion = modelo_ocr.transcribir(
                ruta_img, prompt, max_new_tokens
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

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

        # ── Guardado ──────────────────────────────────────────────────────────
        guardar_resultado(
            ruta_img, transcripcion, duracion, tipo,
            dir_salida, guardar_jsonl,
            clasificacion_raw=clas_raw,
            clasificacion_fallback=clas_fallback,
            transcripcion_raw=raw,
            formato_salida=formato,
        )

        resultado = {
            "imagen":         str(ruta_img),
            "tipo_pagina":    tipo,
            "duracion_s":     round(duracion, 2),
            "formato_salida": formato,
        }
        if clas_raw is not None:
            resultado["clasificacion_fallback"] = clas_fallback
        if cer is not None:
            resultado["cer"] = cer
        resultados.append(resultado)

        procesadas.add(stem)
        guardar_checkpoint(ruta_checkpoint, procesadas)
        del transcripcion

    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — EVALUACIÓN SOBRE EL TEST SET
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
        print(f"ERROR: {config.JSONL_TEST} no existe. Ejecuta 02_build_dataset.py.")
        sys.exit(1)

    registros: list[dict] = []
    with open(config.JSONL_TEST, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                registros.append(json.loads(linea))

    # Del JSONL solo se usan la imagen y el ground truth; el prompt guardado se ignora
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

    cers = [r["cer"] for r in resultados if "cer" in r]
    if cers:
        cer_medio = sum(cers) / len(cers)
        print(f"\n{'─'*55}")
        print(f"CER medio sobre {len(cers)} páginas: {cer_medio:.4f}")

        tipos_agrup: dict[str, list] = {}
        for r in resultados:
            if "cer" in r:
                tipos_agrup.setdefault(r["tipo_pagina"], []).append(r["cer"])
        print(f"\nCER por tipo de página (según el modelo):")
        for tipo, vals in sorted(tipos_agrup.items()):
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

        ruta_resumen = dir_salida / "evaluacion_test_baseline.json"
        ruta_resumen.write_text(
            json.dumps({
                "modelo":                   config.MODELO_BASE,
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
# SECCIÓN 6 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inferencia con olmOCR BASE (sin fine-tuning).\n"
            "No usa XML, nombre de fichero ni DocLayout: el propio modelo "
            "clasifica la página viendo la imagen (modo auto) o se usa un "
            "único prompt (modo unico)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/03_inference.py --evaluar --jsonl
  python scripts/03_inference.py --evaluar --modo-prompt unico --salida outputs/baseline_unico
  python scripts/03_inference.py --imagen ../corpus/fémina/jpg/femina_1922_p03.jpg
  python scripts/03_inference.py --dir ../corpus/fémina/jpg/ --jsonl
        """,
    )
    parser.add_argument(
        "--imagen", default=None,
        help="Ruta a un JPG individual",
    )
    parser.add_argument(
        "--dir", default=None,
        help="Directorio con JPGs",
    )
    parser.add_argument(
        "--evaluar", action="store_true",
        help="Evaluar sobre el test set del dataset",
    )
    parser.add_argument(
        "--modo-prompt", choices=["auto", "unico"], default="auto",
        help="auto: olmOCR clasifica la página y se aplican las reglas de ese "
             "tipo (default). unico: un solo prompt (config.PROMPT_UNICO)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=config.INFERENCIA_MAX_NEW_TOKENS,
        help=f"Tokens máximos de generación (default: {config.INFERENCIA_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--max-ancho", type=int, default=None,
        help="Ancho máximo de imagen en px antes de escalar",
    )
    parser.add_argument(
        "--jsonl", action="store_true",
        help="Guardar la salida también como JSONL",
    )
    parser.add_argument(
        "--mostrar", action="store_true",
        help="Imprimir la transcripción en consola",
    )
    parser.add_argument(
        "--salida", default=None,
        help=f"Directorio de salida (default: {OUTPUTS_BASELINE_DIR})",
    )
    args = parser.parse_args()

    if not any([args.imagen, args.dir, args.evaluar]):
        parser.print_help()
        sys.exit(1)

    dir_salida = Path(args.salida) if args.salida else OUTPUTS_BASELINE_DIR

    # ── Recopilación de imágenes ──────────────────────────────────────────────
    imagenes: list[Path] = []
    registros: list[dict] = []

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
            print(f"ERROR: {config.JSONL_TEST} no existe. Ejecuta 02_build_dataset.py.")
            sys.exit(1)
        with open(config.JSONL_TEST, encoding="utf-8") as f:
            registros = [json.loads(l) for l in f if l.strip()]
        imagenes = [Path(r["image"]) for r in registros]

    # ── Carga de olmOCR BASE y transcripción ──────────────────────────────────
    print("Cargando olmOCR BASE …")
    modelo_ocr = ModeloOCR(max_ancho=args.max_ancho)
    print()

    if args.evaluar:
        evaluar_test_set(
            modelo_ocr     = modelo_ocr,
            modo_prompt    = args.modo_prompt,
            dir_salida     = dir_salida,
            mostrar        = args.mostrar,
            max_new_tokens = args.max_tokens,
            guardar_jsonl  = args.jsonl,
        )
    else:
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