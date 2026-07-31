#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_inference.py
===============
Inferencia con el modelo olmOCR fine-tuneado para prensa histórica hispanófona.

Modos de uso:
  1. Imagen individual:
       python scripts/05_inference.py --imagen ruta/pagina.jpg

  2. Directorio (procesa todos los JPG en orden):
       python scripts/05_inference.py --dir ruta/directorio/

  3. Evaluar contra el test set del dataset:
       python scripts/05_inference.py --evaluar

En todos los casos la salida se guarda en outputs/ como TXT
y opcionalmente como JSONL con metadatos.

Opciones adicionales:
  --modelo      Ruta al modelo fusionado (default: models/prensa_hispanica_merged)
  --max-tokens  Tokens máximos de generación (default: 4096)
  --jsonl       Guardar salida también como JSONL con metadatos
  --mostrar     Imprimir transcripción en consola además de guardarla
"""

import argparse
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

# ── función auxiliar ───────────────────

def seleccionar_prompt(ruta_imagen: Path) -> str:
    """
    Selecciona el prompt adecuado para una imagen en tiempo de inferencia.

    Usa la misma lógica que 02_build_dataset.py:
    - Detecta portadas por nombre de fichero (_p001, _001).
    - Para el resto, intenta parsear el XML paralelo (si existe) para
      obtener los metadatos reales de columnas e imágenes.
    - Si no hay XML, cae en 'mixta' como prompt seguro por defecto.
    """
    stem = ruta_imagen.stem.lower()
    es_portada = stem.endswith("_p001") or stem.endswith("_001")
    if es_portada:
        return config.PROMPTS["portada"]

    # Buscar XML paralelo para clasificación real
    # Convención del corpus: la carpeta page/ está al mismo nivel que jpg/
    posibles_xml = [
        ruta_imagen.parent.parent / "page" / (ruta_imagen.stem + ".xml"),
        ruta_imagen.parent / "page" / (ruta_imagen.stem + ".xml"),
        ruta_imagen.with_suffix(".xml"),
    ]
    for ruta_xml in posibles_xml:
        if ruta_xml.exists():
            _, meta = _mod.xml_a_transcripcion(ruta_xml)
            if "error" not in meta:
                meta["es_portada"] = es_portada
                tipo = clasificar_tipo_pagina(meta)
                return config.PROMPTS[tipo]

    # Sin XML disponible: prompt mixta como fallback razonable
    return config.PROMPTS["mixta"]

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CARGA DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

class ModeloOCR:
    """
    Wrapper para Qwen2-VL fine-tuneado.
    Carga el modelo fusionado (base + LoRA) en bfloat16.
    """

    def __init__(self, ruta_modelo: Path):
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
        self._torch = torch
        print("  Modelo cargado correctamente.")

    def transcribir(
        self,
        ruta_imagen: Path,
        max_new_tokens: int = config.INFERENCIA_MAX_NEW_TOKENS,
        prompt: str | None = None,          # ← nuevo parámetro
    ) -> tuple[str, float]:

        from PIL import Image

        try:
            img_pil = Image.open(ruta_imagen).convert("RGB")
        except Exception as e:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}") from e

        # Seleccionar prompt: prioridad al pasado explícitamente, luego adaptativo
        instruccion = prompt or seleccionar_prompt(ruta_imagen)
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

        # Tokenizar
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            # Fallback para versiones antiguas del processor
            text = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = self._processor(
                text=[text], images=[img_pil], return_tensors="pt"
            )

        # Mover a GPU si está disponible
        if self._torch.cuda.is_available():
            inputs = {
                k: v.cuda() if hasattr(v, "cuda") else v
                for k, v in inputs.items()
            }

        # Generar
        t0 = time.time()
        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=config.INFERENCIA_DO_SAMPLE,
            )
        duracion = time.time() - t0

        # Decodificar solo los tokens generados (excluir el prompt)
        input_len = inputs["input_ids"].shape[1]
        gen_tokens = output_ids[0][input_len:]
        transcripcion = self._processor.decode(gen_tokens, skip_special_tokens=True)

        return transcripcion.strip(), duracion


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — GUARDADO DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def guardar_resultado(
    ruta_imagen: Path,
    transcripcion: str,
    duracion: float,
    dir_salida: Path,
    guardar_jsonl: bool,
) -> None:
    """Guarda la transcripción como TXT y opcionalmente como línea JSONL."""
    dir_salida.mkdir(parents=True, exist_ok=True)

    # TXT
    ruta_txt = dir_salida / (ruta_imagen.stem + ".txt")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(transcripcion)

    # JSONL
    if guardar_jsonl:
        ruta_jsonl = dir_salida / "inferencias.jsonl"
        registro = {
            "imagen":        str(ruta_imagen.resolve()),
            "transcripcion": transcripcion,
            "duracion_s":    round(duracion, 2),
            "chars":         len(transcripcion),
        }
        with open(ruta_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — EVALUACIÓN SOBRE EL TEST SET
# ══════════════════════════════════════════════════════════════════════════════

def evaluar_test_set(modelo: ModeloOCR, dir_salida: Path, mostrar: bool) -> None:
    """
    Ejecuta inferencia sobre todas las imágenes del test set y calcula
    la similitud de caracteres (CER aproximado) respecto a la referencia.
    """
    if not config.JSONL_TEST.exists():
        print(f"ERROR: {config.JSONL_TEST} no existe. Ejecuta 03_split_dataset.py.")
        sys.exit(1)

    registros = []
    with open(config.JSONL_TEST, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                registros.append(json.loads(linea))

    print(f"Evaluando {len(registros)} páginas del test set …\n")

    resultados = []
    for i, reg in enumerate(registros, 1):
        ruta_img = Path(reg["image"])
        ref = reg["conversations"][1]["content"] 
        prompt_usado = reg["conversations"][0]["content"]

        print(f"  [{i:>3}/{len(registros)}] {ruta_img.name} … ", end="", flush=True)
        try:
            pred, dur = modelo.transcribir(ruta_img, prompt=prompt_usado)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # CER aproximado (ratio de edición a nivel de carácter)
        cer = _cer_aproximado(ref, pred)
        print(f"CER={cer:.3f}  {dur:.1f}s")

        if mostrar:
            print(f"\n    REFERENCIA: {ref[:200]}")
            print(f"    PREDICCIÓN: {pred[:200]}\n")

        resultados.append({"imagen": str(ruta_img), "cer": cer, "duracion_s": dur})
        guardar_resultado(ruta_img, pred, dur, dir_salida, guardar_jsonl=True)

    if resultados:
        cer_medio = sum(r["cer"] for r in resultados) / len(resultados)
        print(f"\n{'─'*50}")
        print(f"CER medio sobre {len(resultados)} páginas: {cer_medio:.4f}")
        print(f"Resultados guardados en: {dir_salida}")

        # Guardar resumen
        ruta_resumen = dir_salida / "evaluacion_test.json"
        with open(ruta_resumen, "w", encoding="utf-8") as f:
            json.dump({
                "cer_medio":    cer_medio,
                "n_paginas":    len(resultados),
                "resultados":   resultados,
            }, f, ensure_ascii=False, indent=2)
        print(f"Resumen guardado en: {ruta_resumen}")


def _cer_aproximado(referencia: str, prediccion: str) -> float:
    """
    Character Error Rate aproximado usando distancia de Levenshtein
    implementada de forma simple (suficiente para comparación relativa).
    """
    ref = referencia.strip()
    pred = prediccion.strip()
    if not ref:
        return 0.0 if not pred else 1.0

    # Levenshtein simplificado con programación dinámica
    m, n = len(ref), len(pred)
    # Limitar a 5000 chars para no saturar memoria en páginas largas
    ref  = ref[:5000]
    pred = pred[:5000]
    m, n = len(ref), len(pred)

    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if ref[i-1] == pred[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])

    return dp[n] / max(m, n)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Inferencia con el modelo olmOCR fine-tuneado.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Imagen individual
  python scripts/05_inference.py --imagen corpus/pero/jpg/Excelsior-10-09-1932_p001.jpg

  # Directorio completo
  python scripts/05_inference.py --dir corpus/pero/jpg/ --jsonl

  # Evaluar contra el test set
  python scripts/05_inference.py --evaluar --mostrar
        """,
    )
    parser.add_argument("--imagen",     default=None, help="Ruta a un JPG individual")
    parser.add_argument("--dir",        default=None, help="Directorio con JPGs")
    parser.add_argument("--evaluar",    action="store_true", help="Evaluar sobre el test set")
    parser.add_argument(
        "--modelo",
        default=None,
        help=f"Ruta al modelo fusionado (default: {config.INFERENCIA_MODELO_DIR})"
    )
    parser.add_argument("--max-tokens", type=int, default=config.INFERENCIA_MAX_NEW_TOKENS)
    parser.add_argument("--jsonl",      action="store_true", help="Guardar también como JSONL")
    parser.add_argument("--mostrar",    action="store_true", help="Imprimir transcripción en consola")
    parser.add_argument(
        "--salida",
        default=None,
        help=f"Directorio de salida (default: {config.OUTPUTS_DIR})"
    )
    args = parser.parse_args()

    if not any([args.imagen, args.dir, args.evaluar]):
        parser.print_help()
        sys.exit(1)

    ruta_modelo = Path(args.modelo) if args.modelo else config.INFERENCIA_MODELO_DIR
    dir_salida  = Path(args.salida) if args.salida  else config.OUTPUTS_DIR

    # ── Cargar modelo ─────────────────────────────────────────────────────────
    modelo = ModeloOCR(ruta_modelo)

    # ── Modo evaluación ───────────────────────────────────────────────────────
    if args.evaluar:
        evaluar_test_set(modelo, dir_salida / "evaluacion", args.mostrar)
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
    for i, ruta_img in enumerate(imagenes, 1):
        print(f"[{i:>3}/{len(imagenes)}] {ruta_img.name} … ", end="", flush=True)
        try:
            transcripcion, duracion = modelo.transcribir(ruta_img, args.max_tokens)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        print(f"{len(transcripcion)} chars — {duracion:.1f}s")

        guardar_resultado(ruta_img, transcripcion, duracion, dir_salida, args.jsonl)

        if args.mostrar:
            print(f"\n{'─'*60}")
            print(transcripcion)
            print(f"{'─'*60}\n")

    print(f"\nResultados guardados en: {dir_salida}")


if __name__ == "__main__":
    main()