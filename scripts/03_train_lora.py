#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_train_lora.py
================
Orquesta el fine-tuning QLoRA de olmOCR / Qwen2-VL-7B mediante LLaMA-Factory.

El script:
  1. Verifica que LLaMA-Factory esté instalado.
  2. Crea un dataset_info.json LOCAL (junto al YAML) con las rutas absolutas
     a los JSONL de train y validación — evita depender de la instalación pip.
  3. Genera el fichero YAML de configuración apuntando a ese dataset_info.json.
  4. Lanza el entrenamiento (llamafactory-cli train ...).
  5. Exporta el modelo fusionado (LoRA + base) a models/prensa_hispanica_merged/.

Prerrequisitos:
    pip install "llamafactory[torch,metrics,qwen]"

Uso:
    python scripts/04_train_lora.py
    python scripts/04_train_lora.py --solo-config      # genera YAML+JSON pero no entrena
    python scripts/04_train_lora.py --solo-exportar    # solo fusiona checkpoints existentes
    python scripts/04_train_lora.py --epochs 5 --lr 1e-5
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — VERIFICACIÓN DEL ENTORNO
# ══════════════════════════════════════════════════════════════════════════════

def verificar_llamafactory() -> None:
    cli = shutil.which("llamafactory-cli")
    if cli:
        return
    resultado = subprocess.run(
        [sys.executable, "-m", "llamafactory.cli", "--help"],
        capture_output=True,
    )
    if resultado.returncode == 0:
        return
    print(
        "ERROR: llamafactory-cli no encontrado.\n"
        'Instala con: pip install "llamafactory[torch,metrics,qwen]"'
    )
    sys.exit(1)


def ruta_posix(p: Path) -> str:
    """
    Devuelve la ruta como string con barras hacia adelante.
    LLaMA-Factory en Windows falla con barras invertidas en algunos contextos.
    """
    return p.resolve().as_posix()


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — DATASET_INFO.JSON LOCAL
# ══════════════════════════════════════════════════════════════════════════════

def crear_dataset_info_local(data_dir: Path) -> Path:
    """
    Crea un dataset_info.json en data_dir con las rutas absolutas a los JSONL.

    LLaMA-Factory acepta el parámetro 'dataset_dir' en el YAML para indicar
    dónde buscar este fichero, lo que evita modificar la instalación pip.

    Returns: ruta al fichero dataset_info.json creado.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    ruta_info = data_dir / "dataset_info.json"

    # LLaMA-Factory sharegpt espera por defecto "from"/"value".
    # Nuestro JSONL usa "role"/"content" (estándar OpenAI).
    # El bloque "tags" mapea nuestras claves sin reescribir los JSONL.
    entrada_dataset = {
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images":   "image",
        },
        "tags": {
            "role_tag":      "role",
            "content_tag":   "content",
            "user_tag":      "user",
            "assistant_tag": "assistant",
            "system_tag":    "system",
        },
    }

    info = {
        "prensa_hispanica_train": {
            "file_name": ruta_posix(config.JSONL_TRAIN),
            **entrada_dataset,
        },
        "prensa_hispanica_val": {
            "file_name": ruta_posix(config.JSONL_VALIDATION),
            **entrada_dataset,
        },
    }

    with open(ruta_info, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"  dataset_info.json creado en: {ruta_info}")
    return ruta_info


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — GENERACIÓN DEL YAML DE CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def generar_yaml(
    checkpoint_dir: Path,
    data_dir: Path,
    epochs: int,
    lr: float,
    bits: int,
) -> Path:
    """
    Genera el YAML de configuración para LLaMA-Factory.
    Incluye 'dataset_dir' apuntando al data_dir local con el dataset_info.json.

    Returns: ruta al fichero YAML generado.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ruta_yaml = checkpoint_dir / "ft_config.yaml"

    lora_targets = ",".join(config.LORA_TARGET_MODULES)

    # En Windows, YAML con rutas con backslash puede fallar → usamos posix
    yaml_content = f"""\
# ── Configuración QLoRA para olmOCR — prensa histórica hispanófona
# Generado por 04_train_lora.py — edita parámetros en config.py

model_name_or_path: {config.MODELO_BASE}

stage: sft
do_train: true
finetuning_type: lora

# ── LoRA ──────────────────────────────────────────────────────────────────────
lora_target: {lora_targets}
lora_rank: {config.LORA_RANK}
lora_alpha: {config.LORA_ALPHA}
lora_dropout: {config.LORA_DROPOUT}

# ── Datos ─────────────────────────────────────────────────────────────────────
# dataset_dir apunta al dataset_info.json local (no requiere modificar pip)
dataset_dir: {ruta_posix(data_dir)}
dataset: prensa_hispanica_train
eval_dataset: prensa_hispanica_val
template: {config.LLAMA_FACTORY_TEMPLATE}
cutoff_len: {config.MAX_SEQ_LEN}

# ── Imágenes ──────────────────────────────────────────────────────────────────
image_max_pixels: {config.IMAGE_MAX_PIXELS}

# ── Entrenamiento ─────────────────────────────────────────────────────────────
per_device_train_batch_size: {config.BATCH_SIZE_TRAIN}
gradient_accumulation_steps: {config.GRAD_ACCUM_STEPS}
num_train_epochs: {epochs}
learning_rate: {lr}
lr_scheduler_type: {config.LR_SCHEDULER}
warmup_ratio: {config.WARMUP_RATIO}
bf16: true

# ── Cuantización QLoRA ────────────────────────────────────────────────────────
quantization_bit: {bits}

# ── Optimizador ───────────────────────────────────────────────────────────────
optim: adamw_torch_fused

# ── Evaluación y guardado ─────────────────────────────────────────────────────
eval_strategy: steps
eval_steps: {config.EVAL_STEPS}
save_strategy: steps
save_steps: {config.SAVE_STEPS}
save_total_limit: {config.SAVE_TOTAL_LIMIT}
load_best_model_at_end: true
metric_for_best_model: eval_loss

# ── Salida ────────────────────────────────────────────────────────────────────
output_dir: {ruta_posix(checkpoint_dir)}
logging_steps: 10
"""
    with open(ruta_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"  YAML generado: {ruta_yaml}")
    return ruta_yaml


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def lanzar_entrenamiento(ruta_yaml: Path) -> None:
    print(f"\n{'═'*60}")
    print("  LANZANDO ENTRENAMIENTO")
    print(f"{'═'*60}")
    print(f"  Config : {ruta_yaml}")
    print(f"  Modelo : {config.MODELO_BASE}")
    print(f"  Salida : {config.CHECKPOINT_DIR}")
    print()

    cmd = ["llamafactory-cli", "train", str(ruta_yaml)]
    resultado = subprocess.run(cmd)

    if resultado.returncode != 0:
        print(f"\nERROR: entrenamiento terminó con código {resultado.returncode}")
        sys.exit(resultado.returncode)

    print("\nEntrenamiento completado.")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — EXPORTACIÓN DEL MODELO FUSIONADO
# ══════════════════════════════════════════════════════════════════════════════

def exportar_modelo(checkpoint_dir: Path, merged_dir: Path) -> None:
    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'═'*60}")
    print("  EXPORTANDO MODELO FUSIONADO")
    print(f"{'═'*60}")
    print(f"  Destino: {merged_dir}")
    print()

    cmd = [
        "llamafactory-cli", "export",
        "--model_name_or_path",   config.MODELO_BASE,
        "--adapter_name_or_path", ruta_posix(checkpoint_dir),
        "--template",             config.LLAMA_FACTORY_TEMPLATE,
        "--finetuning_type",      "lora",
        "--export_dir",           ruta_posix(merged_dir),
        "--export_size",          "4",
        "--export_legacy_format", "false",
    ]
    resultado = subprocess.run(cmd)

    if resultado.returncode != 0:
        print(f"\nERROR: exportación terminó con código {resultado.returncode}")
        sys.exit(resultado.returncode)

    print(f"\nModelo fusionado guardado en: {merged_dir}")
    print("Siguiente paso: python scripts/05_inference.py --imagen ruta/a/pagina.jpg")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning QLoRA de olmOCR para prensa histórica hispanófona.",
        epilog="Ejemplo: python scripts/04_train_lora.py --epochs 3 --bits 4",
    )
    parser.add_argument("--epochs",        type=int,   default=config.NUM_EPOCHS)
    parser.add_argument("--lr",            type=float, default=config.LEARNING_RATE)
    parser.add_argument("--bits",          type=int,   default=config.QUANTIZATION_BITS,
                        choices=[4, 8, 16])
    parser.add_argument("--solo-config",   action="store_true",
                        help="Solo generar YAML y dataset_info.json, sin entrenar")
    parser.add_argument("--solo-exportar", action="store_true",
                        help="Solo exportar modelo fusionado desde checkpoints existentes")
    args = parser.parse_args()

    checkpoint_dir = config.CHECKPOINT_DIR
    merged_dir     = config.MODELO_MERGED_DIR
    # El dataset_info.json local vive junto al YAML, dentro de checkpoints/
    data_dir       = checkpoint_dir / "data"

    # ── Verificar LLaMA-Factory ───────────────────────────────────────────────
    print("Verificando LLaMA-Factory …")
    verificar_llamafactory()
    print("  LLaMA-Factory encontrado.")

    if args.solo_exportar:
        exportar_modelo(checkpoint_dir, merged_dir)
        return

    # ── Verificar splits ──────────────────────────────────────────────────────
    for ruta in [config.JSONL_TRAIN, config.JSONL_VALIDATION]:
        if not ruta.exists():
            print(f"ERROR: {ruta} no existe. Ejecuta primero 02.")
            sys.exit(1)

    # ── Crear dataset_info.json local ─────────────────────────────────────────
    print("Creando dataset_info.json local …")
    crear_dataset_info_local(data_dir)

    # ── Generar YAML ──────────────────────────────────────────────────────────
    print("Generando YAML de configuración …")
    ruta_yaml = generar_yaml(checkpoint_dir, data_dir, args.epochs, args.lr, args.bits)

    if args.solo_config:
        print("\nFicheros generados. Entrena manualmente con:")
        print(f"  llamafactory-cli train {ruta_yaml}")
        return

    # ── Entrenar ──────────────────────────────────────────────────────────────
    lanzar_entrenamiento(ruta_yaml)

    # ── Exportar ──────────────────────────────────────────────────────────────
    exportar_modelo(checkpoint_dir, merged_dir)


if __name__ == "__main__":
    main()