#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_train_lora.py

Lanza el fine-tuning QLoRA de olmOCR (Qwen2-VL-7B) con LLaMA-Factory sobre el
dataset de regiones (un registro por recorte). Antes de entrenar comprueba que
los JSONL de train y val respeten split.json y que los recortes existan, y se
niega a entrenar si models/checkpoints/ ya contiene un entrenamiento previo
(usa --reanudar para continuar). El test no se carga en ningún momento.
Ejecuta LLaMA-Factory con el Python actual (sys.executable), genera el
dataset_info.json local y el YAML (parámetros en config.py), entrena y exporta
el modelo fusionado (LoRA + base) a config.MODELO_MERGED_DIR.

Requisitos:
    python -m pip install "llamafactory[torch,metrics,qwen]"
    python scripts/03_build_dataset.py

Uso:
    python scripts/04_train_lora.py
    python scripts/04_train_lora.py --solo-config
    python scripts/04_train_lora.py --solo-exportar
    python scripts/04_train_lora.py --epochs 5 --lr 1e-5
    python scripts/04_train_lora.py --reanudar

Opciones:
    --epochs N        épocas de entrenamiento (config.NUM_EPOCHS)
    --lr X            learning rate (config.LEARNING_RATE)
    --bits 4|8|16     cuantización (config.QUANTIZATION_BITS)
    --solo-config     solo genera el YAML y dataset_info.json, sin entrenar
    --solo-exportar   solo fusiona los checkpoints existentes
    --reanudar        permite reanudar desde checkpoints existentes
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

SPLIT_PATH = Path(getattr(config, "SPLIT_PATH", config.ROOT.parent / "split.json"))

# Nombres simples (sin rutas ni caracteres especiales) para el dataset_info.json
NOMBRE_TRAIN = "olmocr_regiones_train"
NOMBRE_VAL   = "olmocr_regiones_val"

_LF_CMD = [sys.executable, "-c", "from llamafactory.cli import main; main()"]


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — VERIFICACIONES
# ══════════════════════════════════════════════════════════════════════════════

def ruta_posix(p: Path) -> str:
    return p.resolve().as_posix()


def verificar_llamafactory() -> None:
    resultado = subprocess.run(_LF_CMD + ["version"], capture_output=True, text=True)
    if resultado.returncode == 0:
        print("  LLaMA-Factory encontrado.")
        return
    print(
        "ERROR: LLaMA-Factory no se puede ejecutar con este Python.\n"
        f"  Python: {sys.executable}\n"
        f"  Detalle: {resultado.stderr.strip()[-300:]}\n"
        'Instala con: python -m pip install "llamafactory[torch,metrics,qwen]"'
    )
    sys.exit(1)


def verificar_checkpoints_previos(checkpoint_dir: Path) -> None:
    previos = list(checkpoint_dir.glob("checkpoint-*")) + \
              list(checkpoint_dir.glob("adapter_model*"))
    if previos:
        print(f"ERROR: {checkpoint_dir} ya contiene un entrenamiento previo "
              f"({previos[0].name}, …).\n"
              f"  Renómbralo o bórralo para entrenar desde cero, o usa --reanudar.")
        sys.exit(1)


def _leer_jsonl(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def verificar_dataset() -> tuple[int, int]:
    if not SPLIT_PATH.exists():
        print(f"ERROR: no existe {SPLIT_PATH} (make_split.py).")
        sys.exit(1)
    with open(SPLIT_PATH, encoding="utf-8") as f:
        split = json.load(f)

    datos = {"train": _leer_jsonl(config.JSONL_TRAIN),
             "val":   _leer_jsonl(config.JSONL_VALIDATION)}
    errores: list[str] = []
    stems = {}

    for nombre, regs in datos.items():
        stems[nombre] = {r["_meta"]["fuente"] for r in regs}
        mal = sorted(s for s in stems[nombre] if split.get(s, "train") != nombre)
        if mal:
            errores.append(f"{nombre}: {len(mal)} páginas que split.json asigna a otro split "
                           f"(p. ej. {mal[:3]})")
        sin_img = [r["image"] for r in regs if not Path(r["image"]).exists()]
        if sin_img:
            errores.append(f"{nombre}: {len(sin_img)} recortes no existen (p. ej. {sin_img[0]})")
        fuera = [r["image"] for r in regs if Path(r["image"]).parent.name != nombre]
        if fuera:
            errores.append(f"{nombre}: {len(fuera)} recortes fuera de crops/{nombre}/ "
                           f"(p. ej. {fuera[0]})")

    inter = sorted(stems["train"] & stems["val"])
    if inter:
        errores.append(f"{len(inter)} páginas en train y val a la vez (p. ej. {inter[:3]})")

    if errores:
        print("ERROR: el dataset no es coherente con split.json:")
        for e in errores:
            print(f"  · {e}")
        print("  Vuelve a ejecutar 03_build_dataset.py.")
        sys.exit(1)

    n_tr, n_va = len(datos["train"]), len(datos["val"])
    print(f"  Dataset coherente con {SPLIT_PATH.name}: "
          f"train {n_tr} regiones / {len(stems['train'])} páginas; "
          f"val {n_va} regiones / {len(stems['val'])} páginas.")
    return n_tr, n_va


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — DATASET_INFO.JSON LOCAL
# ══════════════════════════════════════════════════════════════════════════════

def crear_dataset_info_local(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    ruta_info = data_dir / "dataset_info.json"

    entrada = {
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "image"},
        "tags": {
            "role_tag": "role", "content_tag": "content",
            "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system",
        },
    }
    info = {
        NOMBRE_TRAIN: {"file_name": ruta_posix(config.JSONL_TRAIN),      **entrada},
        NOMBRE_VAL:   {"file_name": ruta_posix(config.JSONL_VALIDATION), **entrada},
    }
    with open(ruta_info, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"  dataset_info.json creado en: {ruta_info}")
    return ruta_info


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — YAML
# ══════════════════════════════════════════════════════════════════════════════

def generar_yaml(checkpoint_dir: Path, data_dir: Path, epochs: int, lr: float, bits: int) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ruta_yaml = checkpoint_dir / "ft_config.yaml"

    lora_targets     = ",".join(config.LORA_TARGET_MODULES)
    image_max_pixels = getattr(config, "CROP_IMAGE_MAX_PIXELS", config.IMAGE_MAX_PIXELS)
    batch_eval       = getattr(config, "BATCH_SIZE_EVAL", 2)

    print(f"  cutoff_len={config.MAX_SEQ_LEN}  image_max_pixels={image_max_pixels}  "
          f"eval_steps={config.EVAL_STEPS}")

    yaml_content = f"""\
# ── Configuración QLoRA para olmOCR — pipeline segmentado por regiones
# Dataset: UN REGISTRO POR REGIÓN (recorte de imagen), no por página completa.
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
dataset_dir: {ruta_posix(data_dir)}
dataset: {NOMBRE_TRAIN}
eval_dataset: {NOMBRE_VAL}
template: {config.LLAMA_FACTORY_TEMPLATE}
cutoff_len: {config.MAX_SEQ_LEN}

# ── Imágenes (recortes de región) ─────────────────────────────────────────────
image_max_pixels: {image_max_pixels}

# ── Entrenamiento ─────────────────────────────────────────────────────────────
per_device_train_batch_size: {config.BATCH_SIZE_TRAIN}
per_device_eval_batch_size: {batch_eval}
gradient_accumulation_steps: {config.GRAD_ACCUM_STEPS}
num_train_epochs: {epochs}
learning_rate: {lr}
lr_scheduler_type: {config.LR_SCHEDULER}
warmup_ratio: {config.WARMUP_RATIO}
bf16: true
seed: {config.SPLIT_SEED}

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
    print(f"  Salida : {config.CHECKPOINT_DIR}\n")

    resultado = subprocess.run(_LF_CMD + ["train", str(ruta_yaml)])
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
    print(f"  Destino: {merged_dir}\n")

    cmd = _LF_CMD + [
        "export",
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
    print("Siguiente paso: python scripts/05_inference.py --evaluar --jsonl")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning QLoRA de olmOCR — pipeline segmentado por regiones.",
        epilog="Ejemplo: python scripts/04_train_lora.py --epochs 3 --bits 4",
    )
    parser.add_argument("--epochs", type=int,   default=config.NUM_EPOCHS)
    parser.add_argument("--lr",     type=float, default=config.LEARNING_RATE)
    parser.add_argument("--bits",   type=int,   default=config.QUANTIZATION_BITS,
                        choices=[4, 8, 16])
    parser.add_argument("--solo-config",   action="store_true",
                        help="Solo generar YAML y dataset_info.json, sin entrenar")
    parser.add_argument("--solo-exportar", action="store_true",
                        help="Solo exportar el modelo fusionado desde los checkpoints existentes")
    parser.add_argument("--reanudar",      action="store_true",
                        help="Permitir reanudar desde checkpoints existentes")
    args = parser.parse_args()

    checkpoint_dir = config.CHECKPOINT_DIR
    merged_dir     = config.MODELO_MERGED_DIR
    data_dir       = checkpoint_dir / "data"

    print("Verificando LLaMA-Factory …")
    verificar_llamafactory()

    if args.solo_exportar:
        exportar_modelo(checkpoint_dir, merged_dir)
        return

    for ruta in (config.JSONL_TRAIN, config.JSONL_VALIDATION):
        if not ruta.exists():
            print(f"ERROR: {ruta} no existe. Ejecuta primero 03_build_dataset.py.")
            sys.exit(1)

    print("Verificando el dataset contra split.json …")
    verificar_dataset()

    if not args.reanudar and not args.solo_config:
        verificar_checkpoints_previos(checkpoint_dir)

    print("Creando dataset_info.json local …")
    crear_dataset_info_local(data_dir)

    print("Generando YAML de configuración …")
    ruta_yaml = generar_yaml(checkpoint_dir, data_dir, args.epochs, args.lr, args.bits)

    if args.solo_config:
        print("\nFicheros generados. Para entrenar manualmente:")
        print(f'  python -c "from llamafactory.cli import main; main()" train {ruta_yaml}')
        return

    lanzar_entrenamiento(ruta_yaml)
    exportar_modelo(checkpoint_dir, merged_dir)


if __name__ == "__main__":
    main()