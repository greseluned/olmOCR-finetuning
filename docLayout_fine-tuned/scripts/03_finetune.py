#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_finetune.py

Hace el fine-tuning de docLayout-YOLO (YOLOv10) preentrenado sobre el corpus
preparado por 01_pagexml_to_yolo.py y 02_prepare_dataset.py. Comprueba
dependencias, modelo base y dataset, entrena con train y val (el test queda
reservado), registra el consumo energético con CodeCarbon si está disponible
(es opcional) y guarda los checkpoints, las métricas y el informe de emisiones.

Requisitos:
    doclayout-yolo, codecarbon, pandas, PyTorch y el modelo base
    doclayout_yolo_docstructbench_imgsz1024.pt en models/ (disponible en
    https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench).

Uso:
    python scripts/03_finetune.py
    python scripts/03_finetune.py --epochs 50 --batch 4
    python scripts/03_finetune.py --device cpu
    python scripts/03_finetune.py --dry-run

Opciones:
    --epochs N        épocas (30)
    --batch N         tamaño de batch (2)
    --device D        dispositivo (cuda:0)
    --img-size N      tamaño de imagen en píxeles (1024)
    --modelo-base F   modelo base en models/ (doclayout_yolo_docstructbench_imgsz1024.pt)
    --corpus NOMBRE   corpus preparado (data-regiones)
    --dry-run         verifica dependencias, modelo y dataset sin entrenar

Salida:
    models/layout-model/train/weights/{best.pt, last.pt},
    models/layout-model/train/results.csv y models/layout-model/emissions.json.
    best.pt es el modelo que usan los pasos posteriores.
"""

import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
 
import argparse
import json
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
 
 
# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
 
# Nombre del corpus generado por 01_pagexml_to_yolo.py + 02_prepare_dataset.py
CORPUS_NAME = "data-regiones"
 
# Directorio raíz de DocLayout (docLayout_fine-tuned/)
DOCLAYOUT_DIR = Path(__file__).resolve().parents[1]
 
# Directorio de modelos
MODELS_DIR = DOCLAYOUT_DIR / "models"
 
# Nombre del modelo de salida
OUTPUT_MODEL_NAME = "layout-model"
 
# Parámetros de entrenamiento por defecto
DEFAULT_EPOCHS     = 30
DEFAULT_BATCH      = 2
DEFAULT_DEVICE     = "cuda:0"
DEFAULT_IMG_SIZE   = 1024
DEFAULT_BASE_MODEL = "doclayout_yolo_docstructbench_imgsz1024.pt"
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — VERIFICACIÓN DEL ENTORNO
# ══════════════════════════════════════════════════════════════════════════════
 
def verificar_dependencias() -> None:
    faltantes = []
    for pkg in ["doclayout_yolo", "codecarbon", "pandas"]:
        try:
            __import__(pkg)
        except ImportError:
            faltantes.append(pkg)
    if faltantes:
        print(
            "ERROR: Faltan dependencias:\n"
            f"  pip install {' '.join(faltantes)}"
        )
        sys.exit(1)
 
 
def verificar_estructura(
    modelo_base: str,
    corpus_name: str,
) -> tuple[Path, Path, Path]:
    ruta_modelo = MODELS_DIR / modelo_base
    if not ruta_modelo.exists():
        print(
            f"ERROR: Modelo base no encontrado: {ruta_modelo}\n"
            f"  Descárgalo desde:\n"
            f"  https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench\n"
            f"  y colócalo en {MODELS_DIR}/"
        )
        sys.exit(1)
 
    corpus_dir = DOCLAYOUT_DIR / corpus_name
    ruta_yaml  = corpus_dir / "data.yaml"
    if not ruta_yaml.exists():
        print(
            f"ERROR: data.yaml no encontrado: {ruta_yaml}\n"
            f"  Ejecuta primero:\n"
            f"  python scripts/02_prepare_dataset.py --corpus {corpus_name}"
        )
        sys.exit(1)
 
    dir_train_imgs = corpus_dir / "images" / "train"
    if not dir_train_imgs.exists() or not any(dir_train_imgs.iterdir()):
        print(
            f"ERROR: No se encontraron imágenes de entrenamiento en {dir_train_imgs}\n"
            f"  Ejecuta primero:\n"
            f"  python scripts/01_pagexml_to_yolo.py\n"
            f"  python scripts/02_prepare_dataset.py --corpus {corpus_name}"
        )
        sys.exit(1)
 
    ruta_output = MODELS_DIR / OUTPUT_MODEL_NAME
    return ruta_modelo, ruta_yaml, ruta_output
 
 
def _contar_imagenes(corpus_name: str) -> dict[str, int]:
    corpus_dir = DOCLAYOUT_DIR / corpus_name
    conteo = {}
    for split in ["train", "val", "test"]:
        d = corpus_dir / "images" / split
        if d.exists():
            conteo[split] = sum(1 for f in d.iterdir()
                                if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        else:
            conteo[split] = 0
    return conteo
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — EMISIONES DE CARBONO
# ══════════════════════════════════════════════════════════════════════════════
 
def iniciar_tracker(ruta_emisiones: Path):
    try:
        import logging
        logging.getLogger("codecarbon").disabled = True
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(output_file=str(ruta_emisiones))
        tracker.start()
        return tracker
    except Exception as e:
        print(f"  [AVISO] CodeCarbon no disponible: {e}. Continuando sin tracking.")
        return None
 
 
def detener_tracker(tracker, ruta_emisiones: Path) -> None:
    if tracker is None:
        return
    try:
        tracker.stop()
        import pandas as pd
        df = pd.read_csv(ruta_emisiones)
        if df.empty:
            return
        row = df.iloc[-1]
        datos = {
            "duration_s":      str(row.get("duration",        "N/A")),
            "emissions_kgCO2": str(row.get("emissions",       "N/A")),
            "energy_kWh":      str(row.get("energy_consumed", "N/A")),
            "cpu_energy_kWh":  str(row.get("cpu_energy",      "N/A")),
            "gpu_energy_kWh":  str(row.get("gpu_energy",      "N/A")),
            "ram_energy_kWh":  str(row.get("ram_energy",      "N/A")),
            "cpu_model":       str(row.get("cpu_model",       "N/A")),
            "gpu_model":       str(row.get("gpu_model",       "N/A")),
            "cpu_count":       str(row.get("cpu_count",       "N/A")),
            "gpu_count":       str(row.get("gpu_count",       "N/A")),
            "ram_total_GB":    str(row.get("ram_total_size",  "N/A")),
        }
        ruta_emisiones.write_text(
            json.dumps(datos, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        print(f"  Emisiones guardadas en: {ruta_emisiones}")
    except Exception as e:
        print(f"  [AVISO] No se pudo guardar el informe de emisiones: {e}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
 
def entrenar(
    ruta_modelo_base: Path,
    ruta_yaml: Path,
    ruta_output: Path,
    epochs: int,
    batch: int,
    device: str,
    img_size: int,
) -> Path:
    import os
    # YOLO usa ULTRALYTICS_DATASETS_DIR para resolver rutas relativas en data.yaml
    os.environ["ULTRALYTICS_DATASETS_DIR"] = str(Path(".").resolve())
 
    from doclayout_yolo import YOLOv10
 
    # ── Parche PyTorch 2.6 ────────────────────────────────────────────────────
    # En PyTorch 2.6 el valor por defecto de weights_only cambió a True,
    # lo que impide cargar checkpoints con clases personalizadas de
    # doclayout_yolo durante final_eval. Se añaden los globales necesarios a
    # la lista permitida.
    import torch
    try:
        from doclayout_yolo.nn.tasks import YOLOv10DetectionModel
        torch.serialization.add_safe_globals([YOLOv10DetectionModel])
    except (ImportError, AttributeError):
        # Si la versión instalada no tiene este módulo exacto, se parchea
        # torch.load para que use weights_only=False como antes.
        _torch_load_original = torch.load
 
        def _torch_load_patched(f, *args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _torch_load_original(f, *args, **kwargs)
 
        torch.load = _torch_load_patched
        print("  [INFO] Parche torch.load aplicado (weights_only=False).")
 
    ruta_output.mkdir(parents=True, exist_ok=True)
    ruta_emisiones = ruta_output / "emissions.json"
 
    print(f"\n{'═'*60}")
    print("  ENTRENAMIENTO DocLayout-YOLO")
    print(f"{'═'*60}")
    print(f"  Modelo base : {ruta_modelo_base}")
    print(f"  Dataset     : {ruta_yaml}")
    print(f"  Épocas      : {epochs}")
    print(f"  Batch size  : {batch}")
    print(f"  Imagen (px) : {img_size}")
    print(f"  Dispositivo : {device}")
    print(f"  Salida      : {ruta_output}")
    print(f"{'═'*60}\n")
 
    # Iniciar el seguimiento de emisiones
    tracker = iniciar_tracker(ruta_emisiones)
 
    # Cargar el modelo base
    modelo = YOLOv10(str(ruta_modelo_base))
 
    # Entrenar
    modelo.train(
        data=str(ruta_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        project=str(ruta_output),
        name="train",
        exist_ok=True,
        device=device,
        amp=False,      # AMP desactivado: más estable en corpus pequeños
        val=True,
    )
 
    # Detener el seguimiento de emisiones
    detener_tracker(tracker, ruta_emisiones)
 
    ruta_best = ruta_output / "train" / "weights" / "best.pt"
    return ruta_best
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — RESUMEN POST-ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
 
def imprimir_resultado(ruta_best: Path, ruta_output: Path) -> None:
    print(f"\n{'═'*60}")
    print("  ENTRENAMIENTO COMPLETADO")
    print(f"{'═'*60}")
 
    if ruta_best.exists():
        print(f"  Mejor modelo : {ruta_best}")
    else:
        print("  [AVISO] best.pt no encontrado — revisa los logs de entrenamiento.")
 
    ruta_results = ruta_output / "train" / "results.csv"
    if ruta_results.exists():
        try:
            import pandas as pd
            df = pd.read_csv(ruta_results)
            # La columna de mAP50 puede llamarse distinto según la versión
            map_cols = [c for c in df.columns if "map50" in c.lower() and "95" not in c.lower()]
            if map_cols:
                mejor_map = df[map_cols[0]].max()
                print(f"  Mejor mAP50  : {mejor_map:.4f}")
        except Exception:
            pass
        print(f"  Métricas     : {ruta_results}")
 
    print(f"\n  Siguiente paso:")
    print(f"  python scripts/04_predict.py --modelo {OUTPUT_MODEL_NAME} --split test --parchear")
    print(f"  python olmOCR-finetuning/evaluate.py.py --solo-layout")
    print(f"{'═'*60}\n")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — CLI
# ══════════════════════════════════════════════════════════════════════════════
 
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tuning de DocLayout-YOLO para prensa histórica hispanófona.\n"
            "Requiere haber ejecutado antes 01_pagexml_to_yolo.py y 02_prepare_dataset.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/03_finetune.py
  python scripts/03_finetune.py --epochs 50 --batch 4
  python scripts/03_finetune.py --device cpu
  python scripts/03_finetune.py --dry-run
        """,
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Número de épocas de entrenamiento (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch", type=int, default=DEFAULT_BATCH,
        help=f"Batch size (default: {DEFAULT_BATCH}; sube a 4-8 si tu GPU lo permite)",
    )
    parser.add_argument(
        "--device", default=DEFAULT_DEVICE,
        help=f"Dispositivo (default: {DEFAULT_DEVICE}; usa 'cpu' si no hay GPU)",
    )
    parser.add_argument(
        "--img-size", type=int, default=DEFAULT_IMG_SIZE,
        help=f"Tamaño de imagen de entrenamiento en píxeles (default: {DEFAULT_IMG_SIZE})",
    )
    parser.add_argument(
        "--modelo-base", default=DEFAULT_BASE_MODEL,
        help=f"Nombre del modelo base en models/ (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--corpus", default=CORPUS_NAME,
        help=f"Nombre del corpus preparado (default: {CORPUS_NAME})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verificar configuración sin entrenar",
    )
    args = parser.parse_args()
 
    # ── Dependencias ──────────────────────────────────────────────────────────
    print("Verificando dependencias …")
    verificar_dependencias()
    print("  OK\n")
 
    # ── Estructura del dataset ────────────────────────────────────────────────
    print("Verificando estructura del dataset …")
    ruta_modelo_base, ruta_yaml, ruta_output = verificar_estructura(
        args.modelo_base, args.corpus
    )
 
    conteo = _contar_imagenes(args.corpus)
    print(f"  train: {conteo['train']} imágenes")
    print(f"  val  : {conteo['val']} imágenes")
    print(f"  test : {conteo['test']} imágenes")
    print("  OK\n")
 
    if args.dry_run:
        print("[DRY-RUN] Configuración verificada. Todo listo para entrenar.")
        print(f"  Ejecuta sin --dry-run para iniciar el entrenamiento.")
        return
 
    # ── Entrenamiento ─────────────────────────────────────────────────────────
    ruta_best = entrenar(
        ruta_modelo_base=ruta_modelo_base,
        ruta_yaml=ruta_yaml,
        ruta_output=ruta_output,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        img_size=args.img_size,
    )
 
    # ── Resumen ───────────────────────────────────────────────────────────────
    imprimir_resultado(ruta_best, ruta_output)
 
 
if __name__ == "__main__":
    main()