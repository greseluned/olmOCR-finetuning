#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_finetune.py
========================

Realiza el fine-tuning de un modelo docLayout-YOLO preentrenado sobre
el corpus de prensa histórica hispanófona de GRESEL-UNED.

El objetivo es adaptar un modelo de detección de layout a las
características específicas de las publicaciones del corpus, utilizando
las imágenes y anotaciones preparadas en las fases anteriores.

FUNCIONAMIENTO:
    1. Comprueba que están instaladas las dependencias necesarias y
       que existe el modelo base preentrenado.

    2. Verifica que el dataset está correctamente preparado, localizando
       el archivo data.yaml y las imágenes de entrenamiento.

    3. Carga el modelo base docLayout-YOLO (YOLOv10) y configura los
       parámetros del entrenamiento: número de épocas, batch size,
       tamaño de imagen y dispositivo de ejecución.

    4. Ejecuta el fine-tuning utilizando los conjuntos de entrenamiento
       y validación. El conjunto de test se reserva para la evaluación
       posterior del modelo.

    5. Registra el consumo energético y las emisiones de CO2 asociadas
       al entrenamiento mediante CodeCarbon, siempre que esté disponible.

    6. Guarda los checkpoints del modelo, las métricas obtenidas durante
       el entrenamiento y un informe de emisiones.

    7. Muestra un resumen del entrenamiento, incluyendo la ruta al
       mejor checkpoint y el valor máximo de mAP50, cuando está
       disponible.

REQUISITOS:
    - Python 3.10 o superior.
    - doclayout-yolo.
    - CodeCarbon.
    - pandas.
    - PyTorch.
    - Dataset preparado previamente mediante los scripts anteriores.
    - Modelo base docLayout-YOLO descargado en docLayout/models/.

MODELO BASE:

    doclayout_yolo_docstructbench_imgsz1024.pt

    Disponible en:
    https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench

CONFIGURACIÓN PREDETERMINADA:
    - Épocas: 30.
    - Batch size: 2.
    - Dispositivo: cuda:0.
    - Tamaño de imagen: 1024 píxeles.
    - Modelo base: doclayout_yolo_docstructbench_imgsz1024.pt.

USO:

    Ejecutar el fine-tuning con los parámetros predeterminados:

        python scripts/03_finetune_doclayout.py

    Modificar el número de épocas y el batch size:

        python scripts/03_finetunet.py --epochs 50 --batch 4

    Utilizar otro dispositivo de ejecución (por ejemplo, CPU):

        python scripts/03_finetune.py --device cpu

    Modificar el tamaño de las imágenes:

        python scripts/03_finetune.py --img-size 1280

    Utilizar otro modelo base:

        python scripts/03_finetune.py --modelo-base otro_modelo.pt

    Seleccionar otro corpus:

        python scripts/03_finetune.py --corpus otro_corpus

    Comprobar que las dependencias, el modelo y el dataset están
    correctamente configurados sin iniciar el entrenamiento:

        python scripts/03_finetune.py --dry-run

SALIDA:

    docLayout/models/layout-model/
        train/
            weights/
                best.pt          <- Mejor checkpoint del entrenamiento.
                last.pt          <- Último checkpoint.
            results.csv          <- Métricas registradas por época.
        emissions.json           <- Consumo energético y emisiones de CO2.

    El archivo best.pt contiene el modelo con mejores resultados durante
    el entrenamiento y será el que se utilice posteriormente para
    realizar predicciones sobre nuevas páginas.

IMPORTANTE:
    Este script realiza el fine-tuning del modelo, pero no ejecuta
    la evaluación final sobre el conjunto de test ni realiza
    predicciones sobre el corpus completo.

    Una vez terminado el entrenamiento, se puede utilizar best.pt
    para detectar las regiones de las páginas y evaluar el rendimiento
    del modelo mediante los scripts correspondientes del pipeline.

    CodeCarbon es opcional: si no está disponible o se produce un error
    al inicializarlo, el entrenamiento continúa sin registrar emisiones.
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
    """Comprueba que están instaladas las dependencias; si no, termina con error."""
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
    """
    Verifica que existen el modelo base, el data.yaml y las imágenes de train.
 
    Comprueba:
        - El modelo base en models/<modelo_base>
        - El data.yaml en <corpus_name>/data.yaml
        - Al menos una imagen en <corpus_name>/images/train/
 
    Parámetros:
        modelo_base — nombre del archivo del modelo base
        corpus_name — nombre de la carpeta del corpus preparado
 
    Devuelve:
        (ruta_modelo_base, ruta_data_yaml, ruta_output)
    """
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
    """Cuenta las imágenes de cada split para el resumen."""
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
    """
    Inicia el seguimiento de emisiones con CodeCarbon.
 
    Devuelve:
        El tracker, o None si CodeCarbon no está disponible
    """
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
    """Detiene el tracker y convierte el CSV de emisiones a JSON."""
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
    """
    Lanza el fine-tuning de YOLOv10.
 
    Parámetros:
        ruta_modelo_base — checkpoint del modelo base
        ruta_yaml        — data.yaml del dataset
        ruta_output      — directorio de salida del entrenamiento
        epochs           — número de épocas
        batch            — tamaño de batch
        device           — dispositivo de ejecución (p. ej. 'cuda:0' o 'cpu')
        img_size         — tamaño de imagen en píxeles
 
    Devuelve:
        Ruta al mejor checkpoint (best.pt)
    """
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
    """Imprime la ruta al mejor checkpoint, el mejor mAP50 y los siguientes pasos."""
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