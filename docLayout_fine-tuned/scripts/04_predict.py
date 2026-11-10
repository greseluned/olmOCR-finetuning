#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_predict.py

Ejecuta un modelo docLayout-YOLO (baseline o fine-tuned) sobre las imágenes de un
split y guarda las regiones detectadas en pred_boxes_doclayout.json, que usa la
evaluación de layout. Las clases se normalizan al esquema del corpus. El
modelo baseline se espera en models/doclayout_yolo_docstructbench_imgsz1024.pt y
los fine-tuned en models/<nombre>/train/weights/best.pt.

Requisitos:
    pip install doclayout-yolo

Uso:
    python scripts/04_predict.py --split test
    python scripts/04_predict.py --split test --modelo layout-model
    python scripts/04_predict.py --split test --modelo layout-model --parchear

Opciones:
    --split train|val|test   split a procesar (test)
    --modelo NOMBRE          'baseline' o modelo fine-tuned en models/ (baseline)
    --conf X                 umbral de confianza (config.PRED_CONF)
    --device D               dispositivo (config.DEVICE)
    --parchear               copia pred_boxes_doclayout.json a pred_boxes.json

Salida:
    outputs/predicciones/pred_boxes_doclayout.json
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MAPEO DE CLASES
# ══════════════════════════════════════════════════════════════════════════════
 
# El modelo baseline de DocStructBench usa etiquetas en inglés.
# Si el modelo fine-tuned ya usa las clases del corpus, el diccionario puede
# dejarse vacío ({}): las clases se normalizan a minúsculas.
MAPEO_DOCLAYOUT: dict[str, str] = {
    "text":            "text",
    "title":           "text",
    "figure":          "image",
    "figure_caption":  "text",
    "table":           "table",
    "table_caption":   "text",
    "table_footnote":  "text",
    "isolate_formula": "maths",
    "formula_caption": "text",
    "image":           "image",
    "graphic":         "graphic",
    "separator":       "separator",
    "advert":          "advert",
}
 
 
def _normalizar_clase(nombre: str) -> str:
    return MAPEO_DOCLAYOUT.get(nombre, nombre.lower())
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CARGA DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
 
def _parche_pytorch26() -> None:
    import torch
    try:
        from doclayout_yolo.nn.tasks import YOLOv10DetectionModel
        torch.serialization.add_safe_globals([YOLOv10DetectionModel])
    except (ImportError, AttributeError):
        _original = torch.load
 
        def _patched(f, *args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original(f, *args, **kwargs)
 
        torch.load = _patched
        print("  [INFO] Parche torch.load aplicado (weights_only=False).")
 
 
def cargar_modelo(nombre_modelo: str):
    try:
        from doclayout_yolo import YOLOv10
    except ImportError:
        print("ERROR: doclayout-yolo no está instalado.\n"
              "  pip install doclayout-yolo")
        sys.exit(1)
 
    _parche_pytorch26()
 
    if nombre_modelo == "baseline":
        ruta = config.MODELS_DIR / config.BASE_MODEL
    else:
        ruta = config.MODELS_DIR / nombre_modelo / "train" / "weights" / "best.pt"
 
    if not ruta.exists():
        print(f"ERROR: No se encontró el modelo en {ruta}")
        if nombre_modelo == "baseline":
            print("  Descárgalo desde https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench")
        sys.exit(1)
 
    print(f"  Modelo: {ruta}")
    return YOLOv10(str(ruta))
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — LOCALIZACIÓN DE IMÁGENES DEL SPLIT
# ══════════════════════════════════════════════════════════════════════════════
 
def _obtener_imagenes_del_split(split: str) -> list[tuple[str, Path]]:
    dir_split = config.CORPUS_OUT_DIR / "images" / split
    if dir_split.exists():
        imagenes = []
        for img in sorted(dir_split.glob("*")):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                imagenes.append((img.stem, img))
        if imagenes:
            return imagenes
 
    # Alternativa: recorrer el corpus directamente
    print(f"  [AVISO] Carpeta {dir_split} no encontrada. Recorriendo corpus completo.")
    imagenes = []
    for pub_dir in sorted(config.CORPUS_DIR.iterdir()):
        jpg_dir = pub_dir / "jpg"
        if jpg_dir.is_dir():
            for img in sorted(jpg_dir.glob("*.jpg")):
                imagenes.append((img.stem, img))
    return imagenes
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — PREDICCIÓN
# ══════════════════════════════════════════════════════════════════════════════
 
def predecir_imagen(
    ruta_imagen: Path,
    modelo,
    conf: float,
    device: str,
) -> list[dict]:
    resultados = modelo.predict(
        str(ruta_imagen),
        imgsz=config.IMG_SIZE,
        conf=conf,
        device=device,
        verbose=False,
    )
    det = resultados[0]
    boxes = det.boxes
 
    detecciones = []
    for xyxy, conf_score, cls_id in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.conf.cpu().numpy(),
        boxes.cls.cpu().numpy(),
    ):
        nombre_clase = _normalizar_clase(det.names[int(cls_id)])
        detecciones.append({
            "class_name": nombre_clase,
            "xyxy":       xyxy.tolist(),
            "confidence": round(float(conf_score), 4),
        })
    return detecciones
 
 
def predecir_split(
    imagenes: list[tuple[str, Path]],
    modelo,
    conf: float,
    device: str,
) -> dict[str, list[dict]]:
    pred_boxes: dict[str, list] = {}
    total = len(imagenes)
 
    for i, (stem, ruta) in enumerate(imagenes, 1):
        print(f"  [{i:>4}/{total}] {ruta.name} … ", end="", flush=True)
        try:
            dets = predecir_imagen(ruta, modelo, conf, device)
            pred_boxes[stem] = dets
            print(f"{len(dets)} regiones")
        except Exception as e:
            print(f"ERROR: {e}")
            pred_boxes[stem] = []
 
    return pred_boxes
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — GUARDADO Y RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
 
def guardar_pred_boxes(pred_boxes: dict[str, list], dir_eval: Path) -> Path:
    dir_eval.mkdir(parents=True, exist_ok=True)
    ruta = dir_eval / "pred_boxes_doclayout.json"
    ruta.write_text(
        json.dumps(pred_boxes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  pred_boxes guardado en: {ruta}")
    return ruta
 
 
def _estadisticas(pred_boxes: dict[str, list]) -> None:
    conteo: dict[str, int] = {}
    for dets in pred_boxes.values():
        for d in dets:
            conteo[d["class_name"]] = conteo.get(d["class_name"], 0) + 1
 
    print("\n  Regiones detectadas por clase:")
    for cls, n in sorted(conteo.items(), key=lambda x: -x[1]):
        print(f"    {cls:<20} {n:>5}")
    print(f"    {'TOTAL':<20} {sum(conteo.values()):>5}")
 
 
def parchear_evaluate_fpp(dir_eval: Path) -> None:
    src = dir_eval / "pred_boxes_doclayout.json"
    dst = dir_eval / "pred_boxes.json"
    if src.exists():
        shutil.copy(src, dst)
        print(f"  pred_boxes.json actualizado con predicciones YOLO reales.")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — CLI
# ══════════════════════════════════════════════════════════════════════════════
 
def main():
    parser = argparse.ArgumentParser(
        description="Predicción de layout con DocLayout-YOLO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/04_predict.py --split test
  python scripts/04_predict.py --split test --modelo layout-model
  python scripts/04_predict.py --split test --modelo layout-model --parchear
        """,
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Split a procesar (default: test)",
    )
    parser.add_argument(
        "--modelo", default="baseline",
        help="'baseline' o nombre del modelo fine-tuned en models/ (default: baseline)",
    )
    parser.add_argument(
        "--conf", type=float, default=config.PRED_CONF,
        help=f"Umbral de confianza (default: {config.PRED_CONF})",
    )
    parser.add_argument(
        "--device", default=config.DEVICE,
        help=f"Dispositivo (default: {config.DEVICE})",
    )
    parser.add_argument(
        "--parchear", action="store_true",
        help="Copiar pred_boxes_doclayout.json → pred_boxes.json para olmOCR-finetuning/evaluate.py",
    )
    args = parser.parse_args()
 
    dir_eval = config.EVAL_DIR
 
    # ── Modelo ────────────────────────────────────────────────────────────────
    print(f"Cargando modelo '{args.modelo}' …")
    modelo = cargar_modelo(args.modelo)
    print("  OK\n")
 
    # ── Imágenes del split ────────────────────────────────────────────────────
    print(f"Buscando imágenes del split '{args.split}' …")
    imagenes = _obtener_imagenes_del_split(args.split)
    if not imagenes:
        print("ERROR: No se encontraron imágenes.")
        sys.exit(1)
    print(f"  {len(imagenes)} imágenes\n")
 
    # ── Predicción ────────────────────────────────────────────────────────────
    print("Ejecutando DocLayout-YOLO …")
    pred_boxes = predecir_split(imagenes, modelo, args.conf, args.device)
 
    # ── Guardado ──────────────────────────────────────────────────────────────
    guardar_pred_boxes(pred_boxes, dir_eval)
    _estadisticas(pred_boxes)
 
    if args.parchear:
        parchear_evaluate_fpp(dir_eval)
 
    print(
        "\nSiguiente paso:\n"
        "  python olmOCR-finetuning/evaluate.py"
    )
 
 
if __name__ == "__main__":
    main()