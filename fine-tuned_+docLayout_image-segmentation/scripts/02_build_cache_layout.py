#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_build_cache_layout.py

Lanza docLayout-YOLO sobre todo el corpus y guarda en
dataset/layout_regions_cache.json las regiones detectadas de cada página (id,
tipo text|graphic, orden de lectura y bbox en píxeles), indexadas por stem.
03_build_dataset.py la usa para recortar las regiones de train y val. El orden
de lectura se infiere por bandas horizontales (tolerancia de 20 px) y, dentro
de cada banda, de izquierda a derecha. El modelo es el reentrenado con el split
compartido: <raíz del repo>/docLayout_fine-tuned/models/layout-model/train/
weights/best.pt. Si se reentrena docLayout hay que regenerar esta caché, el
dataset (03) y el entrenamiento (04).

Requisitos:
    pip install doclayout-yolo Pillow

Uso:
    python scripts/02_build_cache_layout.py
    python scripts/02_build_cache_layout.py --publicacion fémina
    python scripts/02_build_cache_layout.py --conf 0.30
    python scripts/02_build_cache_layout.py --dry-run

Opciones:
    --modelo RUTA         best.pt de docLayout
    --conf X              umbral de confianza (0.25)
    --publicacion NOMBRE  procesa solo esa publicación
    --corpus RUTA         directorio del corpus (config.CORPUS_DIR)
    --salida RUTA         JSON de salida (dataset/layout_regions_cache.json)
    --dry-run             escanea el corpus sin lanzar docLayout ni escribir nada
"""

import os
# PyTorch >= 2.6 rechaza por defecto los checkpoints de docLayout-YOLO
# (weights_only=True). Los pesos son propios, así que se permite la carga.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

DOCLAYOUT_MODEL = (
    config.ROOT.parent
    / "docLayout_fine-tuned" / "models" / "layout-model"
    / "train" / "weights" / "best.pt"
)

CONF_DEFAULT = 0.25
CACHE_PATH   = config.DATASET_DIR / "layout_regions_cache.json"

# Clases de docLayout que se tratan como regiones gráficas (el resto → text)
CLASES_GRAPHIC = {"no-text", "graphic", "image", "table"}


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CARGA DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

def cargar_modelo(ruta: Path):
    try:
        from doclayout_yolo import YOLOv10
    except ImportError:
        print("ERROR: doclayout-yolo no instalado.\n  pip install doclayout-yolo")
        sys.exit(1)

    # Evita "'Conv' object has no attribute 'bn'" al fusionar Conv+BN en los
    # bloques g2l_crm. Solo afecta a la velocidad, no a los resultados.
    try:
        from doclayout_yolo.nn.tasks import BaseModel
        BaseModel.fuse = lambda self, verbose=True: self
    except Exception:
        pass

    if not ruta.exists():
        print(f"ERROR: Modelo no encontrado: {ruta}")
        sys.exit(1)

    fecha = datetime.datetime.fromtimestamp(ruta.stat().st_mtime)
    print(f"  Cargando modelo: {ruta}")
    print(f"  Última modificación del modelo: {fecha:%Y-%m-%d %H:%M}")
    return YOLOv10(str(ruta))


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — PREDICCIÓN Y READING ORDER
# ══════════════════════════════════════════════════════════════════════════════

def _dimensiones_imagen(ruta_jpg: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(ruta_jpg) as im:
        return im.width, im.height


def predecir_cajas_raw(modelo, ruta_jpg: Path, conf: float) -> list[dict]:
    resultados = modelo.predict(source=str(ruta_jpg), conf=conf, verbose=False)
    cajas = []
    for r in resultados:
        nombres = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cajas.append({
                "clase": nombres[cls_id],
                "conf":  float(box.conf[0]),
                "x0": x1, "y0": y1, "x1": x2, "y1_": y2,   # xyxy
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "ancho": x2 - x1,
            })
    return cajas


def _inferir_reading_order(cajas: list[dict], ancho_pagina: int) -> list[dict]:
    if not cajas:
        return []

    TOLERANCIA_Y = 20
    cajas_sorted = sorted(cajas, key=lambda c: c["cy"])

    bandas: list[list[dict]] = []
    banda_actual: list[dict] = []
    for caja in cajas_sorted:
        if not banda_actual:
            banda_actual.append(caja)
        elif abs(caja["cy"] - banda_actual[0]["cy"]) <= TOLERANCIA_Y:
            banda_actual.append(caja)
        else:
            bandas.append(banda_actual)
            banda_actual = [caja]
    if banda_actual:
        bandas.append(banda_actual)

    resultado: list[dict] = []
    orden = 0
    for banda in bandas:
        for caja in sorted(banda, key=lambda c: c["cx"]):
            caja["orden"] = orden
            resultado.append(caja)
            orden += 1
    return resultado


def extraer_regiones_pagina(ruta_jpg: Path, modelo, conf: float) -> list[dict]:
    ancho, alto = _dimensiones_imagen(ruta_jpg)
    cajas = predecir_cajas_raw(modelo, ruta_jpg, conf)
    if not cajas:
        return []

    cajas_ordenadas = _inferir_reading_order(cajas, ancho)

    regiones: list[dict] = []
    for i, caja in enumerate(cajas_ordenadas):
        tipo = "graphic" if caja["clase"] in CLASES_GRAPHIC else "text"

        x0 = max(0, int(caja["x0"]))
        y0 = max(0, int(caja["y0"]))
        x1 = min(ancho, int(caja["x1"]))
        y1 = min(alto,  int(caja["y1_"]))
        if x1 <= x0 or y1 <= y0:
            continue

        regiones.append({
            "id":    f"r_{i:03d}",
            "tipo":  tipo,
            "orden": caja["orden"],
            "bbox":  [x0, y0, x1, y1],
        })
    return regiones


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — PROCESADO DEL CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def procesar_corpus(
    modelo,
    publicaciones: dict[str, str],
    corpus_dir: Path,
    conf: float,
    dry_run: bool,
) -> dict[str, list[dict]]:
    cache: dict[str, list[dict]] = {}
    total_ok = total_err = 0
    colisiones: list[str] = []
    sin_regiones: list[str] = []

    for nombre in publicaciones:
        dir_jpg = corpus_dir / nombre / "jpg"
        if not dir_jpg.exists():
            print(f"  [AVISO] {nombre}: carpeta jpg/ no encontrada, saltando.")
            continue

        jpgs = sorted(p for p in dir_jpg.glob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
        if not jpgs:
            print(f"  [AVISO] {nombre}: sin imágenes JPG.")
            continue

        ok = err = n_regiones_total = 0
        for jpg in jpgs:
            stem = jpg.stem
            if stem in cache:
                colisiones.append(f"{nombre}/{stem}")
            try:
                regiones = [] if dry_run else extraer_regiones_pagina(jpg, modelo, conf)
                cache[stem] = regiones
                n_regiones_total += len(regiones)
                if not regiones and not dry_run:
                    sin_regiones.append(stem)
                ok += 1
            except Exception as e:
                print(f"    [ERROR] {jpg.name}: {e}")
                err += 1

        promedio = n_regiones_total / ok if ok else 0
        print(f"  {nombre:<28}  {ok:>4} págs  {n_regiones_total:>5} regiones  ({promedio:.1f} reg/pág)")
        total_ok  += ok
        total_err += err

    print(f"\n{'═'*60}")
    print(f"  RESUMEN GLOBAL")
    print(f"{'─'*60}")
    print(f"  Páginas procesadas : {total_ok}")
    print(f"  Errores            : {total_err}")
    print(f"  Regiones totales   : {sum(len(v) for v in cache.values())}")
    n_text    = sum(1 for v in cache.values() for r in v if r["tipo"] == "text")
    n_graphic = sum(1 for v in cache.values() for r in v if r["tipo"] == "graphic")
    print(f"    text             : {n_text}")
    print(f"    graphic          : {n_graphic}")
    print(f"{'═'*60}")

    if sin_regiones:
        print(f"\n  [AVISO] {len(sin_regiones)} páginas sin ninguna región detectada "
              f"(no aportarán registros a train/val): "
              + ", ".join(sin_regiones[:5]) + (" …" if len(sin_regiones) > 5 else ""))
    if colisiones:
        print(f"\n  [AVISO] {len(colisiones)} stems repetidos entre publicaciones "
              f"(la caché y split.json se indexan por stem): "
              + ", ".join(colisiones[:5]))
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extrae regiones de layout con docLayout-YOLO y las guarda\n"
            "en dataset/layout_regions_cache.json para el pipeline segmentado."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/02_build_cache_layout.py
  python scripts/02_build_cache_layout.py --publicacion fémina
  python scripts/02_build_cache_layout.py --conf 0.30
  python scripts/02_build_cache_layout.py --dry-run
        """,
    )
    parser.add_argument("--modelo",      default=None)
    parser.add_argument("--conf",        type=float, default=CONF_DEFAULT)
    parser.add_argument("--publicacion", default=None)
    parser.add_argument("--corpus",      default=None)
    parser.add_argument("--salida",      default=None)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    corpus_dir  = Path(args.corpus)  if args.corpus  else config.CORPUS_DIR
    cache_path  = Path(args.salida)  if args.salida  else CACHE_PATH
    ruta_modelo = Path(args.modelo)  if args.modelo  else DOCLAYOUT_MODEL

    if not corpus_dir.exists():
        print(f"ERROR: corpus no encontrado: {corpus_dir}")
        sys.exit(1)

    publicaciones = config.PUBLICACIONES
    if args.publicacion:
        if args.publicacion not in publicaciones:
            print(f"ERROR: '{args.publicacion}' no está en config.PUBLICACIONES")
            print(f"Disponibles: {list(publicaciones.keys())}")
            sys.exit(1)
        publicaciones = {args.publicacion: publicaciones[args.publicacion]}

    if args.dry_run:
        print("[DRY-RUN] Escaneando corpus sin lanzar docLayout …\n")
        modelo = None
    else:
        print("Cargando modelo docLayout …")
        modelo = cargar_modelo(ruta_modelo)
        print("  OK\n")

    print(f"Procesando {len(publicaciones)} publicaciones …\n")
    cache = procesar_corpus(
        modelo        = modelo,
        publicaciones = publicaciones,
        corpus_dir    = corpus_dir,
        conf          = args.conf,
        dry_run       = args.dry_run,
    )

    if args.dry_run:
        print("\n[DRY-RUN] No se ha escrito ningún fichero.")
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"\n  Caché guardada en: {cache_path}")
    print(f"  Siguiente paso: python scripts/03_build_dataset.py")


if __name__ == "__main__":
    main()