#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_build_cache_layout.py

Lanza docLayout-YOLO sobre todo el corpus y guarda en dataset/layout_cache.json
el tipo de página de cada imagen (columnar, mixta, texto_corrido,
imagen_dominante o portada), indexado por stem. 03_build_dataset.py usa esa
caché para asignar el prompt de cada página. El modelo es el reentrenado con el
split compartido: <raíz del repo>/docLayout_fine-tuned/models/layout-model/
train/weights/best.pt.

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
    --salida RUTA         JSON de salida (dataset/layout_cache.json)
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
# CONFIGURACIÓN — ruta al modelo docLayout
# ══════════════════════════════════════════════════════════════════════════════

# Ruta absoluta derivada de la raíz del repo: no depende del directorio
# desde el que se lance el script.
DOCLAYOUT_MODEL = (
    config.ROOT.parent
    / "docLayout_fine-tuned" / "models" / "layout-model"
    / "train" / "weights" / "best.pt"
)

# Umbral de confianza por defecto (el que mejor funcionó en evaluación)
CONF_DEFAULT = 0.25

# Ruta de salida de la caché
CACHE_PATH = config.DATASET_DIR / "layout_cache.json"


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CARGA DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

def cargar_modelo(ruta: Path):
    try:
        from doclayout_yolo import YOLOv10
    except ImportError:
        print("ERROR: doclayout-yolo no instalado.\n"
              "  pip install doclayout-yolo")
        sys.exit(1)

    # Evita el error "'Conv' object has no attribute 'bn'" al fusionar
    # Conv+BN en los bloques g2l_crm. Solo afecta a la velocidad, no a los
    # resultados.
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
# SECCIÓN 2 — PREDICCIÓN Y ANÁLISIS GEOMÉTRICO
# ══════════════════════════════════════════════════════════════════════════════

def predecir_cajas(modelo, ruta_jpg: Path, conf: float) -> list[dict]:
    resultados = modelo.predict(
        source=str(ruta_jpg),
        conf=conf,
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


def _dimensiones_imagen(ruta_jpg: Path) -> tuple[float, float]:
    from PIL import Image
    with Image.open(ruta_jpg) as im:
        return float(im.width), float(im.height)


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


def clasificar_pagina(ruta_jpg: Path, modelo, conf: float, es_portada: bool) -> str:
    if es_portada:
        return "portada"

    ancho, _ = _dimensiones_imagen(ruta_jpg)
    cajas    = predecir_cajas(modelo, ruta_jpg, conf)

    if not cajas:
        return "texto_corrido"

    # Compatibilidad con modelo text/no-text y modelo de 4 clases
    CLASES_NOTEX = {"no-text", "graphic", "image", "table"}
    cajas_texto = [c for c in cajas if c["clase"] == "text"]
    cajas_notex = [c for c in cajas if c["clase"] in CLASES_NOTEX]

    n_texto  = len(cajas_texto)
    n_imagen = len(cajas_notex)
    total    = n_texto + n_imagen

    fraccion_imagen = n_imagen / total if total else 0.0

    if fraccion_imagen >= 0.50:
        return "imagen_dominante"

    num_columnas = _detectar_columnas(cajas_texto, ancho) if cajas_texto else 1

    if num_columnas <= 1:
        return "texto_corrido"

    tiene_cabecera = any(c["ancho"] > ancho * 0.60 for c in cajas_texto)

    if tiene_cabecera:
        return "mixta"

    return "columnar"


def _es_portada(stem: str) -> bool:
    s = stem.lower()
    return s.endswith("_p001") or s.endswith("_001")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — PROCESADO DEL CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def procesar_corpus(
    modelo,
    publicaciones: dict[str, str],
    corpus_dir: Path,
    conf: float,
    dry_run: bool,
) -> dict[str, str]:
    cache: dict[str, str] = {}
    stats_tipos: dict[str, int] = {}
    total_ok = 0
    total_err = 0
    colisiones: list[str] = []

    for nombre in publicaciones:
        dir_jpg = corpus_dir / nombre / "jpg"
        if not dir_jpg.exists():
            print(f"  [AVISO] {nombre}: carpeta jpg/ no encontrada, saltando.")
            continue

        jpgs = sorted(
            p for p in dir_jpg.glob("*")
            if p.suffix.lower() in (".jpg", ".jpeg")
        )
        if not jpgs:
            print(f"  [AVISO] {nombre}: sin imágenes JPG.")
            continue

        ok = 0
        err = 0
        tipos_pub: dict[str, int] = {}

        for jpg in jpgs:
            stem = jpg.stem
            portada = _es_portada(stem)
            if stem in cache:
                colisiones.append(f"{nombre}/{stem}")
            try:
                if dry_run:
                    tipo = "portada" if portada else "mixta"  # placeholder
                else:
                    tipo = clasificar_pagina(jpg, modelo, conf, portada)
                cache[stem] = tipo
                tipos_pub[tipo] = tipos_pub.get(tipo, 0) + 1
                stats_tipos[tipo] = stats_tipos.get(tipo, 0) + 1
                ok += 1
            except Exception as e:
                print(f"    [ERROR] {jpg.name}: {e}")
                err += 1

        resumen_tipos = "  ".join(f"{t}={n}" for t, n in sorted(tipos_pub.items()))
        print(f"  {nombre:<28}  {ok:>4} págs  {resumen_tipos}")
        total_ok  += ok
        total_err += err

    print(f"\n{'═'*60}")
    print(f"  RESUMEN GLOBAL")
    print(f"{'─'*60}")
    print(f"  Páginas clasificadas : {total_ok}")
    print(f"  Errores              : {total_err}")
    print(f"{'─'*60}")
    for tipo, n in sorted(stats_tipos.items(), key=lambda x: -x[1]):
        print(f"  {tipo:<22} {n:>4} páginas")
    print(f"{'═'*60}")

    if colisiones:
        print(f"\n  [AVISO] {len(colisiones)} stems repetidos entre publicaciones "
              f"(la caché y el split.json se indexan por stem):")
        for c in colisiones[:10]:
            print(f"    · {c}")

    return cache


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clasifica el tipo de página de todo el corpus usando docLayout-YOLO\n"
            "y guarda el resultado en dataset/layout_cache.json."
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
    parser.add_argument(
        "--modelo", default=None,
        help=f"Ruta al best.pt (default: {DOCLAYOUT_MODEL})",
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_DEFAULT,
        help=f"Umbral de confianza docLayout (default: {CONF_DEFAULT})",
    )
    parser.add_argument(
        "--publicacion", default=None,
        help="Procesar solo esta publicación (diagnóstico)",
    )
    parser.add_argument(
        "--corpus", default=None,
        help=f"Ruta al corpus (default: {config.CORPUS_DIR})",
    )
    parser.add_argument(
        "--salida", default=None,
        help=f"Ruta de salida del JSON (default: {CACHE_PATH})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Escanear el corpus sin lanzar docLayout ni escribir ficheros",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus) if args.corpus else config.CORPUS_DIR
    cache_path = Path(args.salida) if args.salida else CACHE_PATH
    ruta_modelo = Path(args.modelo) if args.modelo else DOCLAYOUT_MODEL

    if not corpus_dir.exists():
        print(f"ERROR: corpus no encontrado: {corpus_dir}")
        sys.exit(1)

    # Seleccionar publicaciones
    publicaciones = config.PUBLICACIONES
    if args.publicacion:
        if args.publicacion not in publicaciones:
            print(f"ERROR: '{args.publicacion}' no está en config.PUBLICACIONES")
            print(f"Disponibles: {list(publicaciones.keys())}")
            sys.exit(1)
        publicaciones = {args.publicacion: publicaciones[args.publicacion]}

    # Cargar modelo (no en dry-run)
    if args.dry_run:
        print("[DRY-RUN] Escaneando corpus sin lanzar docLayout …\n")
        modelo = None
    else:
        print("Cargando modelo docLayout …")
        modelo = cargar_modelo(ruta_modelo)
        print("  OK\n")

    print(f"Procesando {len(publicaciones)} publicaciones …\n")
    cache = procesar_corpus(
        modelo       = modelo,
        publicaciones = publicaciones,
        corpus_dir   = corpus_dir,
        conf         = args.conf,
        dry_run      = args.dry_run,
    )

    if args.dry_run:
        print("\n[DRY-RUN] No se ha escrito ningún fichero.")
        return

    # Guardar caché
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"\n  Caché guardada en: {cache_path}")
    print(f"  Siguiente paso: python scripts/03_build_dataset.py")


if __name__ == "__main__":
    main()