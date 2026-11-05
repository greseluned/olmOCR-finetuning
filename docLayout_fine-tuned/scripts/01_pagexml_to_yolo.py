#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_pagexml_to_yolo.py
=====================

Convierte las anotaciones de regiones de los archivos PAGE-XML del
corpus de prensa histórica de GRESEL-UNED a formato YOLO, para preparar
un dataset con el que entrenar un modelo de detección de layout
(docLayout-YOLO).

El script utiliza las coordenadas de las regiones anotadas en los
PAGE-XML para generar bounding boxes normalizadas y asociarlas a las
clases correspondientes (texto, imágenes, gráficos, etc.).

FUNCIONAMIENTO:
    1. Recorre las publicaciones del corpus y localiza los pares de
       imágenes JPG y archivos PAGE-XML correspondientes.

    2. Extrae las regiones anotadas en los XML, identifica sus tipos
       y obtiene sus coordenadas.

    3. Detecta automáticamente las clases presentes en el corpus.
       También permite seleccionar manualmente las clases que se
       desean incluir o excluir aquellas con pocas anotaciones.

    4. Aplica el mismo split train/val/test utilizado por el pipeline
       de olmOCR, garantizando que ambos modelos trabajen con las
       mismas páginas en los conjuntos de entrenamiento, validación
       y prueba.

    5. Convierte las coordenadas de las regiones al formato YOLO y
       genera la estructura de carpetas necesaria para preparar
       posteriormente el dataset de entrenamiento.

    6. Guarda un informe con las clases detectadas, las clases
       excluidas y las estadísticas de distribución del dataset.

REQUISITOS:
    - Python 3.10 o superior.
    - Pillow.
    - Archivo config.py con las rutas del corpus, las publicaciones
      y la configuración del proyecto.
    - Archivo split.json generado por el pipeline de olmOCR.

USO:

    Ejecutar con la configuración predeterminada:

        python scripts/01_pagexml_to_yolo.py

    Comprobar las clases detectadas y las estadísticas sin generar
    archivos:

        python scripts/01_pagexml_to_yolo.py --dry-run

    Seleccionar manualmente las clases que se incluirán:

        python scripts/01_pagexml_to_yolo.py --clases text graphic unknown

    Excluir las clases que aparezcan en menos de 5 páginas:

        python scripts/01_pagexml_to_yolo.py --min-paginas-clase 5

    Procesar únicamente determinadas publicaciones:

        python scripts/01_pagexml_to_yolo.py --publicaciones Semana Fémina

    Especificar otras rutas para el corpus y el directorio de salida:

        python scripts/01_pagexml_to_yolo.py --corpus RUTA_CORPUS --output RUTA_SALIDA

    Crear enlaces simbólicos a las imágenes en lugar de copiarlas
    (opción pensada principalmente para Linux/macOS):

        python scripts/01_pagexml_to_yolo.py --symlinks

SALIDA:

    data/data_regiones/
        classes.json
        informe_clases.json
        train/
            images/
            yolo_labels/
        val/
            images/
            yolo_labels/
        test/
            images/
            yolo_labels/

    El directorio generado se utiliza como entrada del siguiente
    paso del pipeline:

        python docLayout/02_prepare_dataset.py --corpus data_regiones

IMPORTANTE:
    Este script no entrena el modelo ni realiza la detección automática
    de regiones sobre las imágenes. Utiliza las anotaciones existentes
    en los PAGE-XML para generar las etiquetas YOLO que servirán como
    datos de entrenamiento.

    Debe ejecutarse antes de preparar y entrenar el modelo docLayout.
"""

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — PARSEO DE PAGE-XML
# ══════════════════════════════════════════════════════════════════════════════
 
# Tipos de región PAGE-XML que se detectan como clases YOLO.
# Clave: nombre local del tag (sin namespace).
# Valor: nombre de clase normalizado que usará el modelo.
# Se ajusta si Transkribus usa nombres distintos en el corpus.
_TAG_A_CLASE: dict[str, str] = {
    "TextRegion":    "text",
    "ImageRegion":   "graphic",
    "TableRegion":   "graphic",
    "GraphicRegion": "graphic",
    "UnknownRegion": "unknown",
}
 
 
def _coords_a_xyxy(points_str: str) -> tuple[int, int, int, int] | None:
    """
    Convierte el polígono PAGE-XML 'x1,y1 x2,y2 …' a una caja delimitadora.
 
    Parámetros:
        points_str — valor del atributo 'points' de un elemento Coords
 
    Devuelve:
        (xmin, ymin, xmax, ymax), o None si la cadena está vacía o mal formada
    """
    try:
        puntos = [
            tuple(int(v) for v in p.split(","))
            for p in points_str.strip().split()
            if "," in p
        ]
        if len(puntos) < 2:
            return None
        xs = [p[0] for p in puntos]
        ys = [p[1] for p in puntos]
        return min(xs), min(ys), max(xs), max(ys)
    except (ValueError, IndexError):
        return None
 
 
def parsear_regiones(ruta_xml: Path) -> list[dict] | None:
    """
    Extrae todas las regiones de un PAGE-XML.
 
    Parámetros:
        ruta_xml — ruta al archivo PAGE-XML
 
    Devuelve:
        Lista de {"clase", "xmin", "ymin", "xmax", "ymax", "img_w", "img_h"},
        o None si el XML no se puede parsear o no tiene página válida
    """
    try:
        tree = ET.parse(ruta_xml)
    except ET.ParseError:
        return None
 
    root = tree.getroot()
 
    # Localizar el elemento Page para obtener las dimensiones de la imagen
    page_elem = None
    for elem in root.iter():
        local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if local == "Page":
            page_elem = elem
            break
 
    if page_elem is None:
        return None
 
    try:
        img_w = int(page_elem.get("imageWidth", 0))
        img_h = int(page_elem.get("imageHeight", 0))
    except (ValueError, TypeError):
        return None
 
    if img_w <= 0 or img_h <= 0:
        return None
 
    regiones = []
    for elem in root.iter():
        tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag_local not in _TAG_A_CLASE:
            continue
 
        clase = _TAG_A_CLASE[tag_local]
 
        coords_elem = elem.find(".//{*}Coords")
        if coords_elem is None:
            continue
        bbox = _coords_a_xyxy(coords_elem.get("points", ""))
        if bbox is None:
            continue
 
        xmin, ymin, xmax, ymax = bbox
 
        # Descartar cajas degeneradas
        if xmax <= xmin or ymax <= ymin:
            continue
 
        # Recortar a los límites de la imagen y volver a validar
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_w, xmax)
        ymax = min(img_h, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
 
        regiones.append({
            "clase": clase,
            "xmin":  xmin,
            "ymin":  ymin,
            "xmax":  xmax,
            "ymax":  ymax,
            "img_w": img_w,
            "img_h": img_h,
        })
 
    return regiones
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — CONVERSIÓN A FORMATO YOLO
# ══════════════════════════════════════════════════════════════════════════════
 
def region_a_yolo(r: dict, clase_a_id: dict[str, int]) -> str | None:
    """
    Convierte una región a una línea de etiqueta YOLO normalizada.
 
    Parámetros:
        r          — región con clase, coordenadas absolutas y tamaño de imagen
        clase_a_id — mapeo de nombre de clase a identificador numérico
 
    Devuelve:
        'clase_id x_center y_center width height' con valores en [0, 1],
        o None si la clase no está en clase_a_id
    """
    cls_id = clase_a_id.get(r["clase"])
    if cls_id is None:
        return None
 
    img_w, img_h = r["img_w"], r["img_h"]
    x_center = ((r["xmin"] + r["xmax"]) / 2) / img_w
    y_center = ((r["ymin"] + r["ymax"]) / 2) / img_h
    width    = (r["xmax"] - r["xmin"]) / img_w
    height   = (r["ymax"] - r["ymin"]) / img_h
 
    # Limitar a [0, 1] por seguridad
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width    = max(0.0, min(1.0, width))
    height   = max(0.0, min(1.0, height))
 
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — ESCANEO DEL CORPUS Y DETECCIÓN DE CLASES
# ══════════════════════════════════════════════════════════════════════════════
 
def escanear_corpus(
    publicaciones: dict[str, str],
    corpus_dir: Path,
) -> tuple[list[dict], dict[str, int]]:
    """
    Recorre los pares JPG+XML del corpus, parsea sus regiones y contabiliza
    las clases presentes.
 
    Parámetros:
        publicaciones — mapeo de nombre de publicación a modo de split
        corpus_dir    — directorio raíz del corpus
 
    Devuelve:
        pares         — lista de {pub, modo, stem, jpg, xml, regiones}
        conteo_clases — {clase: n.º de páginas en que aparece}
    """
    pares: list[dict] = []
    conteo_clases: dict[str, int] = defaultdict(int)
 
    for nombre, modo in publicaciones.items():
        dir_pub  = corpus_dir / nombre
        dir_jpg  = dir_pub / "jpg"
        dir_page = dir_pub / "page"
 
        if not dir_jpg.exists() or not dir_page.exists():
            print(f"  [AVISO] {nombre}: carpetas jpg/ o page/ no encontradas, saltando.")
            continue
 
        jpgs = sorted(
            {p.resolve(): p for p in dir_jpg.glob("*")
             if p.suffix.lower() in (".jpg", ".jpeg")}.values()
        )
 
        encontrados = 0
        for jpg in jpgs:
            xml = dir_page / (jpg.stem + ".xml")
            if not xml.exists():
                continue
 
            regiones = parsear_regiones(xml)
            if regiones is None:
                print(f"  [AVISO] {jpg.stem}: XML inválido o sin dimensiones, omitido.")
                continue
 
            # Las páginas sin regiones anotadas se incluyen con etiqueta vacía
 
            # Contabilizar las clases presentes en esta página
            for cls in {r["clase"] for r in regiones}:
                conteo_clases[cls] += 1
 
            pares.append({
                "pub":      nombre,
                "modo":     modo,
                "stem":     jpg.stem,
                "jpg":      jpg,
                "xml":      xml,
                "regiones": regiones,
            })
            encontrados += 1
 
        print(f"  {nombre:<28}  {encontrados:>4} pares")
 
    return pares, dict(conteo_clases)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — SPLIT (compartido con olmOCR)
# ══════════════════════════════════════════════════════════════════════════════
 
def cargar_split(ruta: Path) -> dict[str, str]:
    """Carga el split.json que asigna cada página a train, val o test."""
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)
 
 
def asignar_splits(pares: list[dict], split_map: dict[str, str]) -> dict[str, list[dict]]:
    """
    Asigna cada página al split definido en split.json (compartido con olmOCR).
 
    Parámetros:
        pares     — páginas devueltas por escanear_corpus
        split_map — mapeo de stem a nombre de split
 
    Devuelve:
        {"train": [...], "val": [...], "test": [...]}; las páginas ausentes
        de split.json se asignan a train
    """
    resultado: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    sin_asignar = []
    for par in sorted(pares, key=lambda p: (p["pub"], p["stem"])):
        s = split_map.get(par["stem"])
        if s is None:
            sin_asignar.append(par["stem"])
            s = "train"
        resultado[s].append(par)
    if sin_asignar:
        print(f"  [AVISO] {len(sin_asignar)} páginas no están en split.json → train: "
              + ", ".join(sin_asignar[:5]) + (" …" if len(sin_asignar) > 5 else ""))
    return resultado
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — ESCRITURA DE LA ESTRUCTURA DE CARPETAS
# ══════════════════════════════════════════════════════════════════════════════
 
def _copiar_o_enlazar(src: Path, dst: Path, usar_symlinks: bool) -> None:
    """Copia el archivo o crea un symlink al original, según la opción."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if usar_symlinks:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)
 
 
def escribir_dataset(
    splits: dict[str, list[dict]],
    clase_a_id: dict[str, int],
    output_dir: Path,
    usar_symlinks: bool,
) -> dict:
    """
    Escribe la estructura de carpetas que espera 02_prepare_dataset.py.
 
    Parámetros:
        splits        — páginas asignadas a cada split
        clase_a_id    — mapeo de nombre de clase a identificador numérico
        output_dir    — directorio de salida
        usar_symlinks — enlazar las imágenes en lugar de copiarlas
 
    Devuelve:
        Estadísticas por split (páginas, regiones, páginas sin región, clases)
    """
    stats: dict[str, dict] = {}
 
    for split_name, pares in splits.items():
        dir_img    = output_dir / split_name / "images"
        dir_labels = output_dir / split_name / "yolo_labels"
        dir_img.mkdir(parents=True, exist_ok=True)
        dir_labels.mkdir(parents=True, exist_ok=True)
 
        n_paginas    = 0
        n_regiones   = 0
        n_sin_region = 0
        conteo_cls: dict[str, int] = defaultdict(int)
 
        for par in pares:
            stem = par["stem"]
 
            # Imagen
            dst_img = dir_img / par["jpg"].name
            _copiar_o_enlazar(par["jpg"], dst_img, usar_symlinks)
 
            # Etiqueta YOLO
            lineas = []
            for r in par["regiones"]:
                linea = region_a_yolo(r, clase_a_id)
                if linea:
                    lineas.append(linea)
                    conteo_cls[r["clase"]] += 1
 
            dst_label = dir_labels / (stem + ".txt")
            dst_label.write_text("\n".join(lineas), encoding="utf-8")
 
            if not lineas:
                n_sin_region += 1
 
            n_paginas  += 1
            n_regiones += len(lineas)
 
        stats[split_name] = {
            "paginas":    n_paginas,
            "regiones":   n_regiones,
            "sin_region": n_sin_region,
            "por_clase":  dict(conteo_cls),
        }
 
    return stats
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — INFORME Y RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
 
def imprimir_resumen(
    conteo_clases: dict[str, int],
    clase_a_id: dict[str, int],
    clases_excluidas: list[str],
    stats: dict[str, dict],
    output_dir: Path,
) -> None:
    """Imprime por pantalla las clases detectadas y la distribución del dataset."""
    total_pags = sum(s["paginas"] for s in stats.values())
 
    print(f"\n{'═'*65}")
    print("  CLASES DETECTADAS EN EL CORPUS")
    print(f"{'─'*65}")
    print(f"  {'CLASE':<20} {'ID':>3}  {'PÁGINAS':>8}  {'ESTADO'}")
    print(f"{'─'*65}")
    for cls, n_pags in sorted(conteo_clases.items(), key=lambda x: -x[1]):
        cls_id = clase_a_id.get(cls)
        estado = f"id={cls_id}" if cls_id is not None else "EXCLUIDA"
        print(f"  {cls:<20} {str(cls_id) if cls_id is not None else '—':>3}  {n_pags:>8}  {estado}")
    if clases_excluidas:
        print(f"\n  Excluidas por --min-paginas-clase o --clases: {clases_excluidas}")
 
    print(f"\n{'═'*65}")
    print("  DISTRIBUCIÓN DEL DATASET")
    print(f"{'─'*65}")
    print(f"  {'SPLIT':<8} {'PÁGINAS':>8}  {'REGIONES':>10}  {'SIN REGIÓN':>10}")
    print(f"{'─'*65}")
    for split_name in ["train", "val", "test"]:
        s = stats.get(split_name, {})
        print(f"  {split_name:<8} {s.get('paginas', 0):>8}  "
              f"{s.get('regiones', 0):>10}  {s.get('sin_region', 0):>10}")
    print(f"{'─'*65}")
    print(f"  {'TOTAL':<8} {total_pags:>8}")
    print(f"{'═'*65}\n")
    print(f"  Dataset escrito en: {output_dir}")
    print(f"  Siguiente paso: python scripts/02_prepare_dataset.py --corpus data-regiones")
 
 
def guardar_informe(
    conteo_clases: dict[str, int],
    clase_a_id: dict[str, int],
    clases_excluidas: list[str],
    stats: dict[str, dict],
    output_dir: Path,
) -> None:
    """Guarda en informe_clases.json las clases detectadas y las estadísticas."""
    informe = {
        "clases_detectadas": conteo_clases,
        "clase_a_id":        clase_a_id,
        "clases_excluidas":  clases_excluidas,
        "stats_por_split":   stats,
    }
    ruta = output_dir / "informe_clases.json"
    ruta.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Informe de clases guardado en: {ruta}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — CLI
# ══════════════════════════════════════════════════════════════════════════════
 
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convierte PAGE-XML del corpus a etiquetas YOLO para entrenar DocLayout-YOLO.\n"
            "Usa el mismo split 80-10-10 estratificado por publicación que olmOCR."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Primera vez: detección automática de clases
  python scripts/01_pagexml_to_yolo.py
 
  # Ver clases sin escribir nada
  python scripts/01_pagexml_to_yolo.py --dry-run
 
  # Excluir clases con pocas páginas
  python scripts/01_pagexml_to_yolo.py --min-paginas-clase 10
 
  # Forzar solo estas clases (el resto se ignora)
  python scripts/01_pagexml_to_yolo.py --clases text graphic unknown
 
  # Más rápido en Linux/macOS con symlinks
  python scripts/01_pagexml_to_yolo.py --symlinks
        """,
    )
    parser.add_argument(
        "--corpus", default=None,
        help=f"Ruta al directorio corpus (default: {config.CORPUS_DIR})",
    )
    parser.add_argument(
        "--output", default=None,
        help=f"Directorio de salida (default: {config.DATA_DIR})",
    )
    parser.add_argument(
        "--clases", nargs="+", default=None,
        help="Lista explícita de clases a incluir (el resto se ignora)",
    )
    parser.add_argument(
        "--min-paginas-clase", type=int, default=3,
        help="Excluir clases que aparecen en menos de N páginas (default: 3)",
    )
    parser.add_argument(
        "--split", default=str(config.SPLIT_PATH),
        help="split.json compartido con el pipeline olmOCR",
    )
    parser.add_argument(
        "--symlinks", action="store_true",
        help="Crear symlinks en vez de copiar imágenes (solo Linux/macOS)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Escanear y mostrar estadísticas sin escribir ficheros",
    )
    parser.add_argument(
        "--publicaciones", nargs="+", default=None,
        help="Nombres de publicaciones a incluir (default: todas las de config.PUBLICACIONES)",
    )
    args = parser.parse_args()
 
    corpus_dir = Path(args.corpus) if args.corpus else config.CORPUS_DIR
    output_dir = Path(args.output) if args.output else config.DATA_DIR
 
    if not corpus_dir.exists():
        print(f"ERROR: directorio corpus no encontrado: {corpus_dir}")
        sys.exit(1)
 
    # ── Escaneo del corpus ────────────────────────────────────────────────────
    publicaciones = config.PUBLICACIONES
    if args.publicaciones:
        publicaciones = {
            k: v for k, v in publicaciones.items()
            if k in args.publicaciones
        }
        no_encontradas = [p for p in args.publicaciones if p not in config.PUBLICACIONES]
        if no_encontradas:
            print(f"[AVISO] Estas publicaciones no están en config: {no_encontradas}")
 
    print(f"Escaneando corpus en {corpus_dir} …")
    pares, conteo_clases = escanear_corpus(publicaciones, corpus_dir)
 
    if not pares:
        print("ERROR: No se encontraron pares JPG+XML.")
        sys.exit(1)
 
    print(f"\n  {len(pares)} pares JPG+XML encontrados en total.")
    print(f"\n  Clases detectadas automáticamente:")
    for cls, n in sorted(conteo_clases.items(), key=lambda x: -x[1]):
        print(f"    {cls:<20} aparece en {n:>4} páginas")
 
    # ── Selección de clases ───────────────────────────────────────────────────
    if args.clases:
        clases_validas   = [c for c in args.clases if c in conteo_clases]
        clases_excluidas = [c for c in conteo_clases if c not in args.clases]
        clases_no_vistas = [c for c in args.clases if c not in conteo_clases]
        if clases_no_vistas:
            print(f"\n  [AVISO] Estas clases no aparecen en el corpus: {clases_no_vistas}")
    else:
        clases_validas = [
            c for c, n in conteo_clases.items()
            if n >= args.min_paginas_clase
        ]
        clases_excluidas = [
            c for c, n in conteo_clases.items()
            if n < args.min_paginas_clase
        ]
        if clases_excluidas:
            print(f"\n  Clases excluidas por --min-paginas-clase={args.min_paginas_clase}: "
                  f"{clases_excluidas}")
 
    if not clases_validas:
        print("ERROR: No quedan clases válidas. Revisa --clases o --min-paginas-clase.")
        sys.exit(1)
 
    # Asignar IDs ordenados alfabéticamente para garantizar reproducibilidad
    clases_validas_sorted = sorted(clases_validas)
    clase_a_id = {cls: i for i, cls in enumerate(clases_validas_sorted)}
 
    print(f"\n  Clases finales del modelo ({len(clase_a_id)}):")
    for cls, idx in sorted(clase_a_id.items(), key=lambda x: x[1]):
        print(f"    {idx}: {cls}")
 
    if args.dry_run:
        print("\n[DRY-RUN] No se han escrito ficheros.")
        return
 
    # ── Split ─────────────────────────────────────────────────────────────────
    print(f"\nAplicando split de {args.split} …")
    splits = asignar_splits(pares, cargar_split(Path(args.split)))
    for split_name, lista in splits.items():
        print(f"  {split_name:<6} {len(lista):>4} páginas")
 
    # ── Escritura del dataset ─────────────────────────────────────────────────
    print(f"\nEscribiendo dataset en {output_dir} …")
    output_dir.mkdir(parents=True, exist_ok=True)
 
    stats = escribir_dataset(splits, clase_a_id, output_dir, args.symlinks)
 
    # classes.json (formato que espera 02_prepare_dataset.py)
    classes_json = output_dir / "classes.json"
    classes_json.write_text(
        json.dumps(clase_a_id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  classes.json escrito en: {classes_json}")
 
    # ── Informe ───────────────────────────────────────────────────────────────
    guardar_informe(conteo_clases, clase_a_id, clases_excluidas, stats, output_dir)
    imprimir_resumen(conteo_clases, clase_a_id, clases_excluidas, stats, output_dir)
 
 
if __name__ == "__main__":
    main()