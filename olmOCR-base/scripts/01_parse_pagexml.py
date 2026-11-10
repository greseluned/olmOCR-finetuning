#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_parse_pagexml.py

Convierte archivos PAGE-XML de Transkribus en una transcripción estructurada
(respetando el ReadingOrder, con marcadores de imagen, pie de foto y columna)
y calcula metadatos de la página para clasificarla como portada,
imagen_dominante, texto_corrido, mixta o columnar. Sirve como módulo
(xml_a_transcripcion, clasificar_tipo_pagina) y como CLI de diagnóstico.

Uso:
    python scripts/01_parse_pagexml.py ../corpus/excelsior/page/
    python scripts/01_parse_pagexml.py ../corpus/excelsior/page/0001_p001.xml -t
    python scripts/01_parse_pagexml.py "../corpus/*/page/*.xml" --errores-solo

Opciones:
    --mostrar-texto, -t   muestra la transcripción extraída
    --verbose, -v         información detallada por fichero
    --limite, -n N        procesa como máximo N ficheros
    --max-chars N         caracteres de transcripción a mostrar (800)
    --errores-solo        muestra solo los ficheros con error
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CORRECCIÓN DE MOJIBAKE
# ══════════════════════════════════════════════════════════════════════════════

def _reparar_mojibake(texto: str) -> str:
    if not texto:
        return texto
    try:
        return texto.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — UTILIDADES DE NAMESPACE
# ══════════════════════════════════════════════════════════════════════════════

def detectar_ns(root: ET.Element) -> str:
    m = re.match(r"\{(.+?)\}", root.tag)
    return m.group(1) if m else ""


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — EXTRACCIÓN DE TEXTO POR REGIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _extraer_reading_order_custom(custom_attr: str) -> int:
    m = re.search(r"readingOrder\s*\{index:(\d+)", custom_attr)
    return int(m.group(1)) if m else 9999


def _tipo_region(region: ET.Element) -> str:
    tag = region.tag.split("}")[-1].lower()
    attr_type = region.get("type", "").lower()

    if tag == "textregion":
        if attr_type in ("heading", "header", "drop-capital"):
            return "heading"
        if attr_type in ("caption", "footnote"):
            return "caption"
        return "paragraph"

    if tag in ("imageregion", "graphicregion"):
        return "image"

    return "ignorar"


def texto_de_region(region: ET.Element, ns: str) -> str:
    equiv_region = region.find(f"{{{ns}}}TextEquiv")
    if equiv_region is not None:
        u = equiv_region.find(f"{{{ns}}}Unicode")
        if u is not None and u.text and u.text.strip():
            return _reparar_mojibake(u.text.strip())

    lineas: list[tuple[int, str]] = []
    for linea in region.findall(f"{{{ns}}}TextLine"):
        orden = _extraer_reading_order_custom(linea.get("custom", ""))
        equiv = linea.find(f"{{{ns}}}TextEquiv")
        if equiv is not None:
            u = equiv.find(f"{{{ns}}}Unicode")
            if u is not None and u.text and u.text.strip():
                lineas.append((orden, _reparar_mojibake(u.text.strip())))

    if lineas:
        lineas.sort(key=lambda x: x[0])
        return "\n".join(t for _, t in lineas)

    return ""


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — COORDENADAS Y DETECCIÓN DE COLUMNAS
# ══════════════════════════════════════════════════════════════════════════════

def bbox_region(region: ET.Element, ns: str) -> tuple[float, float, float, float]:
    coords = region.find(f"{{{ns}}}Coords")
    if coords is None:
        return 0.0, 0.0, 0.0, 0.0

    puntos_str = coords.get("points", "")
    xs, ys = [], []
    for p in puntos_str.split():
        partes = p.split(",")
        if len(partes) == 2:
            try:
                xs.append(float(partes[0]))
                ys.append(float(partes[1]))
            except ValueError:
                pass

    if not xs:
        return 0.0, 0.0, 0.0, 0.0

    return min(xs), min(ys), max(xs), max(ys)


def centroide_x(region: ET.Element, ns: str) -> float:
    x0, _, x1, _ = bbox_region(region, ns)
    return (x0 + x1) / 2.0


def _detectar_num_columnas(cxs: list[float], ancho_pagina: float) -> int:
    if len(cxs) < 2:
        return 1
    cxs_sorted = sorted(set(round(c) for c in cxs))
    umbral = ancho_pagina * 0.12
    cortes = sum(
        1 for i in range(len(cxs_sorted) - 1)
        if cxs_sorted[i + 1] - cxs_sorted[i] > umbral
    )
    return cortes + 1


def _asignar_columna(cx: float, cxs_ordenados: list[float], ancho_pagina: float) -> int:
    umbral = ancho_pagina * 0.12
    cxs_sorted = sorted(set(round(c) for c in cxs_ordenados))
    col = 1
    for i in range(len(cxs_sorted) - 1):
        if cxs_sorted[i + 1] - cxs_sorted[i] > umbral:
            if cx > cxs_sorted[i] + umbral / 2:
                col += 1
    return col


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def xml_a_transcripcion(ruta_xml: Path) -> tuple[str, dict]:
    try:
        tree = ET.parse(str(ruta_xml))
    except ET.ParseError as e:
        return "", {"error": f"XML inválido: {e}"}

    root = tree.getroot()
    ns = detectar_ns(root)
    if not ns:
        return "", {"error": "Namespace PAGE XML no detectado"}

    page = root.find(f"{{{ns}}}Page")
    if page is None:
        return "", {"error": "Elemento <Page> no encontrado"}

    ancho_pagina = float(page.get("imageWidth",  "2000"))
    alto_pagina  = float(page.get("imageHeight", "3000"))

    # ── Recopilar todas las regiones relevantes por ID ────────────────────────
    todas_regiones: dict[str, ET.Element] = {}
    for tag in ("TextRegion", "ImageRegion", "GraphicRegion"):
        for region in page.findall(f".//{{{ns}}}{tag}"):
            rid = region.get("id", "")
            if rid:
                todas_regiones[rid] = region

    # ── Orden de lectura desde ReadingOrder ───────────────────────────────────
    orden_ids: list[str] = []
    reading_order = page.find(f"{{{ns}}}ReadingOrder")
    if reading_order is not None:
        refs: list[tuple[int, str]] = []
        for elem in reading_order.iter():
            ref     = elem.get("regionRef")
            idx_str = elem.get("index")
            if ref and idx_str is not None:
                refs.append((int(idx_str), ref))
        refs.sort(key=lambda x: x[0])
        orden_ids = [ref for _, ref in refs]

    # Alternativa: orden DOM
    if not orden_ids:
        orden_ids = list(todas_regiones.keys())

    # ── Primera pasada: clasificar regiones y obtener centroides ──────────────
    elementos: list[dict] = []
    n_imagen = 0
    n_texto  = 0

    for rid in orden_ids:
        region = todas_regiones.get(rid)
        if region is None:
            continue

        tipo = _tipo_region(region)
        if tipo == "ignorar":
            continue

        x0, y0, x1, y1 = bbox_region(region, ns)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        ancho_region = x1 - x0

        if tipo == "image":
            n_imagen += 1
            elementos.append({
                "tipo":              "image",
                "cx":                cx,
                "cy":                cy,
                "ancho_region":      ancho_region,
                "texto":             "",
                "es_ancho_completo": True,
            })
        else:
            texto = texto_de_region(region, ns)
            if not texto.strip():
                continue
            n_texto += 1
            es_ancho_completo = (
                tipo == "heading"
                or ancho_region > ancho_pagina * 0.60
            )
            elementos.append({
                "tipo":              tipo,
                "cx":                cx,
                "cy":                cy,
                "ancho_region":      ancho_region,
                "texto":             texto,
                "es_ancho_completo": es_ancho_completo,
            })

    if not elementos:
        return "", {"error": "Sin regiones de texto extraíbles"}

    # ── Detectar número de columnas ───────────────────────────────────────────
    cxs_columna = [
        e["cx"] for e in elementos
        if not e["es_ancho_completo"] and e["tipo"] != "image"
    ]
    num_columnas = _detectar_num_columnas(cxs_columna, ancho_pagina) if cxs_columna else 1

    # ── Componer la transcripción respetando el ReadingOrder ──────────────────
    partes: list[str] = []
    col_actual: int | None = None

    for e in elementos:
        if e["tipo"] == "image":
            col_actual = None
            partes.append("[Fotografía: descripción]")
            continue

        if e["tipo"] == "caption":
            col_actual = None
            partes.append(f"*[Pie: {e['texto']}]*")
            continue

        if e["es_ancho_completo"] or num_columnas <= 1:
            col_actual = None
            partes.append(e["texto"])
        else:
            col = _asignar_columna(e["cx"], cxs_columna, ancho_pagina)
            if col != col_actual:
                col_actual = col
                partes.append(f"---COLUMNA {col}---")
            partes.append(e["texto"])

    transcripcion = "\n\n".join(partes)

    # ── Metadatos ─────────────────────────────────────────────────────────────
    total_regiones = n_imagen + n_texto
    fraccion_imagen = n_imagen / total_regiones if total_regiones else 0.0

    tiene_cabecera_ancha = any(e["es_ancho_completo"] for e in elementos
                               if e["tipo"] not in ("image", "caption"))
    tiene_columnas       = any(not e["es_ancho_completo"] for e in elementos
                               if e["tipo"] not in ("image", "caption"))

    metadatos = {
        "num_columnas_detectadas": num_columnas,
        "num_regiones":            len(elementos),
        "n_texto":                 n_texto,
        "n_imagen":                n_imagen,
        "ancho_pagina":            ancho_pagina,
        "alto_pagina":             alto_pagina,
        "fraccion_imagen":         round(fraccion_imagen, 3),
        "tiene_cabecera_ancha":    tiene_cabecera_ancha and tiene_columnas,
        "es_portada":              False,
    }
    return transcripcion, metadatos


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — CLASIFICADOR DE TIPO DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════

def clasificar_tipo_pagina(meta: dict) -> str:
    if "error" in meta:
        return "texto_corrido"

    if meta.get("es_portada", False):
        return "portada"

    if meta.get("fraccion_imagen", 0.0) >= 0.50:
        return "imagen_dominante"

    num_columnas = meta.get("num_columnas_detectadas", 1)

    if num_columnas <= 1:
        return "texto_corrido"

    if meta.get("tiene_cabecera_ancha", False):
        return "mixta"

    return "columnar"


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — CLI DE DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════════════

def _recopilar_xmls(entrada: str) -> list[Path]:
    p = Path(entrada)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.xml"))
    import glob as _glob
    return sorted(Path(f) for f in _glob.glob(entrada, recursive=True))


def _procesar_uno(ruta_xml: Path, verbose: bool, mostrar_texto: bool, max_chars: int) -> dict:
    transcripcion, meta = xml_a_transcripcion(ruta_xml)
    tipo = clasificar_tipo_pagina({**meta, "es_portada": False})
    meta["fichero"] = ruta_xml.name
    meta["chars"]   = len(transcripcion)
    meta["tipo"]    = tipo

    if verbose or mostrar_texto:
        print(f"\n{'─'*60}")
        print(f"Fichero  : {ruta_xml.name}")
        if "error" in meta:
            print(f"ERROR    : {meta['error']}")
        else:
            print(f"Página   : {meta['ancho_pagina']:.0f} × {meta['alto_pagina']:.0f} px")
            print(f"Regiones : texto={meta['n_texto']}  imagen={meta['n_imagen']}")
            print(f"Columnas : {meta['num_columnas_detectadas']}")
            print(f"Tipo     : {tipo}")
            print(f"Chars    : {meta['chars']}")
        if mostrar_texto and "error" not in meta:
            print(f"{'─'*60}")
            preview = (transcripcion if len(transcripcion) <= max_chars
                       else transcripcion[:max_chars] + f"\n… [{len(transcripcion)} chars total]")
            print(preview)
    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Parsea PAGE-XML de Transkribus y muestra la transcripción estructurada.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/01_parse_pagexml.py ../corpus/excelsior/page/0001_p001.xml
  python scripts/01_parse_pagexml.py ../corpus/excelsior/page/ --mostrar-texto
  python scripts/01_parse_pagexml.py ../corpus/excelsior/page/ --limite 10 -t
  python scripts/01_parse_pagexml.py "../corpus/*/page/*.xml" --errores-solo
        """,
    )
    parser.add_argument(
        "entrada",
        help="Fichero .xml, directorio con XMLs o glob entre comillas",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Mostrar información detallada de cada fichero",
    )
    parser.add_argument(
        "--mostrar-texto", "-t", action="store_true",
        help="Mostrar la transcripción extraída",
    )
    parser.add_argument(
        "--limite", "-n", type=int, default=0,
        help="Procesar como máximo N ficheros (default: 0 = todos)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=800,
        help="Máximo de caracteres de transcripción a mostrar (default: 800)",
    )
    parser.add_argument(
        "--errores-solo", action="store_true",
        help="Mostrar únicamente los ficheros con errores",
    )
    args = parser.parse_args()

    xmls = _recopilar_xmls(args.entrada)
    if not xmls:
        print(f"ERROR: No se encontraron XMLs en «{args.entrada}»")
        sys.exit(1)

    if args.limite:
        xmls = xmls[:args.limite]

    verbose_efectivo = args.verbose or args.mostrar_texto
    if not verbose_efectivo:
        print(f"Procesando {len(xmls)} ficheros XML …")

    # ── Procesamiento ─────────────────────────────────────────────────────────
    resultados = []
    for ruta_xml in xmls:
        if args.errores_solo:
            _, meta = xml_a_transcripcion(ruta_xml)
            if "error" in meta:
                print(f"  ERROR  {ruta_xml.name}: {meta['error']}")
            resultados.append(meta)
        else:
            meta = _procesar_uno(ruta_xml, verbose_efectivo, args.mostrar_texto, args.max_chars)
            resultados.append(meta)

    # ── Resumen ───────────────────────────────────────────────────────────────
    total   = len(resultados)
    errores = [r for r in resultados if "error" in r]
    ok      = [r for r in resultados if "error" not in r]

    print(f"\n{'═'*60}")
    print(f"  RESUMEN — {total} ficheros procesados")
    print(f"{'═'*60}")
    print(f"  OK      : {len(ok)}")
    print(f"  Errores : {len(errores)}")

    if ok:
        chars_total = sum(r["chars"] for r in ok)
        print(f"  Chars totales : {chars_total:,}")

        tipos_dist: dict[str, int] = {}
        for r in ok:
            t = r.get("tipo", "?")
            tipos_dist[t] = tipos_dist.get(t, 0) + 1
        print("  Distribución de tipos:")
        for t, n in sorted(tipos_dist.items(), key=lambda x: -x[1]):
            print(f"    {t:<20} {n:>4}")

        cols_dist: dict[int, int] = {}
        for r in ok:
            c = r.get("num_columnas_detectadas", 0)
            cols_dist[c] = cols_dist.get(c, 0) + 1
        print("  Distribución de columnas detectadas:")
        for ncols in sorted(cols_dist):
            barra = "█" * (cols_dist[ncols] * 30 // len(ok))
            print(f"    {ncols} col  {cols_dist[ncols]:>4} págs  {barra}")

    if errores:
        print("\n  Ficheros con error:")
        for r in errores:
            print(f"    · {r.get('fichero','?')}: {r['error']}")

    print(f"{'═'*60}")


if __name__ == "__main__":
    main()