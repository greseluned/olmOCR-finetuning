#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_build_dataset.py

Construye los JSONL de train, validation y test para olmOCR usando el tipo de
página predicho por docLayout (dataset/layout_cache.json). El split no se
calcula aquí: se lee de split.json (config.SPLIT_PATH), compartido con
docLayout. Si una página no está en la caché, usa el clasificador XML de
01_parse_pagexml.py; _meta.tipo_desde indica el origen ('doclayout',
'heuristica' o 'xml').

Requisitos:
    Haber ejecutado antes 02_build_cache_layout.py y tener split.json.

Uso:
    python scripts/03_build_dataset.py
    python scripts/03_build_dataset.py --publicacion excelsior
    python scripts/03_build_dataset.py --dry-run
    python scripts/03_build_dataset.py --sin-cache

Opciones:
    --corpus RUTA         directorio del corpus (config.CORPUS_DIR)
    --cache RUTA          layout_cache.json (dataset/layout_cache.json)
    --publicacion NOMBRE  procesa solo esa publicación
    --semilla N           solo afecta al orden de mezcla dentro de cada split
    --sin-cache           ignora la caché y clasifica siempre desde el XML
    --dry-run             procesa sin escribir los JSONL

Salida:
    dataset/{train,validation,test}/*.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from config import PROMPTS

import importlib.util as _ilu

# ── Importar 01_parse_pagexml.py ─────────────────────────────────────────────
_spec = _ilu.spec_from_file_location(
    "parse_pagexml",
    Path(__file__).parent / "01_parse_pagexml.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
xml_a_transcripcion    = _mod.xml_a_transcripcion
clasificar_tipo_pagina = _mod.clasificar_tipo_pagina   # fallback

# Ruta por defecto de la caché (generada por 02_build_cache_layout.py)
CACHE_PATH = config.DATASET_DIR / "layout_cache.json"


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CARGA DE LA CACHÉ
# ══════════════════════════════════════════════════════════════════════════════

def cargar_cache(ruta: Path) -> dict[str, str]:
    if not ruta.exists():
        print(f"  [AVISO] Caché no encontrada: {ruta}")
        print(f"  Ejecuta primero: python scripts/02_build_cache_layout.py")
        print(f"  Continuando con clasificación desde XML (modo fallback).\n")
        return {}

    with open(ruta, encoding="utf-8") as f:
        cache = json.load(f)

    print(f"  Caché cargada: {len(cache)} páginas  ({ruta})")
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — DESCUBRIMIENTO DE PARES
# ══════════════════════════════════════════════════════════════════════════════

def descubrir_pares_publicacion(
    dir_publicacion: Path,
    nombre: str,
) -> list[tuple[Path, Path, str]]:
    dir_jpg  = dir_publicacion / "jpg"
    dir_page = dir_publicacion / "page"

    if not dir_jpg.exists():
        print(f"  [AVISO] {nombre}: no se encontró {dir_jpg}")
        return []
    if not dir_page.exists():
        print(f"  [AVISO] {nombre}: no se encontró {dir_page}")
        return []

    pares = []
    sin_xml = []
    jpgs = sorted(
        {p.resolve(): p for p in dir_jpg.glob("*")
         if p.suffix.lower() in (".jpg", ".jpeg")}.values()
    )

    for jpg in jpgs:
        xml = dir_page / (jpg.stem + ".xml")
        if xml.exists():
            pares.append((jpg, xml, nombre))
        else:
            sin_xml.append(jpg.name)

    if sin_xml:
        print(f"  [AVISO] {nombre}: {len(sin_xml)} JPG sin XML — "
              + ", ".join(sin_xml[:3]) + (" …" if len(sin_xml) > 3 else ""))
    return pares


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — PROCESADO DE UN PAR (JPG + XML → registro JSONL)
# ══════════════════════════════════════════════════════════════════════════════

def _es_portada(stem: str) -> bool:
    s = stem.lower()
    return s.endswith("_p001") or s.endswith("_001")


def procesar_par(
    jpg: Path,
    xml: Path,
    publicacion: str,
    cache: dict[str, str],
) -> dict | None:
    transcripcion, meta = xml_a_transcripcion(xml)

    if "error" in meta or not transcripcion.strip():
        return None

    es_portada = _es_portada(jpg.stem)
    meta["es_portada"] = es_portada

    # ── Clasificación del tipo de página ─────────────────────────────────────
    if es_portada:
        tipo_pagina = "portada"
        tipo_desde  = "heuristica"

    elif jpg.stem in cache:
        tipo_pagina = cache[jpg.stem]
        tipo_desde  = "doclayout"

    else:
        # Fallback al clasificador XML original
        tipo_pagina = clasificar_tipo_pagina(meta)
        tipo_desde  = "xml"

    instruccion = PROMPTS[tipo_pagina]

    return {
        "image": str(jpg.resolve()),
        "conversations": [
            {"role": "user",      "content": instruccion},
            {"role": "assistant", "content": transcripcion},
        ],
        "_meta": {
            "fuente":       jpg.stem,
            "publicacion":  publicacion,
            "tipo_pagina":  tipo_pagina,
            "tipo_desde":   tipo_desde,
            "columnas":     meta.get("num_columnas_detectadas"),
            "n_texto":      meta.get("n_texto"),
            "n_imagen":     meta.get("n_imagen"),
            "frac_imagen":  meta.get("fraccion_imagen"),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — SPLIT (desde split.json)
# ══════════════════════════════════════════════════════════════════════════════

if not Path(config.SPLIT_PATH).exists():
    print(f"ERROR: no existe {config.SPLIT_PATH}\n"
          f"  Genera el split compartido con make_split.py (en la raíz del repo).")
    sys.exit(1)

SPLIT_MAP: dict[str, str] = json.load(open(config.SPLIT_PATH, encoding="utf-8"))
_AVISADAS: set[str] = set()


def split_publicacion(registros, modo, nombre, rng):
    out = {"train": [], "val": [], "test": []}
    for r in registros:
        stem = r["_meta"]["fuente"]
        s = SPLIT_MAP.get(stem)
        if s is None:
            if stem not in _AVISADAS:
                print(f"  [AVISO] {stem} no está en split.json → train")
                _AVISADAS.add(stem)
            s = "train"
        out[s].append(r)
    return out["train"], out["val"], out["test"]


def verificar_split(val_total: list[dict], test_total: list[dict]) -> bool:
    ok = True
    print(f"\n  Verificación contra {config.SPLIT_PATH.name}:")
    for nombre, regs in (("val", val_total), ("test", test_total)):
        reales    = {r["_meta"]["fuente"] for r in regs}
        esperados = {s for s, v in SPLIT_MAP.items() if v == nombre}
        faltan    = sorted(esperados - reales)
        sobran    = sorted(reales - esperados)
        estado    = "OK" if not faltan and not sobran else "REVISAR"
        print(f"    {nombre:<5} esperadas={len(esperados):>3}  "
              f"generadas={len(reales):>3}  [{estado}]")
        if faltan:
            print(f"      faltan : {faltan[:8]}{' …' if len(faltan) > 8 else ''}")
            ok = False
        if sobran:
            print(f"      sobran : {sobran[:8]}{' …' if len(sobran) > 8 else ''}")
            ok = False
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — ESCRITURA DE JSONL
# ══════════════════════════════════════════════════════════════════════════════

def escribir_jsonl(registros: list[dict], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — RESUMEN ESTADÍSTICO
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_resumen(
    stats_pub: dict,
    train_total: list,
    val_total: list,
    test_total: list,
) -> None:
    total = len(train_total) + len(val_total) + len(test_total)

    print(f"\n{'═'*70}")
    print(f"  {'PUBLICACIÓN':<28} {'TOTAL':>5}  {'TRAIN':>5}  {'VAL':>4}  {'TEST':>4}")
    print(f"{'─'*70}")
    for nombre, s in sorted(stats_pub.items()):
        print(f"  {nombre:<28} {s['total']:>5}  {s['train']:>5}  "
              f"{s['val']:>4}  {s['test']:>4}")
    print(f"{'─'*70}")
    print(f"  {'TOTAL':<28} {total:>5}  {len(train_total):>5}  "
          f"{len(val_total):>4}  {len(test_total):>4}")

    # Distribución de tipos de página
    todos = train_total + val_total + test_total
    tipos: dict[str, int] = {}
    for r in todos:
        t = r["_meta"]["tipo_pagina"]
        tipos[t] = tipos.get(t, 0) + 1

    print(f"\n  Tipos de página:")
    for t, n in sorted(tipos.items(), key=lambda x: -x[1]):
        print(f"    {t:<22} {n:>4}")

    # Cobertura de la caché
    desde: dict[str, int] = {}
    for r in todos:
        d = r["_meta"]["tipo_desde"]
        desde[d] = desde.get(d, 0) + 1

    print(f"\n  Origen de la clasificación:")
    for d, n in sorted(desde.items(), key=lambda x: -x[1]):
        pct = n / total * 100 if total else 0
        print(f"    {d:<22} {n:>4}  ({pct:.1f} %)")

    print(f"{'═'*70}")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Construye los JSONL de train/val/test para olmOCR usando\n"
            "el tipo de página predicho por docLayout (layout_cache.json)\n"
            "y el split compartido (split.json)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/03_build_dataset.py
  python scripts/03_build_dataset.py --publicacion excelsior
  python scripts/03_build_dataset.py --dry-run
  python scripts/03_build_dataset.py --sin-cache
        """,
    )
    parser.add_argument(
        "--corpus", default=None,
        help=f"Ruta al corpus (default: {config.CORPUS_DIR})",
    )
    parser.add_argument(
        "--cache", default=None,
        help=f"Ruta a layout_cache.json (default: {CACHE_PATH})",
    )
    parser.add_argument(
        "--publicacion", default=None,
        help="Procesar solo esta publicación",
    )
    parser.add_argument(
        "--semilla", type=int, default=config.SPLIT_SEED,
        help="Sin efecto sobre el split (viene de split.json); "
             "solo afecta al orden de mezcla dentro de cada split",
    )
    parser.add_argument(
        "--sin-cache", action="store_true",
        help="Ignorar la caché y clasificar siempre desde el XML (comportamiento original)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Procesar pares pero no escribir ficheros JSONL",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus) if args.corpus else config.CORPUS_DIR
    cache_path = Path(args.cache)  if args.cache  else CACHE_PATH

    if not corpus_dir.exists():
        print(f"ERROR: corpus no encontrado: {corpus_dir}")
        sys.exit(1)

    rng = random.Random(args.semilla)

    # ── Cargar caché ──────────────────────────────────────────────────────────
    if args.sin_cache:
        print("Modo --sin-cache: clasificación desde XML en todas las páginas.\n")
        cache: dict[str, str] = {}
    else:
        print("Cargando caché de layout …")
        cache = cargar_cache(cache_path)

    # ── Seleccionar publicaciones ─────────────────────────────────────────────
    publicaciones = config.PUBLICACIONES
    if args.publicacion:
        if args.publicacion not in publicaciones:
            print(f"ERROR: '{args.publicacion}' no está en config.PUBLICACIONES")
            print(f"Disponibles: {list(publicaciones.keys())}")
            sys.exit(1)
        publicaciones = {args.publicacion: publicaciones[args.publicacion]}

    # ── Procesar publicaciones ────────────────────────────────────────────────
    train_total: list[dict] = []
    val_total:   list[dict] = []
    test_total:  list[dict] = []
    stats_pub:   dict       = {}
    omitidos_total = 0

    print(f"\nProcesando {len(publicaciones)} publicaciones …\n")

    for nombre, modo in publicaciones.items():
        dir_pub = corpus_dir / nombre
        if not dir_pub.exists():
            print(f"  [AVISO] Carpeta no encontrada, saltando: {dir_pub}")
            continue

        pares = descubrir_pares_publicacion(dir_pub, nombre)
        if not pares:
            print(f"  [AVISO] {nombre}: sin pares JPG+XML, saltando.")
            continue

        registros = []
        omitidos  = 0
        for jpg, xml, pub in pares:
            reg = procesar_par(jpg, xml, pub, cache)
            if reg is None:
                omitidos += 1
            else:
                registros.append(reg)

        if omitidos:
            print(f"  [INFO] {nombre}: {omitidos} pares omitidos (XML vacío o con error)")
        omitidos_total += omitidos

        if not registros:
            print(f"  [AVISO] {nombre}: sin registros válidos, saltando.")
            continue

        train, val, test = split_publicacion(registros, modo, nombre, rng)

        stats_pub[nombre] = {
            "total": len(registros),
            "train": len(train),
            "val":   len(val),
            "test":  len(test),
        }

        train_total.extend(train)
        val_total.extend(val)
        test_total.extend(test)

        print(f"  {nombre:<28}  total={len(registros):>3}  "
              f"train={len(train):>3}  val={len(val):>2}  test={len(test):>2}")

    if not train_total and not val_total and not test_total:
        print("ERROR: No se generó ningún registro.")
        sys.exit(1)

    rng.shuffle(train_total)
    rng.shuffle(val_total)
    rng.shuffle(test_total)

    imprimir_resumen(stats_pub, train_total, val_total, test_total)

    if omitidos_total:
        print(f"\n  Pares omitidos en total: {omitidos_total}")

    # Solo tiene sentido verificar si se han procesado todas las publicaciones
    split_ok = True
    if not args.publicacion:
        split_ok = verificar_split(val_total, test_total)

    if args.dry_run:
        print("\n[DRY-RUN] No se han escrito ficheros JSONL.")
        return

    if not split_ok:
        print("\nERROR: el dataset no coincide con split.json (ver arriba). "
              "No se escriben los JSONL.")
        sys.exit(1)

    # ── Escribir JSONL ────────────────────────────────────────────────────────
    escribir_jsonl(train_total, config.JSONL_TRAIN)
    escribir_jsonl(val_total,   config.JSONL_VALIDATION)
    escribir_jsonl(test_total,  config.JSONL_TEST)

    print(f"\nJSONL escritos en: {config.DATASET_DIR}")
    print(f"  train      : {config.JSONL_TRAIN}")
    print(f"  validation : {config.JSONL_VALIDATION}")
    print(f"  test       : {config.JSONL_TEST}")
    print(f"\nSiguiente paso: python scripts/04_train_lora.py")


if __name__ == "__main__":
    main()