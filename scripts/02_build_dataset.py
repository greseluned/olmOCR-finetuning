#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_build_dataset.py
===================
Descubre todos los pares JPG + PAGE XML del corpus, aplica el split
80-10-10 dentro de cada publicación y escribe directamente los tres
JSONL finales (train / validation / test).

El split anterior en dos pasos (02 + 03) queda unificado aquí.

Estructura del corpus esperada (fuera del directorio de trabajo):
    ../corpus/<publicacion>/jpg/<stem>.jpg
    ../corpus/<publicacion>/page/<stem>.xml

Las publicaciones y su comportamiento se configuran en config.PUBLICACIONES:
    'estratificar' → 80-10-10 dentro de la publicación
    'solo_train'   → todo va a train (material insuficiente)

Si una publicación marcada como 'estratificar' tiene menos de
config.SPLIT_MIN_PAGINAS páginas válidas, se trata como 'solo_train'
con un aviso.

Uso:
    python scripts/02_build_dataset.py
    python scripts/02_build_dataset.py --publicacion excelsior
    python scripts/02_build_dataset.py --dry-run
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
_spec = _ilu.spec_from_file_location(
    "parse_pagexml",
    Path(__file__).parent / "01_parse_pagexml.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
xml_a_transcripcion    = _mod.xml_a_transcripcion
clasificar_tipo_pagina = _mod.clasificar_tipo_pagina


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — DESCUBRIMIENTO DE PARES POR PUBLICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def descubrir_pares_publicacion(
    dir_publicacion: Path,
    nombre: str,
) -> list[tuple[Path, Path, str]]:
    """
    Busca pares (jpg, xml) en:
        dir_publicacion/jpg/<stem>.jpg
        dir_publicacion/page/<stem>.xml

    Returns: lista de (ruta_jpg, ruta_xml, nombre_publicacion)
    """
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
    # Glob insensible a mayúsculas en Windows: buscar .jpg y .JPG
    jpgs = sorted({p.resolve(): p for p in dir_jpg.glob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg")}.values())

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
# SECCIÓN 2 — PROCESADO DE UN PAR (JPG + XML → registro JSONL)
# ══════════════════════════════════════════════════════════════════════════════

def _es_portada(stem: str) -> bool:
    """
    Heurística de portada por nombre de fichero.
    Los stems que terminan en _p001 o _001 se consideran portada.
    Ajusta el patrón a tu nomenclatura si es necesario.
    """
    s = stem.lower()
    return s.endswith("_p001") or s.endswith("_001")


def procesar_par(
    jpg: Path,
    xml: Path,
    publicacion: str,
) -> dict | None:
    """
    Parsea el PAGE XML y construye el registro JSONL.
    Devuelve None si hay error o la transcripción está vacía.
    """
    transcripcion, meta = xml_a_transcripcion(xml)

    if "error" in meta or not transcripcion.strip():
        return None

    meta["es_portada"] = _es_portada(jpg.stem)
    tipo_pagina = clasificar_tipo_pagina(meta)
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
            "columnas":     meta.get("num_columnas_detectadas"),
            "n_texto":      meta.get("n_texto"),
            "n_imagen":     meta.get("n_imagen"),
            "frac_imagen":  meta.get("fraccion_imagen"),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — SPLIT POR PUBLICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def split_publicacion(
    registros: list[dict],
    modo: str,
    nombre: str,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Divide los registros de una publicación en train / val / test.

    modo='solo_train'   → todo a train.
    modo='estratificar' → 80-10-10; si hay menos de SPLIT_MIN_PAGINAS,
                          cae a solo_train con aviso.

    Returns: (train, val, test)
    """
    n = len(registros)

    if modo == "solo_train" or n < config.SPLIT_MIN_PAGINAS:
        if modo == "estratificar" and n < config.SPLIT_MIN_PAGINAS:
            print(f"  [AVISO] {nombre}: solo {n} págs (<{config.SPLIT_MIN_PAGINAS}), "
                  f"todo a train.")
        shuffled = registros[:]
        rng.shuffle(shuffled)
        return shuffled, [], []

    shuffled = registros[:]
    rng.shuffle(shuffled)

    n_test = max(1, round(n * config.SPLIT_TEST))
    n_val  = max(1, round(n * config.SPLIT_VALIDATION))
    # Ajustar para no sobrepasar n
    if n_test + n_val >= n:
        n_test = 1
        n_val  = 1

    test  = shuffled[:n_test]
    val   = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]

    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — ESCRITURA DE JSONL
# ══════════════════════════════════════════════════════════════════════════════

def escribir_jsonl(registros: list[dict], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — RESUMEN ESTADÍSTICO
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
    tipos: dict[str, int] = {}
    for r in train_total + val_total + test_total:
        t = r["_meta"]["tipo_pagina"]
        tipos[t] = tipos.get(t, 0) + 1
    print(f"\n  Tipos de página en el dataset completo:")
    for t, n in sorted(tipos.items(), key=lambda x: -x[1]):
        print(f"    {t:<22} {n:>4}")
    print(f"{'═'*70}")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Construye los JSONL de train/val/test con split 80-10-10 "
            "por publicación a partir del corpus externo."
        ),
        epilog="Ejemplo: python scripts/02_build_dataset.py --publicacion excelsior",
    )
    parser.add_argument(
        "--corpus", default=None,
        help=f"Ruta al directorio corpus (default: {config.CORPUS_DIR})",
    )
    parser.add_argument(
        "--publicacion", default=None,
        help="Procesar solo esta publicación (nombre de carpeta)",
    )
    parser.add_argument(
        "--semilla", type=int, default=config.SPLIT_SEED,
        help=f"Semilla aleatoria (default: {config.SPLIT_SEED})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Descubrir y parsear pares pero no escribir ficheros JSONL",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus) if args.corpus else config.CORPUS_DIR
    if not corpus_dir.exists():
        print(f"ERROR: directorio corpus no encontrado: {corpus_dir}")
        sys.exit(1)

    rng = random.Random(args.semilla)

    # ── Seleccionar publicaciones a procesar ──────────────────────────────────
    publicaciones = config.PUBLICACIONES
    if args.publicacion:
        if args.publicacion not in publicaciones:
            print(f"ERROR: '{args.publicacion}' no está en config.PUBLICACIONES")
            print(f"Disponibles: {list(publicaciones.keys())}")
            sys.exit(1)
        publicaciones = {args.publicacion: publicaciones[args.publicacion]}

    # ── Procesar cada publicación ─────────────────────────────────────────────
    train_total: list[dict] = []
    val_total:   list[dict] = []
    test_total:  list[dict] = []
    stats_pub:   dict       = {}
    omitidos_total = 0

    for nombre, modo in publicaciones.items():
        dir_pub = corpus_dir / nombre
        if not dir_pub.exists():
            print(f"  [AVISO] Carpeta no encontrada, saltando: {dir_pub}")
            continue

        # Descubrir pares
        pares = descubrir_pares_publicacion(dir_pub, nombre)
        if not pares:
            print(f"  [AVISO] {nombre}: sin pares JPG+XML, saltando.")
            continue

        # Parsear y construir registros
        registros = []
        omitidos  = 0
        for jpg, xml, pub in pares:
            reg = procesar_par(jpg, xml, pub)
            if reg is None:
                omitidos += 1
            else:
                registros.append(reg)

        if omitidos:
            print(f"  [INFO] {nombre}: {omitidos} pares omitidos "
                  f"(XML vacío o con error)")
        omitidos_total += omitidos

        if not registros:
            print(f"  [AVISO] {nombre}: sin registros válidos, saltando.")
            continue

        # Split
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

    # Mezclar para que los splits no queden ordenados por publicación
    rng.shuffle(train_total)
    rng.shuffle(val_total)
    rng.shuffle(test_total)

    imprimir_resumen(stats_pub, train_total, val_total, test_total)

    if omitidos_total:
        print(f"\n  Pares omitidos en total: {omitidos_total}")

    if args.dry_run:
        print("\n[DRY-RUN] No se han escrito ficheros.")
        return

    # ── Escribir JSONL ────────────────────────────────────────────────────────
    escribir_jsonl(train_total, config.JSONL_TRAIN)
    escribir_jsonl(val_total,   config.JSONL_VALIDATION)
    escribir_jsonl(test_total,  config.JSONL_TEST)

    print(f"\nJSONL escritos en: {config.DATASET_DIR}")
    print(f"  train      : {config.JSONL_TRAIN}")
    print(f"  validation : {config.JSONL_VALIDATION}")
    print(f"  test       : {config.JSONL_TEST}")
    print("\nSiguiente paso: python scripts/03_train_lora.py")


if __name__ == "__main__":
    main()