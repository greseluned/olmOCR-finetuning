#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_build_dataset.py

Construye los JSONL de train, validation y test con un registro por región
(recorte de imagen, prompt text|graphic y texto GT de la región). El split se
lee de split.json (config.SPLIT_PATH) y se decide una vez por página; los
recortes se guardan en dataset/crops/<split>/. En train y val las regiones son
las de docLayout (layout_regions_cache.json), emparejadas con el GT del PAGE-XML
por IoU; en test se usan todas las regiones GT del PAGE-XML. Al final verifica
el resultado contra split.json.

Requisitos:
    Haber ejecutado antes 02_build_cache_layout.py y tener split.json.

Uso:
    python scripts/03_build_dataset.py
    python scripts/03_build_dataset.py --dry-run
    python scripts/03_build_dataset.py --iou 0.40
    python scripts/03_build_dataset.py --sin-cache
    python scripts/03_build_dataset.py --publicacion excelsior

Opciones:
    --corpus RUTA          directorio del corpus (config.CORPUS_DIR)
    --cache RUTA           layout_regions_cache.json (dataset/layout_regions_cache.json)
    --publicacion NOMBRE   procesa solo esa publicación
    --semilla N            solo afecta al orden de mezcla del train
    --iou X                umbral IoU docLayout/GT (0.30)
    --sin-cache            usa como layout las regiones del XML
    --dry-run              procesa y verifica sin escribir recortes ni JSONL
    --conservar-recortes   no vacía dataset/crops/ antes de empezar
    --forzar               escribe los JSONL aunque falle la verificación

Salida:
    dataset/{train,validation,test}/*.jsonl, dataset/crops/<split>/ y
    dataset/paginas_por_split.json.
"""

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import importlib.util as _ilu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# ── Importar 01_parse_pagexml.py ─────────────────────────────────────────────
_spec = _ilu.spec_from_file_location(
    "parse_pagexml",
    Path(__file__).parent / "01_parse_pagexml.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
xml_a_regiones = _mod.xml_a_regiones

# ── Constantes ───────────────────────────────────────────────────────────────
# Los prompts viven SOLO en config.PROMPTS ("text" y "graphic"): 03 y 05_inference
# los importan de ahí para que entrenamiento e inferencia usen el mismo texto.
PROMPT_TEXT      = config.PROMPTS["text"]
PROMPT_GRAPHIC   = config.PROMPTS["graphic"]
MARCADOR_GRAPHIC = "[Fotografía: descripción]"

CACHE_PATH  = config.DATASET_DIR / "layout_regions_cache.json"
DIR_CROPS   = config.DATASET_DIR / "crops"
IOU_DEFAULT = 0.30
SPLITS      = ("train", "val", "test")
SPLIT_PATH  = Path(getattr(config, "SPLIT_PATH", config.ROOT.parent / "split.json"))


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 0 — SPLIT COMPARTIDO
# ══════════════════════════════════════════════════════════════════════════════

def cargar_split(ruta: Path) -> dict[str, str]:
    if not ruta.exists():
        print(f"ERROR: no existe {ruta}\n"
              f"  Genera el split compartido con make_split.py (raíz del repo).")
        sys.exit(1)
    with open(ruta, encoding="utf-8") as f:
        split = json.load(f)
    invalidos = sorted({v for v in split.values() if v not in SPLITS})
    if invalidos:
        print(f"ERROR: valores de split no válidos en {ruta}: {invalidos} "
              f"(esperados: {SPLITS})")
        sys.exit(1)
    return split


SPLIT_MAP: dict[str, str] = cargar_split(SPLIT_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CARGA DE LA CACHÉ DE REGIONES
# ══════════════════════════════════════════════════════════════════════════════

def cargar_cache(ruta: Path) -> dict[str, list[dict]]:
    if not ruta.exists():
        print(f"  [AVISO] Caché no encontrada: {ruta}")
        print(f"  Ejecuta primero: python scripts/02_build_cache_layout.py")
        print(f"  Continuando con regiones del XML (modo fallback).\n")
        return {}

    with open(ruta, encoding="utf-8") as f:
        cache = json.load(f)

    n_regiones = sum(len(v) for v in cache.values())
    print(f"  Caché cargada: {len(cache)} páginas, {n_regiones} regiones  ({ruta})")
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — GEOMETRÍA Y EMPAREJAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a: list[int], b: list[int]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    area_i = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_u = area_a + area_b - area_i
    return area_i / area_u if area_u > 0 else 0.0


def _emparejar_regiones(
    regiones_dl: list[dict],
    regiones_gt: list[dict],
    iou_min: float,
) -> list[tuple[dict, dict | None, float]]:
    candidatos = []
    for i, dl in enumerate(regiones_dl):
        for j, gt in enumerate(regiones_gt):
            v = _iou(dl["bbox"], gt["bbox"])
            if v >= iou_min:
                candidatos.append((v, i, j))
    candidatos.sort(reverse=True)

    asignado: dict[int, tuple[int, float]] = {}
    gt_usadas: set[int] = set()
    for v, i, j in candidatos:
        if i in asignado or j in gt_usadas:
            continue
        asignado[i] = (j, v)
        gt_usadas.add(j)

    pares = []
    for i, dl in sorted(enumerate(regiones_dl), key=lambda t: t[1]["orden"]):
        if i in asignado:
            j, v = asignado[i]
            pares.append((dl, regiones_gt[j], v))
        else:
            pares.append((dl, None, 0.0))
    return pares


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — RECORTE DE IMÁGENES
# ══════════════════════════════════════════════════════════════════════════════

def recortar_region(
    jpg: Path,
    bbox: list[int],
    split: str,
    dry_run: bool,
    margen: int = 4,
) -> Path | None:
    from PIL import Image

    x0, y0, x1, y1 = bbox
    ruta_crop = DIR_CROPS / split / f"{jpg.stem}__{x0}_{y0}_{x1}_{y1}.jpg"

    if dry_run or ruta_crop.exists():
        return ruta_crop

    try:
        with Image.open(jpg) as im:
            w, h = im.size
            box = (
                max(0, x0 - margen), max(0, y0 - margen),
                min(w, x1 + margen), min(h, y1 + margen),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                return None
            ruta_crop.parent.mkdir(parents=True, exist_ok=True)
            im.crop(box).save(ruta_crop, quality=95)
        return ruta_crop
    except Exception as e:
        print(f"    [ERROR recorte] {jpg.name} bbox={bbox}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — REGISTROS POR PÁGINA
# ══════════════════════════════════════════════════════════════════════════════

def _es_portada(stem: str) -> bool:
    s = stem.lower()
    return s.endswith("_p001") or s.endswith("_001")


def _stats_vacio() -> dict[str, int]:
    return {
        "con_gt": 0, "sin_gt_text": 0, "sin_gt_graphic": 0,
        "tipo_discordante": 0, "gt_texto_vacio": 0,
        "duplicadas": 0, "recorte_fallido": 0,
    }


def _registro(ruta_crop: Path, prompt: str, objetivo: str, meta: dict) -> dict:
    return {
        "image": str(ruta_crop.resolve()),
        "conversations": [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": objetivo},
        ],
        "_meta": meta,
    }


def registros_test(
    jpg: Path, publicacion: str, regiones_gt: list[dict], dry_run: bool,
) -> tuple[list[dict], dict]:
    stem = jpg.stem
    stats = _stats_vacio()
    registros = []

    for gt in sorted(regiones_gt, key=lambda r: r["orden"]):
        if gt["tipo"] == "graphic":
            prompt, objetivo = PROMPT_GRAPHIC, MARCADOR_GRAPHIC
        elif gt["texto"].strip():
            prompt, objetivo = PROMPT_TEXT, gt["texto"]
        else:
            stats["gt_texto_vacio"] += 1
            continue

        ruta_crop = recortar_region(jpg, gt["bbox"], "test", dry_run)
        if ruta_crop is None:
            stats["recorte_fallido"] += 1
            continue

        stats["con_gt"] += 1
        registros.append(_registro(ruta_crop, prompt, objetivo, {
            "fuente":        stem,
            "publicacion":   publicacion,
            "split":         "test",
            "region_id":     gt["id"],
            "region_tipo":   gt["tipo"],
            "region_orden":  gt["orden"],
            "region_bbox":   gt["bbox"],
            "origen_layout": "gt",
            "tiene_gt":      True,
            "match_iou":     None,
            "es_portada":    _es_portada(stem),
        }))
    return registros, stats


def registros_train_val(
    jpg: Path, publicacion: str, split: str, regiones_gt: list[dict],
    cache: dict[str, list[dict]], iou_min: float, sin_cache: bool, dry_run: bool,
) -> tuple[list[dict], dict]:
    stem = jpg.stem
    stats = _stats_vacio()

    if sin_cache or stem not in cache:
        regiones_dl = [
            {"id": r["id"], "tipo": r["tipo"], "orden": r["orden"], "bbox": r["bbox"]}
            for r in regiones_gt
        ]
        origen_layout = "xml"
    else:
        regiones_dl   = cache[stem]
        origen_layout = "doclayout"

    if not regiones_dl:
        return [], {"omitido": "sin_regiones_layout"}

    registros = []
    bboxes_vistos: set[tuple] = set()

    for dl, gt, iou_val in _emparejar_regiones(regiones_dl, regiones_gt, iou_min):
        bbox_t = tuple(dl["bbox"])
        if bbox_t in bboxes_vistos:          # docLayout a veces devuelve cajas idénticas
            stats["duplicadas"] += 1
            continue
        bboxes_vistos.add(bbox_t)

        tipo_dl = dl["tipo"]                 # "text" | "graphic"

        if gt is None:
            if tipo_dl == "graphic":
                prompt, objetivo = PROMPT_GRAPHIC, MARCADOR_GRAPHIC
                stats["sin_gt_graphic"] += 1
            else:
                stats["sin_gt_text"] += 1    # no entrenamos con texto inventado
                continue
        else:
            if gt["tipo"] != tipo_dl:        # p. ej. docLayout=text, GT=graphic
                stats["tipo_discordante"] += 1
                continue
            if tipo_dl == "graphic":
                prompt, objetivo = PROMPT_GRAPHIC, MARCADOR_GRAPHIC
            elif gt["texto"].strip():
                prompt, objetivo = PROMPT_TEXT, gt["texto"]
            else:
                stats["gt_texto_vacio"] += 1
                continue
            stats["con_gt"] += 1

        ruta_crop = recortar_region(jpg, dl["bbox"], split, dry_run)
        if ruta_crop is None:
            stats["recorte_fallido"] += 1
            continue

        registros.append(_registro(ruta_crop, prompt, objetivo, {
            "fuente":        stem,
            "publicacion":   publicacion,
            "split":         split,
            "region_id":     dl["id"],
            "region_tipo":   tipo_dl,
            "region_orden":  dl["orden"],
            "region_bbox":   dl["bbox"],
            "origen_layout": origen_layout,
            "tiene_gt":      gt is not None,
            "match_iou":     round(iou_val, 3) if gt is not None else None,
            "es_portada":    _es_portada(stem),
        }))
    return registros, stats


def procesar_pagina(
    jpg: Path, xml: Path, publicacion: str, split: str,
    cache: dict[str, list[dict]], iou_min: float, sin_cache: bool, dry_run: bool,
) -> tuple[list[dict], dict]:
    regiones_gt = xml_a_regiones(xml)
    if not regiones_gt:
        return [], {"omitido": "xml_sin_regiones"}

    if split == "test":
        return registros_test(jpg, publicacion, regiones_gt, dry_run)
    return registros_train_val(
        jpg, publicacion, split, regiones_gt, cache, iou_min, sin_cache, dry_run
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — DESCUBRIMIENTO DE PARES JPG+XML
# ══════════════════════════════════════════════════════════════════════════════

def descubrir_pares_publicacion(
    dir_publicacion: Path, nombre: str,
) -> list[tuple[Path, Path, str]]:
    dir_jpg  = dir_publicacion / "jpg"
    dir_page = dir_publicacion / "page"

    if not dir_jpg.exists():
        print(f"  [AVISO] {nombre}: no se encontró {dir_jpg}")
        return []
    if not dir_page.exists():
        print(f"  [AVISO] {nombre}: no se encontró {dir_page}")
        return []

    pares, sin_xml = [], []
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
# SECCIÓN 6 — VERIFICACIÓN CONTRA split.json
# ══════════════════════════════════════════════════════════════════════════════

def verificar(datasets: dict[str, list[dict]]) -> bool:
    ok = True
    stems = {sp: {r["_meta"]["fuente"] for r in regs} for sp, regs in datasets.items()}

    print(f"\n  Verificación contra {SPLIT_PATH.name}:")

    for sp, ss in stems.items():
        mal = sorted(s for s in ss if SPLIT_MAP.get(s, "train") != sp)
        if mal:
            ok = False
            print(f"    [ERROR] {sp}: {len(mal)} páginas que split.json asigna a otro split: {mal[:5]}")

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = sorted(stems[a] & stems[b])
        if inter:
            ok = False
            print(f"    [ERROR] {len(inter)} páginas en {a} y {b} a la vez: {inter[:5]}")

    for sp, regs in datasets.items():
        fuera = [r["image"] for r in regs if Path(r["image"]).parent.name != sp]
        if fuera:
            ok = False
            print(f"    [ERROR] {sp}: {len(fuera)} recortes fuera de crops/{sp}/ (p. ej. {fuera[0]})")

    for sp in ("val", "test"):
        esperadas = {s for s, v in SPLIT_MAP.items() if v == sp}
        faltan = sorted(esperadas - stems[sp])
        estado = "OK" if not faltan else ("ERROR" if sp == "test" else "AVISO")
        print(f"    {sp:<5} esperadas={len(esperadas):>3}  con registros={len(stems[sp]):>3}  [{estado}]")
        if faltan:
            print(f"          sin registros: {faltan[:8]}{' …' if len(faltan) > 8 else ''}")
            if sp == "test":
                ok = False

    return ok


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — ESCRITURA Y RESUMEN
# ══════════════════════════════════════════════════════════════════════════════

def escribir_jsonl(registros: list[dict], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def imprimir_resumen(por_pub: dict, datasets: dict, stats_total: dict, omitidas: dict) -> None:
    print(f"\n{'═'*86}")
    print(f"  {'PUBLICACIÓN':<26} {'PÁGINAS tr/val/test':>22}   {'REGIONES tr/val/test':>24}")
    print(f"{'─'*86}")
    for nombre, s in sorted(por_pub.items()):
        pag  = "/".join(str(s[sp]["paginas"]) for sp in SPLITS)
        regs = "/".join(str(s[sp]["regs"])    for sp in SPLITS)
        print(f"  {nombre:<26} {pag:>22}   {regs:>24}")
    print(f"{'─'*86}")
    tp = "/".join(str(len({r['_meta']['fuente'] for r in datasets[sp]})) for sp in SPLITS)
    tr = "/".join(str(len(datasets[sp])) for sp in SPLITS)
    print(f"  {'TOTAL':<26} {tp:>22}   {tr:>24}")

    todos = [r for sp in SPLITS for r in datasets[sp]]
    n = len(todos)
    for titulo, campo in (("Tipos de región", "region_tipo"), ("Origen del layout", "origen_layout")):
        cont: dict[str, int] = defaultdict(int)
        for r in todos:
            cont[r["_meta"][campo]] += 1
        print(f"\n  {titulo}:")
        for k, v in sorted(cont.items(), key=lambda x: -x[1]):
            print(f"    {k:<22} {v:>6}  ({v / n * 100:.1f} %)" if n else f"    {k}: {v}")

    print("\n  Regiones descartadas o incluidas por regla (train + val + test):")
    etiquetas = {
        "con_gt": "emparejadas con GT", "sin_gt_graphic": "gráficas sin GT (marcador)",
        "sin_gt_text": "texto sin GT (descartadas)", "tipo_discordante": "tipo docLayout ≠ GT (descartadas)",
        "gt_texto_vacio": "GT sin texto (descartadas)", "duplicadas": "cajas docLayout duplicadas",
        "recorte_fallido": "recorte fallido",
    }
    for k, etq in etiquetas.items():
        print(f"    {etq:<38} {stats_total[k]:>6}")

    for sp in SPLITS:
        if omitidas[sp]:
            print(f"\n  Páginas omitidas en {sp} ({len(omitidas[sp])}): "
                  + ", ".join(f"{s} [{m}]" for s, m in omitidas[sp][:6])
                  + (" …" if len(omitidas[sp]) > 6 else ""))
    print(f"{'═'*86}")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Construye los JSONL (un registro por región) usando split.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus",      default=None)
    parser.add_argument("--cache",       default=None,
                        help=f"layout_regions_cache.json (default: {CACHE_PATH})")
    parser.add_argument("--publicacion", default=None)
    parser.add_argument("--semilla",     type=int, default=config.SPLIT_SEED,
                        help="Solo afecta al orden de mezcla del train (el split viene de split.json)")
    parser.add_argument("--iou",         type=float, default=IOU_DEFAULT,
                        help=f"Umbral IoU docLayout↔GT (default: {IOU_DEFAULT})")
    parser.add_argument("--sin-cache",   action="store_true",
                        help="Usar como layout las regiones del XML")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Procesar y verificar sin escribir recortes ni JSONL")
    parser.add_argument("--conservar-recortes", action="store_true",
                        help="No vaciar dataset/crops/ antes de empezar")
    parser.add_argument("--forzar",      action="store_true",
                        help="Escribir los JSONL aunque la verificación falle")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus) if args.corpus else config.CORPUS_DIR
    cache_path = Path(args.cache)  if args.cache  else CACHE_PATH
    if not corpus_dir.exists():
        print(f"ERROR: corpus no encontrado: {corpus_dir}")
        sys.exit(1)

    rng = random.Random(args.semilla)

    # ── Caché ─────────────────────────────────────────────────────────────────
    if args.sin_cache:
        print("Modo --sin-cache: layout = regiones del XML.\n")
        cache: dict[str, list[dict]] = {}
    else:
        print("Cargando caché de regiones de layout …")
        cache = cargar_cache(cache_path)

    # ── Publicaciones ─────────────────────────────────────────────────────────
    publicaciones = config.PUBLICACIONES
    if args.publicacion:
        if args.publicacion not in publicaciones:
            print(f"ERROR: '{args.publicacion}' no está en config.PUBLICACIONES")
            sys.exit(1)
        publicaciones = {args.publicacion: publicaciones[args.publicacion]}

    # ── Recortes antiguos: pueden ser de otro docLayout u otro split ──────────
    if not args.dry_run and not args.conservar_recortes and DIR_CROPS.exists():
        print(f"Vaciando {DIR_CROPS} (recortes de ejecuciones anteriores) …")
        shutil.rmtree(DIR_CROPS)

    datasets: dict[str, list[dict]] = {sp: [] for sp in SPLITS}
    omitidas: dict[str, list[tuple[str, str]]] = {sp: [] for sp in SPLITS}
    por_pub = defaultdict(lambda: {sp: {"paginas": 0, "regs": 0} for sp in SPLITS})
    stats_total = _stats_vacio()
    stems_vistos: dict[str, str] = {}
    sin_split: list[str] = []

    print(f"\nProcesando {len(publicaciones)} publicaciones …\n")

    for nombre in publicaciones:
        dir_pub = corpus_dir / nombre
        if not dir_pub.exists():
            print(f"  [AVISO] Carpeta no encontrada, saltando: {dir_pub}")
            continue

        pares = descubrir_pares_publicacion(dir_pub, nombre)
        if not pares:
            print(f"  [AVISO] {nombre}: sin pares JPG+XML, saltando.")
            continue

        for jpg, xml, pub in pares:
            stem = jpg.stem
            if stem in stems_vistos and stems_vistos[stem] != nombre:
                print(f"  [AVISO] stem repetido entre publicaciones: {stem} "
                      f"({stems_vistos[stem]} y {nombre})")
            stems_vistos[stem] = nombre

            # El split se decide UNA vez por página, antes de procesarla
            split = SPLIT_MAP.get(stem)
            if split is None:
                split = "train"
                sin_split.append(stem)

            regs, s = procesar_pagina(
                jpg, xml, pub, split, cache, args.iou, args.sin_cache, args.dry_run
            )
            if "omitido" in s:
                omitidas[split].append((stem, s["omitido"]))
                continue
            if not regs:
                omitidas[split].append((stem, "sin_registros_validos"))
                continue

            datasets[split].extend(regs)
            por_pub[nombre][split]["paginas"] += 1
            por_pub[nombre][split]["regs"]    += len(regs)
            for k in stats_total:
                stats_total[k] += s.get(k, 0)

        s = por_pub[nombre]
        print(f"  {nombre:<28} regs  train={s['train']['regs']:>4}  "
              f"val={s['val']['regs']:>3}  test={s['test']['regs']:>3}")

    if sin_split:
        print(f"\n  [AVISO] {len(sin_split)} páginas no están en split.json → train: "
              + ", ".join(sin_split[:5]) + (" …" if len(sin_split) > 5 else ""))

    if not datasets["train"]:
        print("ERROR: no se generó ningún registro de train.")
        sys.exit(1)

    # Train mezclado; val/test agrupados por página en orden de lectura
    rng.shuffle(datasets["train"])
    for sp in ("val", "test"):
        datasets[sp].sort(key=lambda r: (r["_meta"]["fuente"], r["_meta"]["region_orden"]))

    imprimir_resumen(por_pub, datasets, stats_total, omitidas)
    ok = verificar(datasets) if not args.publicacion else True

    if args.dry_run:
        print("\n[DRY-RUN] No se han escrito recortes ni JSONL.")
        return

    if not ok and not args.forzar:
        print("\nERROR: la verificación contra split.json ha fallado (ver arriba). "
              "No se escriben los JSONL. Usa --forzar para escribirlos igualmente.")
        sys.exit(1)

    escribir_jsonl(datasets["train"], config.JSONL_TRAIN)
    escribir_jsonl(datasets["val"],   config.JSONL_VALIDATION)
    escribir_jsonl(datasets["test"],  config.JSONL_TEST)

    # Auditoría: qué páginas hay en cada split del dataset generado
    ruta_aud = config.DATASET_DIR / "paginas_por_split.json"
    with open(ruta_aud, "w", encoding="utf-8") as f:
        json.dump(
            {sp: sorted({r["_meta"]["fuente"] for r in datasets[sp]}) for sp in SPLITS},
            f, ensure_ascii=False, indent=2,
        )

    print(f"\nJSONL escritos en: {config.DATASET_DIR}")
    print(f"  train      : {config.JSONL_TRAIN}")
    print(f"  validation : {config.JSONL_VALIDATION}")
    print(f"  test       : {config.JSONL_TEST}")
    print(f"  auditoría  : {ruta_aud}")
    print(f"\nSiguiente paso: python scripts/04_train_lora.py")


if __name__ == "__main__":
    main()