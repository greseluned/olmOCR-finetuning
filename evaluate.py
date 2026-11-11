#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
evaluate.py

Evaluación única para todos los flujos (olmOCR baseline y fine-tuneado, con TXT
o PAGE XML, con o sin docLayout). Calcula CER y WER por página y los agrega en
total, promedio (macro) y normalizado (micro), sobre todo el test set y por
publicación. Con docLayout evalúa además las cajas (IoU, P/R/F1 por clase, EMR y
composite score). Es independiente: no importa config.py.

Cada flujo debe tener las predicciones (<stem>.txt o <stem>.xml) en
<flujo>/outputs/transcripciones/ y, si usa docLayout,
<flujo>/outputs/predicciones/pred_boxes_doclayout.json. El GT es un único JSONL
(--jsonl) de página o de recorte.

Requisitos:
    pip install editdistance numpy

Uso (en Linux/macOS usa \ en lugar de ^):
    python evaluate.py --jsonl C:\ruta\test.jsonl --root C:\ruta\olmOCR-finetuning
    python evaluate.py --jsonl C:\ruta\test.jsonl ^
        --workflow C:\ruta\olmOCR-finetuning\olmOCR-base ^
                   C:\ruta\olmOCR-finetuning\fine-tuned_prompt-type

Opciones:
    --jsonl RUTA                 JSONL con el GT del split a evaluar (obligatorio)
    --root RUTA                  carpeta con los flujos (se evalúan todos los hijos)
    --workflow RUTA [RUTA ...]   carpetas de flujo concretas
    --formato txt|xml            fuerza el formato de las predicciones
    --dir-eval RUTA              dónde guardar comparativa_metricas.json
    --dir-paginas RUTA           raíz de las imágenes, para resolver la publicación
    --pub-regex REGEX            regex sobre el stem; el primer grupo es la publicación
    --faltantes-vacios           cuenta las páginas sin predicción como predicción vacía
    --conservar-marcas-columna   no elimina las marcas ---COLUMNA N---
    --gt-boxes RUTA              gt_boxes.json de docLayout
    --iou-threshold X            umbral IoU de docLayout (0.5)
    --alpha X                    peso del IoU en el composite score (0.5)
    --dup-iou X                  IoU para considerar duplicada una caja (0.7)

Salida:
    <flujo>/outputs/metricas/{metrics_ocr.json, metrics_layout_boxes.json,
    metrics_summary.json} y comparativa_metricas.json si hay dos o más flujos.
"""

import argparse
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 0 — DEPENDENCIAS
# ══════════════════════════════════════════════════════════════════════════════

def verificar_dependencias() -> None:
    try:
        import editdistance  # noqa: F401
    except ImportError:
        print("ERROR: pip install editdistance")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — NORMALIZACIÓN Y MÉTRICAS DE TEXTO (CER / WER)
# ══════════════════════════════════════════════════════════════════════════════

_PATRON_COLUMNA = re.compile(r"---\s*COLUMNA\s+\d+\s*---", re.IGNORECASE)


def normalizar_texto(texto: str, quitar_marcas_columna: bool = True) -> str:
    texto = unicodedata.normalize("NFC", texto or "")
    if quitar_marcas_columna:
        texto = _PATRON_COLUMNA.sub(" ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def metricas_pagina(pred: str, gt: str) -> dict:
    import editdistance
    pw, gw = pred.split(), gt.split()
    lev_c = editdistance.eval(pred, gt)
    lev_w = editdistance.eval(pw, gw)
    return {
        "errores_caracteres": lev_c,
        "caracteres_gt":      len(gt),
        "errores_palabras":   lev_w,
        "palabras_gt":        len(gw),
        "cer":                lev_c / len(gt),
        "wer":                lev_w / len(gw),
    }


def agregar(paginas: list[dict]) -> dict:
    n = len(paginas)
    if n == 0:
        return {"n_paginas": 0}
    ec = sum(p["errores_caracteres"] for p in paginas)
    cg = sum(p["caracteres_gt"]      for p in paginas)
    ew = sum(p["errores_palabras"]   for p in paginas)
    wg = sum(p["palabras_gt"]        for p in paginas)
    return {
        "n_paginas": n,
        "total": {
            "errores_caracteres": ec,
            "caracteres_gt":      cg,
            "errores_palabras":   ew,
            "palabras_gt":        wg,
        },
        "promedio": {
            "cer": round(sum(p["cer"] for p in paginas) / n, 4),
            "wer": round(sum(p["wer"] for p in paginas) / n, 4),
        },
        "normalizado": {
            "cer": round(ec / cg, 4) if cg else None,
            "wer": round(ew / wg, 4) if wg else None,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — PAGE XML → TEXTO (orden de lectura)
# ══════════════════════════════════════════════════════════════════════════════

def _texto_region(el) -> str:
    lineas = []
    for k, tl in enumerate(el.iterfind("{*}TextLine")):
        u = tl.find("{*}TextEquiv/{*}Unicode")
        if u is None or not (u.text or "").strip():
            continue
        m = re.search(r"index\s*:\s*(\d+)", tl.get("custom", ""))
        lineas.append((int(m.group(1)) if m else 10**6 + k, u.text.strip()))
    if lineas:
        lineas.sort(key=lambda x: x[0])
        return "\n".join(t for _, t in lineas)
    u = el.find("{*}TextEquiv/{*}Unicode")
    return (u.text or "") if u is not None else ""


def texto_desde_page_xml(contenido) -> tuple[str, dict] | None:
    if isinstance(contenido, str):
        contenido = re.sub(r"^\s*```(?:xml)?\s*|\s*```\s*$", "", contenido.strip())
        contenido = contenido.encode("utf-8")
    try:
        root = ET.fromstring(contenido)
    except ET.ParseError:
        return None
    page = root.find(".//{*}Page")
    if page is None:
        return None

    orden_ref: dict[str, int] = {}
    for ref in root.iterfind(".//{*}RegionRefIndexed"):
        try:
            orden_ref[ref.get("regionRef")] = int(ref.get("index"))
        except (TypeError, ValueError):
            continue

    regiones = []
    for k, el in enumerate(page):
        if not el.tag.rsplit("}", 1)[-1].endswith("Region"):
            continue
        idx = orden_ref.get(el.get("id", f"r_{k}"))
        if idx is None:
            m = re.search(r"index\s*:\s*(\d+)", el.get("custom", ""))
            idx = int(m.group(1)) if m else 10**6 + k
        regiones.append((idx, k, _texto_region(el)))
    regiones.sort(key=lambda r: (r[0], r[1]))
    texto = "\n".join(r[2] for r in regiones)
    return texto, {"imageFilename": page.get("imageFilename", "")}


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — PUBLICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

_EXT_IMG = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
_CARPETAS_IMG = {"jpg", "jpeg", "png", "img", "imgs", "images", "imagenes"}


def _indexar_imagenes(dir_paginas: Path | None) -> dict[str, Path]:
    if dir_paginas is None:
        return {}
    idx: dict[str, Path] = {}
    duplicados = []
    for p in dir_paginas.rglob("*"):
        if p.suffix.lower() not in _EXT_IMG:
            continue
        if p.stem in idx:
            duplicados.append(p.stem)
        idx[p.stem] = p
    if duplicados:
        print(f"  [AVISO] {len(duplicados)} nombres de imagen repetidos entre "
              f"carpetas (p. ej. {duplicados[:3]}): la publicación de esas "
              f"páginas puede resolverse mal")
    return idx


def _pub_desde_ruta(ruta: Path | None) -> str | None:
    if ruta is not None and ruta.parent.name.lower() in _CARPETAS_IMG:
        nombre = unicodedata.normalize("NFC", ruta.parent.parent.name)
        return nombre or None
    return None


def resolver_publicacion(
    meta: dict, stem: str, ruta_imagen_jsonl: str,
    idx_imgs: dict[str, Path], pub_regex: re.Pattern | None,
    pub_map: dict[str, str] | None = None,
) -> str:
    pub = meta.get("publicacion") or meta.get("publication")
    if pub:
        return str(pub)
    if pub_map:
        pub = pub_map.get(unicodedata.normalize("NFC", stem))
        if pub:
            return pub
    pub = _pub_desde_ruta(idx_imgs.get(stem))
    if pub:
        return pub
    if ruta_imagen_jsonl:
        pub = _pub_desde_ruta(Path(ruta_imagen_jsonl.replace("\\", "/")))
        if pub:
            return pub
    if pub_regex:
        m = pub_regex.search(stem)
        if m:
            return m.group(1)
    return "desconocida"


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — CARGA DE GT Y PREDICCIONES
# ══════════════════════════════════════════════════════════════════════════════

def cargar_gt(jsonl_path: Path, idx_imgs: dict, pub_regex,
              pub_map: dict[str, str] | None = None) -> dict[str, dict]:
    paginas_recorte: dict[str, list] = defaultdict(list)
    pub_recorte: dict[str, str] = {}
    gt: dict[str, dict] = {}

    with open(jsonl_path, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            try:
                reg = json.loads(linea)
            except json.JSONDecodeError:
                continue
            meta = reg.get("_meta") or {}
            if not meta.get("tiene_gt", True):
                continue
            texto = next(
                (c["content"] for c in reg.get("conversations", [])
                 if c.get("role") == "assistant"), ""
            )
            img = reg.get("image", "")

            if meta.get("region_orden") is not None and meta.get("fuente"):
                fuente = meta["fuente"]
                paginas_recorte[fuente].append((meta["region_orden"], texto))
                if fuente not in pub_recorte:
                    pub_recorte[fuente] = resolver_publicacion(
                        meta, fuente, img, idx_imgs, pub_regex, pub_map)
            else:
                stem = meta.get("fuente") or Path(img).stem
                gt[stem] = {
                    "texto": texto,
                    "publicacion": resolver_publicacion(
                        meta, stem, img, idx_imgs, pub_regex, pub_map),
                }

    for fuente, regs in paginas_recorte.items():
        regs.sort(key=lambda r: r[0])
        gt[fuente] = {
            "texto": "\n".join(t for _, t in regs),
            "publicacion": pub_recorte[fuente],
        }
    return gt


def cargar_predicciones(dir_pred: Path, formato: str) -> tuple[dict, dict]:
    preds: dict[str, str | None] = {}
    alias: dict[str, str] = {}

    if formato == "txt":
        for ruta in sorted(dir_pred.glob("*.txt")):
            preds[ruta.stem] = ruta.read_text(encoding="utf-8")
    else:
        for ruta in sorted(dir_pred.glob("*.xml")):
            r = texto_desde_page_xml(ruta.read_bytes())
            if r is None:
                print(f"  [AVISO] {ruta.name}: XML inválido")
                preds[ruta.stem] = None
                continue
            texto, info = r
            preds[ruta.stem] = texto
            stem_img = Path(info["imageFilename"]).stem
            if stem_img and stem_img != ruta.stem:
                alias[stem_img] = ruta.stem
    return preds, alias


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — EVALUACIÓN OCR (CER / WER)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_ocr(
    gt: dict[str, dict], preds: dict, alias: dict,
    faltantes_vacios: bool, quitar_marcas: bool,
) -> dict:
    paginas: list[dict] = []
    sin_pred: list[str] = []
    gt_vacio: list[str] = []

    for stem, g in sorted(gt.items()):
        gt_txt = normalizar_texto(g["texto"], quitar_marcas)
        if not gt_txt or not gt_txt.split():
            gt_vacio.append(stem)
            continue

        clave = stem if stem in preds else alias.get(stem)
        pred_raw = preds.get(clave) if clave is not None else None
        if pred_raw is None:
            if not faltantes_vacios:
                sin_pred.append(stem)
                continue
            pred_raw = ""

        m = metricas_pagina(normalizar_texto(pred_raw, quitar_marcas), gt_txt)
        m["stem"] = stem
        m["publicacion"] = g["publicacion"]
        m["cer"] = round(m["cer"], 4)
        m["wer"] = round(m["wer"], 4)
        paginas.append(m)

    if gt_vacio:
        print(f"  [AVISO] {len(gt_vacio)} páginas con GT vacío, omitidas")
    if sin_pred:
        print(f"  [AVISO] {len(sin_pred)} páginas GT sin predicción, omitidas "
              f"(usa --faltantes-vacios para contarlas como predicción vacía)")
    if not paginas:
        return {"error": "Ninguna predicción coincidió con el GT"}

    por_pub: dict[str, list] = defaultdict(list)
    for p in paginas:
        por_pub[p["publicacion"]].append(p)

    paginas_ord = sorted(paginas, key=lambda p: p["cer"], reverse=True)
    return {
        "global": agregar(paginas),
        "por_publicacion": {pub: agregar(ps) for pub, ps in sorted(por_pub.items())},
        "paginas_omitidas": {"sin_prediccion": sin_pred, "gt_vacio": gt_vacio},
        "por_pagina": paginas_ord,  # peores primero
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — DOCLAYOUT: BOUNDING BOXES (sin cambios)
# ══════════════════════════════════════════════════════════════════════════════

def _iou(box1: list, box2: list) -> float:
    xA = max(box1[0], box2[0]);  yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]);  yB = min(box1[3], box2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0.0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def _evaluar_pagina_boxes(gt_boxes: list[dict], pred_boxes: list[dict],
                          iou_threshold: float, dup_iou: float = 0.7) -> dict:
    if not gt_boxes:
        return {"avg_iou": 0.0, "emr": 0.0, "metrics_by_type": {}}

    matches = []
    pred_usadas: set[int] = set()

    for gt in gt_boxes:
        mejor_iou, mejor_pred, mejor_idx = 0.0, None, -1
        for idx, pred in enumerate(pred_boxes):
            v = _iou(gt["xyxy"], pred["xyxy"])
            if v > mejor_iou:
                mejor_iou, mejor_pred, mejor_idx = v, pred, idx
        matches.append((gt, mejor_pred, mejor_iou, mejor_idx))

    avg_iou = sum(m[2] for m in matches) / len(matches)
    type_stats: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    tp_total = 0

    for gt, pred, iou_val, pred_idx in matches:
        cls = gt["class_name"]
        if pred is not None and iou_val >= iou_threshold:
            if pred["class_name"] == cls:
                type_stats[cls]["tp"] += 1
                pred_usadas.add(pred_idx)
                tp_total += 1
            else:
                type_stats[cls]["fn"] += 1
                type_stats[pred["class_name"]]["fp"] += 1
                pred_usadas.add(pred_idx)
        else:
            type_stats[cls]["fn"] += 1

    for idx, pred in enumerate(pred_boxes):
        if idx not in pred_usadas:
            type_stats[pred["class_name"]]["fp"] += 1

    metrics_by_type = {}
    for cls, s in type_stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics_by_type[cls] = {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    # ── Diagnóstico (NO afecta a ninguna métrica) ────────────────────────────
    # Cuántas cajas predichas han sido emparejadas con más de una caja GT.
    por_pred: dict[int, dict] = defaultdict(
        lambda: {"gt_match": 0, "tp": 0, "clase": None})
    for gt, pred, iou_val, pred_idx in matches:
        if pred is not None and iou_val >= iou_threshold:
            d = por_pred[pred_idx]
            d["gt_match"] += 1
            d["clase"] = pred["class_name"]
            if pred["class_name"] == gt["class_name"]:
                d["tp"] += 1
    multi_match: dict[str, dict] = {}
    for d in por_pred.values():
        if d["gt_match"] > 1 or d["tp"] > 1:
            m = multi_match.setdefault(d["clase"], {"preds_multi": 0, "tp_extra": 0})
            if d["gt_match"] > 1:
                m["preds_multi"] += 1   # predicciones usadas por >1 GT (cualquier clase)
            if d["tp"] > 1:
                m["tp_extra"] += d["tp"] - 1  # TP que sobran si cada pred contara una vez

    # Cajas predichas duplicadas (diagnóstico; no toca ninguna métrica):
    #   dup       → solapa > dup_iou con otra predicha ANTERIOR de la misma clase
    #   dup_de_tp → es un falso positivo que solapa > dup_iou con una predicción
    #               correcta (TP) de su misma clase: un duplicado "puro" de una
    #               caja bien detectada; quitarlo no cambia los TP.
    tp_idx = {
        pred_idx for gt, pred, iou_val, pred_idx in matches
        if pred is not None and iou_val >= iou_threshold
        and pred["class_name"] == gt["class_name"]
    }
    n_pred = len(pred_boxes)
    duplicadas: dict[str, dict] = {}
    n_pred_por_clase: dict[str, int] = defaultdict(int)
    for j, pj in enumerate(pred_boxes):
        cls = pj["class_name"]
        n_pred_por_clase[cls] += 1
        es_dup = any(
            pred_boxes[k]["class_name"] == cls
            and _iou(pj["xyxy"], pred_boxes[k]["xyxy"]) > dup_iou
            for k in range(j)
        )
        es_dup_de_tp = j not in tp_idx and any(
            k in tp_idx and pred_boxes[k]["class_name"] == cls
            and _iou(pj["xyxy"], pred_boxes[k]["xyxy"]) > dup_iou
            for k in range(n_pred) if k != j
        )
        if es_dup or es_dup_de_tp:
            d = duplicadas.setdefault(cls, {"dup": 0, "dup_de_tp": 0})
            d["dup"]       += int(es_dup)
            d["dup_de_tp"] += int(es_dup_de_tp)

    return {
        "avg_iou":         round(avg_iou, 4),
        "emr":             round(tp_total / len(gt_boxes), 4),
        "metrics_by_type": metrics_by_type,
        "multi_match":     multi_match,
        "duplicadas":      duplicadas,
        "n_pred_por_clase": dict(n_pred_por_clase),
    }


def evaluate_layout_boxes(gt_path: Path, pred_path: Path,
                          iou_threshold: float = 0.5, alpha: float = 0.5,
                          dup_iou: float = 0.7) -> dict:
    if not gt_path.exists():
        return {"error": f"gt_boxes.json no encontrado: {gt_path}"}
    if not pred_path.exists():
        return {"error": f"pred_boxes.json no encontrado: {pred_path}"}

    gt_data   = json.loads(gt_path.read_text(encoding="utf-8"))
    pred_data = json.loads(pred_path.read_text(encoding="utf-8"))

    sin_pred = [k for k in gt_data if k not in pred_data]
    sin_gt   = [k for k in pred_data if k not in gt_data]
    if sin_pred:
        print(f"  [AVISO] {len(sin_pred)} páginas GT sin predicción de boxes")
    if sin_gt:
        print(f"  [AVISO] {len(sin_gt)} páginas pred sin GT (ignoradas)")

    try:
        import numpy as np
        _mean = lambda lst: float(np.mean(lst)) if lst else 0.0
    except ImportError:
        _mean = lambda lst: sum(lst) / len(lst) if lst else 0.0

    per_page, ious_global = [], []
    for page_id, gt_boxes in sorted(gt_data.items()):
        m = _evaluar_pagina_boxes(gt_boxes, pred_data.get(page_id, []),
                                  iou_threshold, dup_iou)
        m["page_id"] = page_id
        per_page.append(m)
        ious_global.append(m["avg_iou"])

    agg: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for page in per_page:
        for cls, m in page["metrics_by_type"].items():
            for k in ("tp", "fp", "fn"):
                agg[cls][k] += m[k]

    global_by_type: dict[str, dict] = {}
    for cls, s in sorted(agg.items()):
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        global_by_type[cls] = {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    mean_f1   = _mean([m["f1"] for m in global_by_type.values()])
    mean_iou  = _mean(ious_global)
    mean_emr  = _mean([p["emr"] for p in per_page])
    composite = alpha * mean_iou + (1 - alpha) * mean_f1

    # Diagnóstico de emparejamientos múltiples (no cambia ninguna métrica)
    mm_cls: dict[str, dict] = defaultdict(lambda: {"preds_multi": 0, "tp_extra": 0})
    afectadas = []
    for page in per_page:
        mm = page.get("multi_match", {})
        pm = sum(v["preds_multi"] for v in mm.values())
        te = sum(v["tp_extra"] for v in mm.values())
        for cls, v in mm.items():
            mm_cls[cls]["preds_multi"] += v["preds_multi"]
            mm_cls[cls]["tp_extra"]    += v["tp_extra"]
        if pm or te:
            afectadas.append({"page_id": page["page_id"],
                              "preds_multi": pm, "tp_extra": te})
    afectadas.sort(key=lambda a: (a["tp_extra"], a["preds_multi"]), reverse=True)

    tp_all = sum(m["tp"] for m in global_by_type.values())
    extra_all = sum(v["tp_extra"] for v in mm_cls.values())
    multi_global = {
        "tp_total":            tp_all,
        "tp_extra":            extra_all,
        "tp_sin_duplicar":     tp_all - extra_all,
        "pct_tp_extra":        round(extra_all / tp_all, 4) if tp_all else 0.0,
        "preds_multi":         sum(v["preds_multi"] for v in mm_cls.values()),
        "paginas_afectadas":   len(afectadas),
        "n_paginas":           len(per_page),
        "por_clase": {
            cls: {
                "tp":              global_by_type.get(cls, {}).get("tp", 0),
                "tp_extra":        v["tp_extra"],
                "preds_multi":     v["preds_multi"],
                "pct_tp_extra":    round(v["tp_extra"] / global_by_type[cls]["tp"], 4)
                                   if global_by_type.get(cls, {}).get("tp") else 0.0,
            }
            for cls, v in sorted(mm_cls.items())
        },
        "peores_paginas":      afectadas[:10],
    }

    # Diagnóstico de cajas predichas duplicadas (no cambia ninguna métrica)
    dup_cls: dict[str, dict] = defaultdict(
        lambda: {"preds": 0, "dup": 0, "dup_de_tp": 0})
    dup_paginas = []
    for page in per_page:
        for cls, n in page.get("n_pred_por_clase", {}).items():
            dup_cls[cls]["preds"] += n
        pd_, pt_ = 0, 0
        for cls, v in page.get("duplicadas", {}).items():
            dup_cls[cls]["dup"]       += v["dup"]
            dup_cls[cls]["dup_de_tp"] += v["dup_de_tp"]
            pd_ += v["dup"]; pt_ += v["dup_de_tp"]
        if pd_ or pt_:
            dup_paginas.append({"page_id": page["page_id"],
                                "duplicadas": pd_, "fp_duplicados_de_tp": pt_})
    dup_paginas.sort(key=lambda a: (a["fp_duplicados_de_tp"], a["duplicadas"]),
                     reverse=True)

    preds_all = sum(v["preds"] for v in dup_cls.values())
    dup_all   = sum(v["dup"] for v in dup_cls.values())
    dtp_all   = sum(v["dup_de_tp"] for v in dup_cls.values())
    fp_all    = sum(m["fp"] for m in global_by_type.values())

    def _prec_sin_dup(cls: str):
        g = global_by_type.get(cls)
        if not g:
            return None
        den = g["tp"] + g["fp"] - dup_cls[cls]["dup_de_tp"]
        return round(g["tp"] / den, 4) if den > 0 else None

    dup_global = {
        "dup_iou":                 dup_iou,
        "preds_total":             preds_all,
        "duplicadas":              dup_all,
        "pct_duplicadas":          round(dup_all / preds_all, 4) if preds_all else 0.0,
        "fp_total":                fp_all,
        "fp_duplicados_de_tp":     dtp_all,
        "pct_fp_duplicados_de_tp": round(dtp_all / fp_all, 4) if fp_all else 0.0,
        "paginas_afectadas":       len(dup_paginas),
        "n_paginas":               len(per_page),
        "por_clase": {
            cls: {
                "preds":               v["preds"],
                "duplicadas":          v["dup"],
                "fp":                  global_by_type.get(cls, {}).get("fp", 0),
                "fp_duplicados_de_tp": v["dup_de_tp"],
                "precision":           global_by_type.get(cls, {}).get("precision"),
                "precision_sin_duplicados_aprox": _prec_sin_dup(cls),
            }
            for cls, v in sorted(dup_cls.items())
        },
        "peores_paginas":          dup_paginas[:10],
    }

    return {
        "global_metrics": {
            "n_paginas":       len(per_page),
            "avg_iou":         round(mean_iou, 4),
            "mean_f1":         round(mean_f1, 4),
            "mean_emr":        round(mean_emr, 4),
            "composite_score": round(composite, 4),
            "iou_threshold":   iou_threshold,
            "alpha":           alpha,
        },
        "global_by_type":   global_by_type,
        "multi_match":      multi_global,
        "duplicadas":       dup_global,
        "per_page_metrics": per_page,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — RESUMEN E IMPRESIÓN
# ══════════════════════════════════════════════════════════════════════════════

def generar_resumen(etiqueta: str, metrics_ocr: dict | None,
                    metrics_boxes: dict | None, origen_boxes: str | None = None) -> dict:
    resumen: dict = {"etiqueta": etiqueta}
    if metrics_ocr is None:
        pass
    elif "error" in metrics_ocr:
        resumen["ocr"] = {"error": metrics_ocr["error"]}
    else:
        resumen["ocr"] = {
            "global":          metrics_ocr["global"],
            "por_publicacion": metrics_ocr["por_publicacion"],
            "peores_paginas": [
                {"stem": p["stem"], "publicacion": p["publicacion"],
                 "cer": p["cer"], "wer": p["wer"]}
                for p in metrics_ocr["por_pagina"][:6]
            ],
        }
    if metrics_boxes is not None:
        if "global_metrics" in metrics_boxes:
            resumen["layout_boxes"] = metrics_boxes["global_metrics"]
            resumen["layout_boxes_por_clase"] = metrics_boxes.get("global_by_type", {})
            resumen["layout_boxes_multi_match"] = metrics_boxes.get("multi_match", {})
            resumen["layout_boxes_duplicadas"] = metrics_boxes.get("duplicadas", {})
            if origen_boxes:
                resumen["layout_boxes_origen"] = origen_boxes
        else:
            resumen["layout_boxes"] = {"error": metrics_boxes.get("error", "sin datos")}
    return resumen


def _fila(nombre: str, a: dict, ancho: int) -> str:
    t, p, n = a["total"], a["promedio"], a["normalizado"]
    return (f"    {nombre[:ancho]:<{ancho}} {a['n_paginas']:>5}  "
            f"{t['errores_caracteres']:>8}/{t['caracteres_gt']:<8} "
            f"{t['errores_palabras']:>7}/{t['palabras_gt']:<7} "
            f"{p['cer']:>8.4f} {p['wer']:>8.4f}  "
            f"{n['cer']:>8.4f} {n['wer']:>8.4f}")


def imprimir_resumen(resumen: dict) -> None:
    print(f"\n{'═'*100}")
    print(f"  RESULTADOS — {resumen['etiqueta']}")
    print(f"{'═'*100}")

    ocr = resumen.get("ocr")
    if ocr is None:
        pass
    elif "error" in ocr:
        print(f"\n  OCR — {ocr['error']}")
    else:
        ancho = max([12] + [len(k) for k in ocr["por_publicacion"]])
        print("\n  OCR (CER / WER)")
        print(f"    {'':<{ancho}} {'PÁGS':>5}  {'TOTAL err/GT car.':<17}  "
              f"{'TOTAL err/GT pal.':<15}  "
              f"{'PROM CER':>8} {'PROM WER':>8}  {'NORM CER':>8} {'NORM WER':>8}")
        print(f"    {'─'*(ancho + 90)}")
        print(_fila("GLOBAL", ocr["global"], ancho))
        for pub, a in ocr["por_publicacion"].items():
            print(_fila(pub, a, ancho))
        print("\n    PROM = media por página (macro) · NORM = errores totales / GT total (micro)")

        print("\n  PEORES PÁGINAS (CER)")
        for p in ocr["peores_paginas"]:
            print(f"    {p['stem'][:42]:<42} {p['publicacion'][:20]:<20} "
                  f"CER {p['cer']:>7.4f}  WER {p['wer']:>7.4f}")

    boxes = resumen.get("layout_boxes")
    if boxes is not None:
        by_cls = resumen.get("layout_boxes_por_clase", {})
        if "error" in boxes:
            print(f"\n  DOCLAYOUT BOUNDING BOXES — {boxes['error']}")
        else:
            origen = resumen.get("layout_boxes_origen")
            print(f"\n  DOCLAYOUT BOUNDING BOXES  ({boxes['n_paginas']} páginas)"
                  + (f"  [cajas de {origen}]" if origen else ""))
            print(f"    avg IoU          {boxes['avg_iou']:.4f}")
            print(f"    mean F1          {boxes['mean_f1']:.4f}")
            print(f"    mean EMR         {boxes['mean_emr']:.4f}")
            print(f"    composite score  {boxes['composite_score']:.4f}"
                  f"  (α={boxes['alpha']} × IoU + {1-boxes['alpha']:.1f} × F1)")
            if by_cls:
                print(f"\n    {'CLASE':<18} {'P':>6}  {'R':>6}  {'F1':>6}"
                      f"  {'TP':>5}  {'FP':>5}  {'FN':>5}")
                print(f"    {'─'*58}")
                for cls, m in sorted(by_cls.items()):
                    print(f"    {cls:<18} {m['precision']:>6.3f}  {m['recall']:>6.3f}"
                          f"  {m['f1']:>6.3f}  {m['tp']:>5}  {m['fp']:>5}  {m['fn']:>5}")
            mm = resumen.get("layout_boxes_multi_match")
            if mm:
                print("\n    DIAGNÓSTICO: predicciones emparejadas con más de una caja GT")
                print("    (informativo; no modifica ninguna métrica de arriba)")
                print(f"    TP totales {mm['tp_total']}  ·  TP que sobran {mm['tp_extra']} "
                      f"({mm['pct_tp_extra']:.2%})  ·  TP sin duplicar {mm['tp_sin_duplicar']}")
                print(f"    Predicciones usadas por >1 GT: {mm['preds_multi']}  ·  "
                      f"páginas afectadas: {mm['paginas_afectadas']}/{mm['n_paginas']}")
                if mm["por_clase"]:
                    print(f"\n    {'CLASE':<18} {'TP':>5}  {'EXTRA':>5}  {'%':>7}  {'PREDS>1GT':>9}")
                    for cls, v in mm["por_clase"].items():
                        print(f"    {cls:<18} {v['tp']:>5}  {v['tp_extra']:>5}  "
                              f"{v['pct_tp_extra']:>7.2%}  {v['preds_multi']:>9}")
                if mm["peores_paginas"]:
                    print("\n    Páginas más afectadas:")
                    for a in mm["peores_paginas"][:5]:
                        print(f"      {str(a['page_id'])[:50]:<50} TP extra {a['tp_extra']}  "
                              f"preds>1GT {a['preds_multi']}")
            dp = resumen.get("layout_boxes_duplicadas")
            if dp:
                print(f"\n    DIAGNÓSTICO: cajas predichas duplicadas "
                      f"(solapan > {dp['dup_iou']} con otra de la misma clase)")
                print("    (informativo; no modifica ninguna métrica de arriba)")
                print(f"    Predicciones {dp['preds_total']}  ·  redundantes (solapan con una "
                      f"anterior) {dp['duplicadas']} ({dp['pct_duplicadas']:.2%})  ·  "
                      f"páginas afectadas {dp['paginas_afectadas']}/{dp['n_paginas']}")
                print(f"    Falsos positivos {dp['fp_total']}  ·  de ellos duplicados de un "
                      f"TP {dp['fp_duplicados_de_tp']} ({dp['pct_fp_duplicados_de_tp']:.2%})")
                if dp["por_clase"]:
                    print(f"\n    {'CLASE':<18} {'PREDS':>6}  {'REDUND':>6}  {'FP':>5}  "
                          f"{'FP dup TP':>9}  {'P':>6}  {'P sin dup':>9}")
                    for cls, v in dp["por_clase"].items():
                        ps = v["precision_sin_duplicados_aprox"]
                        print(f"    {cls:<18} {v['preds']:>6}  {v['duplicadas']:>6}  "
                              f"{v['fp']:>5}  {v['fp_duplicados_de_tp']:>9}  "
                              f"{(v['precision'] or 0):>6.3f}  "
                              f"{(('%.3f' % ps) if ps is not None else '-'):>9}")
                    print("    (P sin dup: precisión aproximada si se eliminaran los FP que "
                          "duplican un TP; no cambia los TP)")
                if dp["peores_paginas"]:
                    print("\n    Páginas con más duplicados:")
                    for a in dp["peores_paginas"][:5]:
                        print(f"      {str(a['page_id'])[:50]:<50} redundantes {a['duplicadas']}  "
                              f"(FP dup de TP {a['fp_duplicados_de_tp']})")
    print(f"\n{'═'*100}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — FLUJOS DE TRABAJO (descubrimiento y evaluación)
# ══════════════════════════════════════════════════════════════════════════════

def _dir_transcripciones(wf: Path) -> Path:
    return wf / "outputs" / "transcripciones"


def _ruta_pred_boxes(wf: Path) -> Path:
    return wf / "outputs" / "predicciones" / "pred_boxes_doclayout.json"


def _ruta_gt_boxes_default(wf: Path) -> Path:
    return wf / "data-regiones" / "labels" / "test" / "gt_boxes.json"


def descubrir_flujos(root: str | None, explicitos: list[str] | None) -> list[Path]:
    flujos: list[Path] = []
    if root:
        r = Path(root)
        if not r.is_dir():
            print(f"ERROR: --root no es un directorio: {r}")
            sys.exit(1)
        for hijo in sorted(r.iterdir()):
            if hijo.is_dir() and (_dir_transcripciones(hijo).is_dir()
                                  or _ruta_pred_boxes(hijo).exists()):
                flujos.append(hijo)
    for w in explicitos or []:
        pw = Path(w)
        if not pw.is_dir():
            print(f"ERROR: --workflow no es un directorio: {pw}")
            sys.exit(1)
        if pw not in flujos:
            flujos.append(pw)
    return flujos


def detectar_formato(dir_trans: Path, forzado: str | None) -> str | None:
    if forzado:
        return forzado
    n_txt = sum(1 for _ in dir_trans.glob("*.txt"))
    n_xml = sum(1 for _ in dir_trans.glob("*.xml"))
    if n_txt and n_xml:
        print(f"  [AVISO] {dir_trans} tiene .txt y .xml: usa --formato para elegir")
        return None
    return "xml" if n_xml else ("txt" if n_txt else None)


def mapa_publicaciones(dir_trans: Path) -> dict[str, str]:
    mapa: dict[str, str] = {}

    def anotar(ruta_img) -> None:
        if not ruta_img:
            return
        ruta = Path(str(ruta_img).replace("\\", "/"))
        pub = _pub_desde_ruta(ruta)
        if pub:
            mapa[unicodedata.normalize("NFC", ruta.stem)] = pub

    inf = dir_trans / "inferencias.jsonl"
    if inf.exists():
        with open(inf, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    reg = json.loads(linea)
                except json.JSONDecodeError:
                    continue
                anotar(reg.get("imagen") or reg.get("image"))

    for ruta in dir_trans.glob("evaluacion_test*.json"):
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for r in datos.get("resultados", []) if isinstance(datos, dict) else []:
            anotar(r.get("imagen"))
    return mapa


def _guardar(ruta: Path, obj) -> None:
    ruta.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def imprimir_comparativa(resultados: dict[str, dict]) -> dict | None:
    ok = {n: r for n, r in resultados.items() if r and "error" not in r}
    if len(ok) < 2:
        return None
    nombres = list(ok)
    letra = {n: chr(65 + i) for i, n in enumerate(nombres)}

    print(f"\n{'═'*100}")
    print("  COMPARATIVA ENTRE FLUJOS (CER / WER)")
    print(f"{'═'*100}\n")
    for n in nombres:
        print(f"    {letra[n]} = {n}")

    print(f"\n    {'':<3}{'PÁGS':>5}  {'PROM CER':>8} {'PROM WER':>8}  "
          f"{'NORM CER':>8} {'NORM WER':>8}")
    print(f"    {'─'*48}")
    for n in nombres:
        g = ok[n]["global"]
        print(f"    {letra[n]:<3}{g['n_paginas']:>5}  "
              f"{g['promedio']['cer']:>8.4f} {g['promedio']['wer']:>8.4f}  "
              f"{g['normalizado']['cer']:>8.4f} {g['normalizado']['wer']:>8.4f}")

    ns = {n: ok[n]["global"]["n_paginas"] for n in nombres}
    if len(set(ns.values())) > 1:
        print(f"\n    [AVISO] Los flujos no se evaluaron sobre el mismo nº de páginas "
              f"({ns}). Revisa las páginas omitidas antes de comparar.")

    pubs = sorted({p for r in ok.values() for p in r["por_publicacion"]})
    ancho = max([12] + [len(p) for p in pubs])
    print(f"\n    Por publicación — resultado NORMALIZADO, celda = CER/WER (nº páginas)")
    print(f"    {'':<{ancho}}" + "".join(f"  {letra[n]:^20}" for n in nombres))
    print(f"    {'─'*(ancho + 22*len(nombres))}")
    for pub in pubs:
        celdas = []
        for n in nombres:
            a = ok[n]["por_publicacion"].get(pub)
            if a and a["normalizado"]["cer"] is not None:
                celdas.append(f"{a['normalizado']['cer']:.3f}/{a['normalizado']['wer']:.3f} "
                              f"({a['n_paginas']})")
            else:
                celdas.append("-")
        print(f"    {pub[:ancho]:<{ancho}}" + "".join(f"  {c:^20}" for c in celdas))
    print(f"\n{'═'*100}\n")

    return {n: {"global": ok[n]["global"], "por_publicacion": ok[n]["por_publicacion"]}
            for n in nombres}


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluación unificada olmOCR: CER/WER (total, promedio, "
                    "normalizado; global y por publicación) + DocLayout. "
                    "Lee las carpetas de los flujos de trabajo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--jsonl", required=True,
                        help="JSONL con el GT del split a evaluar (el mismo para todos los flujos)")
    parser.add_argument("--root", default=None,
                        help="Carpeta que contiene los flujos (se evalúan todos los hijos "
                             "con outputs/transcripciones o cajas de DocLayout)")
    parser.add_argument("--workflow", nargs="+", default=None,
                        help="Una o más carpetas de flujo concretas")
    parser.add_argument("--formato", choices=["txt", "xml"], default=None,
                        help="Forzar el formato de las predicciones (por defecto se detecta)")
    parser.add_argument("--dir-eval", default=None,
                        help="Dónde guardar comparativa_metricas.json "
                             "(default: --root, o la carpeta actual)")
    parser.add_argument("--dir-paginas", default=None,
                        help="Raíz de las imágenes de página (<raíz>/<publicacion>/jpg/<stem>.jpg); "
                             "respaldo para resolver la publicación")
    parser.add_argument("--pub-regex", default=None,
                        help="Regex sobre el stem; el primer grupo es la publicación "
                             "(último recurso)")
    parser.add_argument("--faltantes-vacios", action="store_true",
                        help="Contar páginas sin predicción (o XML inválido) como "
                             "predicción vacía en vez de omitirlas")
    parser.add_argument("--conservar-marcas-columna", action="store_true",
                        help="No eliminar las marcas ---COLUMNA N--- (formato txt)")
    parser.add_argument("--gt-boxes", default=None,
                        help="gt_boxes.json de DocLayout (default: "
                             "<flujo>/data-regiones/labels/test/gt_boxes.json)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="Umbral IoU para DocLayout (default: 0.5)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Peso IoU en el composite score de DocLayout (default: 0.5)")
    parser.add_argument("--dup-iou", type=float, default=0.7,
                        help="IoU a partir del cual una caja predicha se considera "
                             "duplicada de otra de su misma clase (default: 0.7)")
    args = parser.parse_args()

    if not (args.root or args.workflow):
        parser.error("indica --root o --workflow")

    print("Verificando dependencias …")
    verificar_dependencias()
    print("  OK\n")

    flujos = descubrir_flujos(args.root, args.workflow)
    if not flujos:
        print("ERROR: no se encontró ningún flujo (carpeta con outputs/transcripciones "
              "o outputs/predicciones/pred_boxes_doclayout.json)")
        sys.exit(1)
    print("Flujos encontrados:")
    for wf in flujos:
        print(f"  · {wf.name}")
    print()

    dir_paginas = Path(args.dir_paginas) if args.dir_paginas else None
    if dir_paginas is not None and not dir_paginas.is_dir():
        print(f"ERROR: --dir-paginas no es un directorio: {dir_paginas}")
        sys.exit(1)
    pub_regex = re.compile(args.pub_regex) if args.pub_regex else None

    # ── GT (una sola vez, común a todos los flujos) ───────────────────────────
    jsonl = Path(args.jsonl)
    if not jsonl.exists():
        print(f"ERROR: JSONL no encontrado: {jsonl}")
        sys.exit(1)
    pub_map: dict[str, str] = {}
    for wf in flujos:
        if _dir_transcripciones(wf).is_dir():
            pub_map.update(mapa_publicaciones(_dir_transcripciones(wf)))
    idx_imgs = _indexar_imagenes(dir_paginas)
    print(f"Cargando GT de {jsonl} …")
    gt = cargar_gt(jsonl, idx_imgs, pub_regex, pub_map)
    print(f"  {len(gt)} páginas GT\n")

    # ── DocLayout: cajas (carpetas con pred_boxes_doclayout.json) ─────────────
    boxes_por_flujo: dict[str, dict] = {}
    for wf in flujos:
        pb = _ruta_pred_boxes(wf)
        if not pb.exists():
            continue
        gtb = Path(args.gt_boxes) if args.gt_boxes else _ruta_gt_boxes_default(wf)
        print(f"[{wf.name}] Evaluación layout bounding boxes (DocLayout) …")
        print(f"  GT   : {gtb}")
        print(f"  Pred : {pb}")
        mb = evaluate_layout_boxes(gtb, pb, args.iou_threshold, args.alpha, args.dup_iou)
        dir_m = wf / "outputs" / "metricas"
        dir_m.mkdir(parents=True, exist_ok=True)
        _guardar(dir_m / "metrics_layout_boxes.json", mb)
        boxes_por_flujo[wf.name] = mb
        print(f"  Guardado: {dir_m / 'metrics_layout_boxes.json'}\n")
    ref_boxes = next(((n, m) for n, m in boxes_por_flujo.items()
                      if "global_metrics" in m), None)

    # ── OCR por flujo ─────────────────────────────────────────────────────────
    resultados_ocr: dict[str, dict] = {}
    resumenes: list[dict] = []

    for wf in flujos:
        nombre = wf.name
        dir_m = wf / "outputs" / "metricas"
        dir_m.mkdir(parents=True, exist_ok=True)
        dir_trans = _dir_transcripciones(wf)

        metrics_ocr = None
        if dir_trans.is_dir():
            formato = detectar_formato(dir_trans, args.formato)
            if formato is None:
                print(f"[{nombre}] [AVISO] sin predicciones .txt/.xml en {dir_trans}\n")
            else:
                print(f"[{nombre}] Evaluación OCR (CER / WER), formato {formato} …")
                preds, alias = cargar_predicciones(dir_trans, formato)
                print(f"  {len(preds)} predicciones en {dir_trans}")
                metrics_ocr = evaluate_ocr(
                    gt, preds, alias,
                    faltantes_vacios=args.faltantes_vacios,
                    quitar_marcas=not args.conservar_marcas_columna,
                )
                _guardar(dir_m / "metrics_ocr.json", metrics_ocr)
                print(f"  Guardado: {dir_m / 'metrics_ocr.json'}\n")
                pubs = metrics_ocr.get("por_publicacion", {})
                if list(pubs) == ["desconocida"]:
                    print("  [AVISO] No se pudo resolver la publicación de ninguna página; "
                          "usa --dir-paginas o --pub-regex.\n")
                resultados_ocr[nombre] = metrics_ocr

        # Cajas: propias (flujo DocLayout) o copiadas de él (flujos que lo usan)
        metrics_boxes, origen = boxes_por_flujo.get(nombre), None
        if metrics_boxes is None and "doclayout" in nombre.lower():
            if ref_boxes is not None:
                origen = ref_boxes[0]
                metrics_boxes = {"global_metrics": ref_boxes[1]["global_metrics"]}
            elif boxes_por_flujo:
                err = next(iter(boxes_por_flujo.values())).get("error", "sin datos")
                print(f"[{nombre}] [AVISO] usa DocLayout pero la evaluación de cajas "
                      f"falló: {err}\n  (indica la ruta correcta con --gt-boxes)\n")
            else:
                print(f"[{nombre}] [AVISO] usa DocLayout pero la carpeta de DocLayout "
                      f"(la que tiene outputs/predicciones/pred_boxes_doclayout.json) "
                      f"no está entre los flujos evaluados: añádela a --workflow o "
                      f"usa --root\n")

        if metrics_ocr is None and metrics_boxes is None:
            continue
        resumen = generar_resumen(nombre, metrics_ocr, metrics_boxes, origen)
        _guardar(dir_m / "metrics_summary.json", resumen)
        resumenes.append(resumen)

    for resumen in resumenes:
        imprimir_resumen(resumen)

    comparativa = imprimir_comparativa(resultados_ocr)
    if comparativa is not None:
        base = Path(args.dir_eval) if args.dir_eval else (
            Path(args.root) if args.root else Path.cwd())
        base.mkdir(parents=True, exist_ok=True)
        _guardar(base / "comparativa_metricas.json", comparativa)
        print(f"Comparativa guardada en: {base / 'comparativa_metricas.json'}")

    print("Resultados de cada flujo en <flujo>/outputs/metricas/")


if __name__ == "__main__":
    main()