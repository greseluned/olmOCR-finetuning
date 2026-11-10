#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_inference.py

Inferencia con el olmOCR fine-tuneado sobre regiones, usando docLayout-YOLO para
segmentar la página. Fase 1: docLayout detecta las regiones de todas las
imágenes (con la misma lógica que 02_build_cache_layout.py) y se descarga de
memoria. Fase 2: olmOCR transcribe cada recorte con los prompts de
03_build_dataset.py, así que regiones y prompts coinciden con los del
entrenamiento. Genera un PAGE-XML por página con coordenadas, orden de lectura y
transcripciones, usa checkpoint para reanudar y, con --evaluar, calcula el CER
sobre el test set. Si una región da CUDA out of memory, reintenta con anchos
menores (max_ancho, 1200, 900, 600).

Uso:
    python scripts/05_inference.py --evaluar --jsonl
    python scripts/05_inference.py --imagen corpus/fémina/jpg/femina_1922_p03.jpg
    python scripts/05_inference.py --dir corpus/fémina/jpg/ --jsonl
    python scripts/05_inference.py --evaluar --sin-layout
    python scripts/05_inference.py --evaluar --max-ancho 1200

Opciones:
    --imagen / --dir / --evaluar   origen de las imágenes (una, carpeta o test set)
    --modelo-ocr RUTA              modelo olmOCR fusionado (config.INFERENCIA_MODELO_DIR)
    --modelo-layout RUTA           best.pt de docLayout
    --conf-layout X                umbral de confianza de docLayout (el del 02)
    --max-tokens N                 tokens máximos de generación
    --max-ancho N                  ancho máximo del recorte antes de escalar
    --sin-layout                   regiones desde los XML paralelos, sin docLayout
    --jsonl                        guarda también un JSONL de auditoría
    --mostrar                      imprime las transcripciones por región
    --salida RUTA                  directorio de salida (config.OUTPUTS_DIR)

En Windows conviene fijar antes:
    set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

import argparse
import gc
import importlib.util as _ilu
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# Reducir fragmentación de VRAM antes de importar torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def _cargar_script(nombre_fichero: str, alias: str):
    spec = _ilu.spec_from_file_location(alias, Path(__file__).parent / nombre_fichero)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_parse   = _cargar_script("01_parse_pagexml.py",       "parse_pagexml")
_layout  = _cargar_script("02_build_cache_layout.py",  "build_cache_layout")
_dataset = _cargar_script("03_build_dataset.py",       "build_dataset")

xml_a_regiones         = _parse.xml_a_regiones
clasificar_tipo_pagina = _parse.clasificar_tipo_pagina

# ── Fuente única de verdad con el entrenamiento ───────────────────────────────
CLASES_GRAPHIC          = _layout.CLASES_GRAPHIC
DOCLAYOUT_MODEL_DEFAULT = _layout.DOCLAYOUT_MODEL
CONF_LAYOUT_DEFAULT     = _layout.CONF_DEFAULT
PROMPT_TEXT             = _dataset.PROMPT_TEXT
PROMPT_GRAPHIC          = _dataset.PROMPT_GRAPHIC

NS_PAGE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

# Anchos de reintento en caso de OOM (de mayor a menor)
_FALLBACK_ANCHOS = [1200, 900, 600]


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDAD — anchos a probar ante OOM
# ══════════════════════════════════════════════════════════════════════════════

def _anchos_a_probar(max_ancho: int | None) -> list[int | None]:
    if max_ancho is None:
        return [None] + _FALLBACK_ANCHOS
    return [max_ancho] + [a for a in _FALLBACK_ANCHOS if a < max_ancho]


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CLASIFICADOR DE LAYOUT (docLayout-YOLO)
# Fase 1: detecta regiones en TODAS las imágenes y se descarga antes de olmOCR
# ══════════════════════════════════════════════════════════════════════════════

class ClasificadorLayout:

    def __init__(self, ruta_modelo: Path, conf: float = CONF_LAYOUT_DEFAULT):
        try:
            from doclayout_yolo import YOLOv10
        except ImportError:
            print("ERROR: doclayout-yolo no instalado.\n"
                  "  pip install doclayout-yolo")
            sys.exit(1)

        if not ruta_modelo.exists():
            print(f"ERROR: Modelo docLayout no encontrado: {ruta_modelo}")
            sys.exit(1)

        print(f"  Cargando docLayout desde «{ruta_modelo}» …")
        self._modelo = YOLOv10(str(ruta_modelo))
        self._conf   = conf
        print(f"  docLayout listo (conf={conf}, clases gráficas: "
              f"{sorted(CLASES_GRAPHIC)}).")

    def detectar_lote(self, imagenes: list[Path]) -> dict[str, list[dict]]:
        total     = len(imagenes)
        resultado: dict[str, list[dict]] = {}

        print(f"\nFase 1 — detectando regiones en {total} páginas con docLayout …")
        for i, ruta in enumerate(imagenes, 1):
            try:
                regiones = self._detectar_una(ruta)
            except Exception as e:
                print(f"  [{i:>3}/{total}] {ruta.name}  ERROR: {e} → región completa")
                regiones = self._region_completa(ruta)
            else:
                print(f"  [{i:>3}/{total}] {ruta.name}  → {len(regiones)} regiones")
            resultado[ruta.stem] = regiones

        print("Fase 1 completada.\n")
        return resultado

    def descargar(self) -> None:
        import torch
        del self._modelo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  docLayout descargado de memoria.")

    # ── Métodos internos ──────────────────────────────────────────────────────

    def _detectar_una(self, ruta_jpg: Path) -> list[dict]:
        regiones = _layout.extraer_regiones_pagina(ruta_jpg, self._modelo, self._conf)
        return regiones if regiones else self._region_completa(ruta_jpg)

    @staticmethod
    def _region_completa(ruta_jpg: Path) -> list[dict]:
        from PIL import Image
        with Image.open(ruta_jpg) as im:
            w, h = im.size
        return [{"id": "r_000", "tipo": "text", "orden": 0, "bbox": [0, 0, w, h]}]


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — FALLBACK SIN docLayout (desde XML)
# ══════════════════════════════════════════════════════════════════════════════

def _es_portada(stem: str) -> bool:
    s = stem.lower()
    return s.endswith("_p001") or s.endswith("_001")


def detectar_regiones_sin_layout(imagenes: list[Path]) -> dict[str, list[dict]]:
    total     = len(imagenes)
    resultado: dict[str, list[dict]] = {}

    print(f"\nFase 1 — obteniendo regiones sin docLayout (fallback XML) …")
    for i, ruta in enumerate(imagenes, 1):
        stem = ruta.stem
        regiones = None

        posibles_xml = [
            ruta.parent.parent / "page" / (stem + ".xml"),
            ruta.parent / "page" / (stem + ".xml"),
            ruta.with_suffix(".xml"),
        ]
        for ruta_xml in posibles_xml:
            if ruta_xml.exists():
                regiones_gt = xml_a_regiones(ruta_xml)
                if regiones_gt:
                    regiones = [
                        {"id": r["id"], "tipo": r["tipo"],
                         "orden": r["orden"], "bbox": r["bbox"]}
                        for r in regiones_gt
                    ]
                break

        if not regiones:
            from PIL import Image
            with Image.open(ruta) as im:
                w, h = im.size
            regiones = [{"id": "r_000", "tipo": "text", "orden": 0,
                         "bbox": [0, 0, w, h]}]

        print(f"  [{i:>3}/{total}] {ruta.name}  → {len(regiones)} regiones")
        resultado[stem] = regiones

    print("Fase 1 completada.\n")
    return resultado


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — RECORTE DE REGIONES
# ══════════════════════════════════════════════════════════════════════════════

def recortar_region(
    jpg: Path,
    bbox: list[int],
    dir_tmp: Path,
    margen: int = 4,
    max_ancho: int | None = None,
) -> Path | None:
    from PIL import Image

    dir_tmp.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = bbox
    sufijo_ancho = f"_w{max_ancho}" if max_ancho else ""
    nombre = f"{jpg.stem}_{x0}_{y0}_{x1}_{y1}{sufijo_ancho}.jpg"
    ruta   = dir_tmp / nombre

    if ruta.exists():
        return ruta

    try:
        with Image.open(jpg) as im:
            w, h = im.size
            box  = (
                max(0, x0 - margen),
                max(0, y0 - margen),
                min(w, x1 + margen),
                min(h, y1 + margen),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                return None
            crop = im.convert("RGB").crop(box)
            if max_ancho and crop.width > max_ancho:
                ratio = max_ancho / crop.width
                crop  = crop.resize(
                    (max_ancho, int(crop.height * ratio)),
                    Image.LANCZOS,
                )
            crop.save(ruta, quality=95)
        return ruta
    except Exception as e:
        print(f"    [ERROR recorte] {jpg.name} {bbox}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — MODELO olmOCR
# ══════════════════════════════════════════════════════════════════════════════

class ModeloOCR:

    def __init__(self, ruta_modelo: Path):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        print(f"  Cargando olmOCR desde «{ruta_modelo}» …")
        if not ruta_modelo.exists():
            raise FileNotFoundError(
                f"{ruta_modelo} no existe.\n"
                "Ejecuta primero 04_train_lora.py para generar el modelo fusionado."
            )

        self._processor = AutoProcessor.from_pretrained(
            str(ruta_modelo), trust_remote_code=True
        )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(ruta_modelo),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()
        self._torch = torch
        print("  olmOCR listo.")

    def transcribir(
        self,
        ruta_imagen: Path,
        prompt: str,
        max_new_tokens: int = config.INFERENCIA_MAX_NEW_TOKENS,
    ) -> tuple[str, float]:
        from PIL import Image

        try:
            img_pil = Image.open(ruta_imagen).convert("RGB")
        except Exception as e:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}") from e

        texto_instruccion = prompt.replace("<image>\n", "")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_pil},
                    {"type": "text",  "text":  texto_instruccion},
                ],
            }
        ]

        try:
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            text   = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = self._processor(
                text=[text], images=[img_pil], return_tensors="pt"
            )

        if self._torch.cuda.is_available():
            inputs = {
                k: v.cuda() if hasattr(v, "cuda") else v
                for k, v in inputs.items()
            }

        t0 = time.time()
        with self._torch.no_grad():
            if self._torch.cuda.is_available():
                self._torch.cuda.synchronize()
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=config.INFERENCIA_DO_SAMPLE,
            )
        duracion = time.time() - t0

        input_len     = inputs["input_ids"].shape[1]
        gen_tokens    = output_ids[0][input_len:]
        transcripcion = self._processor.decode(
            gen_tokens, skip_special_tokens=True
        ).strip()

        del output_ids, gen_tokens, inputs, img_pil
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

        return transcripcion, duracion


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — CONSTRUCCIÓN DEL PAGE-XML DE SALIDA
# ══════════════════════════════════════════════════════════════════════════════

def _bbox_to_points(bbox: list[int]) -> str:
    x0, y0, x1, y1 = bbox
    return f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"


def construir_page_xml(
    ruta_jpg: Path,
    regiones: list[dict],
    transcripciones: list[str],
) -> str:
    from PIL import Image

    with Image.open(ruta_jpg) as im:
        img_w, img_h = im.size

    ns = NS_PAGE
    ET.register_namespace("", ns)

    root = ET.Element(f"{{{ns}}}PcGts")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")

    metadata = ET.SubElement(root, f"{{{ns}}}Metadata")
    creator  = ET.SubElement(metadata, f"{{{ns}}}Creator")
    creator.text = "olmOCR-finetune / 05_inference.py"

    page = ET.SubElement(root, f"{{{ns}}}Page")
    page.set("imageFilename", ruta_jpg.name)
    page.set("imageWidth",    str(img_w))
    page.set("imageHeight",   str(img_h))

    reading_order = ET.SubElement(page, f"{{{ns}}}ReadingOrder")
    ordered_group = ET.SubElement(reading_order, f"{{{ns}}}OrderedGroup")
    ordered_group.set("id",      "ro_main")
    ordered_group.set("caption", "")

    regiones_ord = sorted(regiones, key=lambda r: r["orden"])

    for reg in regiones_ord:
        ref = ET.SubElement(ordered_group, f"{{{ns}}}RegionRefIndexed")
        ref.set("index",     str(reg["orden"]))
        ref.set("regionRef", reg["id"])

    for reg, texto in zip(regiones_ord, transcripciones):
        tag = (
            f"{{{ns}}}TextRegion"
            if reg["tipo"] == "text"
            else f"{{{ns}}}ImageRegion"
        )
        elem = ET.SubElement(page, tag)
        elem.set("id",     reg["id"])
        elem.set("custom", f"readingOrder {{index:{reg['orden']}}}")

        coords = ET.SubElement(elem, f"{{{ns}}}Coords")
        coords.set("points", _bbox_to_points(reg["bbox"]))

        text_equiv = ET.SubElement(elem, f"{{{ns}}}TextEquiv")
        unicode_el = ET.SubElement(text_equiv, f"{{{ns}}}Unicode")
        unicode_el.text = texto

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def cargar_checkpoint(ruta: Path) -> set[str]:
    if not ruta.exists():
        return set()
    try:
        data = json.loads(ruta.read_text(encoding="utf-8"))
        procesadas = set(data.get("procesadas", []))
        print(f"  Checkpoint cargado: {len(procesadas)} páginas ya procesadas.")
        return procesadas
    except Exception:
        return set()


def guardar_checkpoint(ruta: Path, procesadas: set[str]) -> None:
    ruta.write_text(
        json.dumps({"procesadas": sorted(procesadas)}, ensure_ascii=False),
        encoding="utf-8",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — MÉTRICAS (CER)
# ══════════════════════════════════════════════════════════════════════════════

def _cer_aproximado(referencia: str, prediccion: str) -> float:
    ref  = referencia.strip()[:5000]
    pred = prediccion.strip()[:5000]
    if not ref:
        return 0.0 if not pred else 1.0
    m, n = len(ref), len(pred)
    dp   = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if ref[i-1] == pred[j-1] else \
                    1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n] / max(m, n)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — PROCESADO DE UNA PÁGINA
# ══════════════════════════════════════════════════════════════════════════════

def procesar_imagen(
    ruta_jpg: Path,
    modelo_ocr: ModeloOCR,
    regiones: list[dict],
    dir_salida: Path,
    dir_tmp: Path,
    max_new_tokens: int,
    max_ancho: int | None,
    guardar_jsonl: bool,
    mostrar: bool,
    gt_regiones: list[dict] | None = None,
) -> dict:
    import torch

    if not regiones:
        return {"imagen": str(ruta_jpg), "error": "sin_regiones"}

    regiones_ord       = sorted(regiones, key=lambda r: r["orden"])
    transcripciones: list[str] = []
    duracion_total     = 0.0

    for reg in regiones_ord:
        prompt = PROMPT_TEXT if reg["tipo"] == "text" else PROMPT_GRAPHIC

        texto: str | None = None
        dur: float        = 0.0
        anchos            = _anchos_a_probar(max_ancho)

        for k, ancho_intento in enumerate(anchos):

            ruta_crop = recortar_region(
                ruta_jpg, reg["bbox"], dir_tmp, max_ancho=ancho_intento
            )
            if ruta_crop is None:
                texto = "[ERROR: recorte fallido]"
                break

            try:
                texto, dur = modelo_ocr.transcribir(ruta_crop, prompt, max_new_tokens)
                break  # éxito

            except torch.cuda.OutOfMemoryError:
                siguiente = anchos[k + 1] if k + 1 < len(anchos) else None
                if siguiente is not None:
                    print(
                        f"    [OOM] {ruta_jpg.name} región {reg['id']} "
                        f"(ancho={ancho_intento}) → "
                        f"liberando VRAM y reintentando con ancho={siguiente} …"
                    )
                else:
                    print(
                        f"    [OOM] {ruta_jpg.name} región {reg['id']} "
                        f"(ancho={ancho_intento}) → sin más fallbacks."
                    )
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                texto = None

            except Exception as e:
                print(f"    [ERROR olmOCR] {ruta_jpg.name} región {reg['id']}: {e}")
                texto = "[ERROR]"
                dur   = 0.0
                break

        if texto is None:
            print(
                f"    [ERROR olmOCR] {ruta_jpg.name} región {reg['id']}: "
                f"OOM en todos los intentos ({anchos})"
            )
            texto = "[ERROR: OOM]"

        transcripciones.append(texto)
        duracion_total += dur

        if mostrar:
            print(f"    [{reg['orden']:>2}] {reg['tipo']:<8}  "
                  f"{reg['bbox']}  →  {texto[:80]!r}")

    # ── PAGE-XML ──────────────────────────────────────────────────────────────
    xml_str  = construir_page_xml(ruta_jpg, regiones_ord, transcripciones)
    ruta_xml = dir_salida / (ruta_jpg.stem + ".xml")
    dir_salida.mkdir(parents=True, exist_ok=True)
    ruta_xml.write_text(xml_str, encoding="utf-8")

    # ── CER ───────────────────────────────────────────────────────────────────
    cer = None
    if gt_regiones:
        texto_gt_plano = "\n".join(
            r["texto"] for r in sorted(gt_regiones, key=lambda x: x["orden"])
            if r["tipo"] == "text" and r.get("texto", "").strip()
        )
        texto_pred_plano = "\n".join(
            t for reg, t in zip(regiones_ord, transcripciones)
            if reg["tipo"] == "text"
        )
        if texto_gt_plano:
            cer = _cer_aproximado(texto_gt_plano, texto_pred_plano)

    resultado = {
        "imagen":     str(ruta_jpg),
        "n_regiones": len(regiones_ord),
        "duracion_s": round(duracion_total, 2),
        "ruta_xml":   str(ruta_xml),
    }
    if cer is not None:
        resultado["cer"] = cer

    # ── JSONL de auditoría ────────────────────────────────────────────────────
    if guardar_jsonl:
        registro_jsonl = {
            "imagen":     str(ruta_jpg.resolve()),
            "n_regiones": len(regiones_ord),
            "regiones": [
                {"id": r["id"], "tipo": r["tipo"], "orden": r["orden"],
                 "bbox": r["bbox"], "texto": t}
                for r, t in zip(regiones_ord, transcripciones)
            ],
            "duracion_s": round(duracion_total, 2),
        }
        if cer is not None:
            registro_jsonl["cer"] = cer
        with open(dir_salida / "inferencias.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(registro_jsonl, ensure_ascii=False) + "\n")

    return resultado


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9 — BUCLE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def procesar_lote(
    imagenes: list[Path],
    modelo_ocr: ModeloOCR,
    todas_regiones: dict[str, list[dict]],
    dir_salida: Path,
    max_new_tokens: int,
    max_ancho: int | None,
    guardar_jsonl: bool,
    mostrar: bool,
    gt_index: dict[str, list[dict]] | None = None,
) -> list[dict]:
    dir_tmp         = dir_salida / "_crops_tmp"
    ruta_checkpoint = dir_salida / "checkpoint.json"
    procesadas      = cargar_checkpoint(ruta_checkpoint)

    total      = len(imagenes)
    resultados = []

    print(f"Fase 2 — transcribiendo {total} páginas con olmOCR …\n")

    for i, ruta_jpg in enumerate(imagenes, 1):
        stem = ruta_jpg.stem

        if stem in procesadas:
            print(f"  [{i:>3}/{total}] {ruta_jpg.name} — ya procesada, saltando")
            continue

        regiones = todas_regiones.get(stem)
        if not regiones:
            print(f"  [{i:>3}/{total}] {ruta_jpg.name} — sin regiones, saltando")
            continue

        gt_regs = (gt_index or {}).get(stem)
        print(f"  [{i:>3}/{total}] {ruta_jpg.name} "
              f"({len(regiones)} regiones) …", end=" ", flush=True)

        try:
            res = procesar_imagen(
                ruta_jpg, modelo_ocr, regiones,
                dir_salida, dir_tmp,
                max_new_tokens, max_ancho,
                guardar_jsonl, mostrar,
                gt_regs,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        cer_str = f"  CER={res['cer']:.3f}" if "cer" in res else ""
        print(f"{res.get('duracion_s', 0):.1f}s{cer_str}")
        resultados.append(res)

        procesadas.add(stem)
        guardar_checkpoint(ruta_checkpoint, procesadas)

    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10 — EVALUACIÓN SOBRE EL TEST SET
# ══════════════════════════════════════════════════════════════════════════════

def evaluar_test_set(
    modelo_ocr: ModeloOCR,
    todas_regiones: dict[str, list[dict]],
    dir_salida: Path,
    mostrar: bool,
    max_new_tokens: int,
    max_ancho: int | None,
    guardar_jsonl: bool,
) -> None:
    if not config.JSONL_TEST.exists():
        print(f"ERROR: {config.JSONL_TEST} no existe. "
              f"Ejecuta primero 03_build_dataset.py.")
        sys.exit(1)

    gt_index: dict[str, list[dict]] = {}
    imagen_paths: dict[str, Path]   = {}

    with open(config.JSONL_TEST, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            r    = json.loads(linea)
            stem = r["_meta"]["fuente"]

            if stem not in imagen_paths:
                for pub in config.PUBLICACIONES:
                    candidata = config.CORPUS_DIR / pub / "jpg" / (stem + ".jpg")
                    if candidata.exists():
                        imagen_paths[stem] = candidata
                        break

            gt_index.setdefault(stem, []).append({
                "id":    r["_meta"]["region_id"],
                "tipo":  r["_meta"]["region_tipo"],
                "orden": r["_meta"]["region_orden"],
                "bbox":  r["_meta"]["region_bbox"],
                "texto": r["conversations"][1]["content"],
            })

    imagenes = [imagen_paths[s] for s in gt_index if s in imagen_paths]
    print(f"  {len(imagenes)} páginas en el test set "
          f"({sum(len(v) for v in gt_index.values())} regiones GT)")

    resultados = procesar_lote(
        imagenes       = imagenes,
        modelo_ocr     = modelo_ocr,
        todas_regiones = todas_regiones,
        dir_salida     = dir_salida,
        max_new_tokens = max_new_tokens,
        max_ancho      = max_ancho,
        guardar_jsonl  = guardar_jsonl,
        mostrar        = mostrar,
        gt_index       = gt_index,
    )

    cers = [r["cer"] for r in resultados if "cer" in r]
    if cers:
        cer_medio = sum(cers) / len(cers)
        print(f"\n{'─'*55}")
        print(f"CER medio sobre {len(cers)} páginas: {cer_medio:.4f}")

        ruta_resumen = dir_salida / "evaluacion_test.json"
        ruta_resumen.write_text(
            json.dumps(
                {"cer_medio": cer_medio, "n_paginas": len(cers),
                 "resultados": resultados},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Resumen: {ruta_resumen}")

    print(f"\nPAGE-XML guardados en: {dir_salida}")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inferencia olmOCR — pipeline segmentado por regiones.\n"
            "Fase 1: docLayout detecta regiones (misma lógica que 02) y se descarga.\n"
            "Fase 2: olmOCR transcribe recorte a recorte con los prompts de 03.\n"
            "Salida: PAGE-XML con coordenadas, reading order y transcripciones."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/05_inference.py --evaluar --jsonl
  python scripts/05_inference.py --imagen corpus/fémina/jpg/femina_1922_p03.jpg
  python scripts/05_inference.py --dir corpus/fémina/jpg/ --jsonl
  python scripts/05_inference.py --evaluar --sin-layout
  python scripts/05_inference.py --evaluar --max-ancho 1200
        """,
    )
    parser.add_argument("--imagen",     default=None)
    parser.add_argument("--dir",        default=None)
    parser.add_argument("--evaluar",    action="store_true")
    parser.add_argument("--modelo-ocr", default=None)
    parser.add_argument(
        "--modelo-layout", default=None,
        help=f"Ruta al best.pt de docLayout (default: {DOCLAYOUT_MODEL_DEFAULT})",
    )
    parser.add_argument(
        "--conf-layout", type=float, default=CONF_LAYOUT_DEFAULT,
        help=f"Umbral de confianza docLayout (default: {CONF_LAYOUT_DEFAULT}, "
             f"el mismo que usa 02)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=config.INFERENCIA_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--max-ancho", type=int, default=None,
        help=(
            "Ancho máximo del recorte antes de escalar. En caso de OOM se "
            f"reintenta con anchos menores (fallbacks: {_FALLBACK_ANCHOS})."
        ),
    )
    parser.add_argument("--jsonl",   action="store_true")
    parser.add_argument("--mostrar", action="store_true")
    parser.add_argument(
        "--sin-layout", action="store_true",
        help="No usar docLayout; obtener regiones desde XML paralelos (fallback)",
    )
    parser.add_argument(
        "--salida", default=None,
        help=f"Directorio de salida (default: {config.OUTPUTS_DIR})",
    )
    args = parser.parse_args()

    if not any([args.imagen, args.dir, args.evaluar]):
        parser.print_help()
        sys.exit(1)

    ruta_modelo_ocr = (
        Path(args.modelo_ocr) if args.modelo_ocr else config.INFERENCIA_MODELO_DIR
    )
    ruta_modelo_layout = (
        Path(args.modelo_layout) if args.modelo_layout else DOCLAYOUT_MODEL_DEFAULT
    )
    dir_salida = Path(args.salida) if args.salida else config.OUTPUTS_DIR

    # ── Recopilar imágenes ────────────────────────────────────────────────────
    imagenes: list[Path] = []

    if args.imagen:
        p = Path(args.imagen)
        if not p.exists():
            print(f"ERROR: {p} no existe.")
            sys.exit(1)
        imagenes = [p]

    elif args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            print(f"ERROR: {d} no es un directorio.")
            sys.exit(1)
        imagenes = sorted(
            {p.resolve(): p for p in d.glob("*")
             if p.suffix.lower() in (".jpg", ".jpeg")}.values()
        )
        if not imagenes:
            print(f"ERROR: No se encontraron JPGs en {d}")
            sys.exit(1)
        print(f"  {len(imagenes)} imágenes encontradas en {d}")

    elif args.evaluar:
        if not config.JSONL_TEST.exists():
            print(f"ERROR: {config.JSONL_TEST} no existe. "
                  f"Ejecuta primero 03_build_dataset.py.")
            sys.exit(1)
        with open(config.JSONL_TEST, encoding="utf-8") as f:
            registros = [json.loads(l) for l in f if l.strip()]
        stems_vistos = set()
        for r in registros:
            stem = r["_meta"]["fuente"]
            if stem in stems_vistos:
                continue
            stems_vistos.add(stem)
            for pub in config.PUBLICACIONES:
                candidata = config.CORPUS_DIR / pub / "jpg" / (stem + ".jpg")
                if candidata.exists():
                    imagenes.append(candidata)
                    break
        print(f"  {len(imagenes)} páginas únicas en el test set")

    # ── Fase 1: detectar regiones ─────────────────────────────────────────────
    if args.sin_layout:
        todas_regiones = detectar_regiones_sin_layout(imagenes)
    else:
        clasificador   = ClasificadorLayout(ruta_modelo_layout, args.conf_layout)
        todas_regiones = clasificador.detectar_lote(imagenes)
        clasificador.descargar()
        del clasificador

    # ── Fase 2: cargar olmOCR y transcribir ───────────────────────────────────
    print("Cargando olmOCR …")
    modelo_ocr = ModeloOCR(ruta_modelo_ocr)
    print()

    if args.evaluar:
        evaluar_test_set(
            modelo_ocr     = modelo_ocr,
            todas_regiones = todas_regiones,
            dir_salida     = dir_salida,
            mostrar        = args.mostrar,
            max_new_tokens = args.max_tokens,
            max_ancho      = args.max_ancho,
            guardar_jsonl  = args.jsonl,
        )
    else:
        resultados = procesar_lote(
            imagenes       = imagenes,
            modelo_ocr     = modelo_ocr,
            todas_regiones = todas_regiones,
            dir_salida     = dir_salida,
            max_new_tokens = args.max_tokens,
            max_ancho      = args.max_ancho,
            guardar_jsonl  = args.jsonl,
            mostrar        = args.mostrar,
        )
        print(f"\n{len(resultados)} páginas procesadas.")
        print(f"PAGE-XML guardados en: {dir_salida}")


if __name__ == "__main__":
    main()