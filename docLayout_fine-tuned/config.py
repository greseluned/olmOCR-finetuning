#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py
=========

Configuración centralizada del pipeline DocLayout-YOLO. 
Es independiente del pipeline de olmOCR.

Los scripts del pipeline importan todos sus parámetros desde este
archivo, por lo que es el único que debe editarse para adaptarlo a
otra máquina o corpus.

CONTENIDO:
    1. Rutas principales (a editar según la máquina).
    2. Rutas derivadas (no requieren modificación).
    3. Publicaciones del corpus y modo de split de cada una.
    4. Parámetros del split train/val/test.
    5. Modelo base y nombre del modelo de salida.
    6. Parámetros de entrenamiento y de predicción.

IMPORTANTE:
    Los parámetros del split y la lista de publicaciones deben coincidir
    exactamente con los del pipeline de olmOCR, para que ambos modelos
    trabajen con los mismos conjuntos de páginas.
"""

from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — RUTAS PRINCIPALES (editar según la máquina)
# ══════════════════════════════════════════════════════════════════════════════

# Raíz del repositorio from-paper-to-pixel
DOCLAYOUT_ROOT = Path(__file__).resolve().parent

# Corpus de imágenes + PAGE-XML (puede estar en cualquier ubicación).
# Estructura esperada:
#   CORPUS_DIR/<publicacion>/jpg/<stem>.jpg
#   CORPUS_DIR/<publicacion>/page/<stem>.xml
CORPUS_DIR = DOCLAYOUT_ROOT.parent / "corpus"

# Carpeta de salida de la evaluación (pred_boxes.json, etc.)
EVAL_DIR = DOCLAYOUT_ROOT / "outputs" / "predicciones"


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — RUTAS DERIVADAS (no es necesario modificarlas)
# ══════════════════════════════════════════════════════════════════════════════

DOCLAYOUT_DIR = DOCLAYOUT_ROOT
MODELS_DIR    = DOCLAYOUT_DIR / "models"

CORPUS_NAME    = "data-regiones"
CORPUS_OUT_DIR = DOCLAYOUT_DIR / CORPUS_NAME             # generado por 02_prepare_dataset.py
DATA_DIR       = DOCLAYOUT_ROOT / "data" / CORPUS_NAME   # generado por 01_pagexml_to_yolo.py


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — PUBLICACIONES DEL CORPUS
# ══════════════════════════════════════════════════════════════════════════════

# Mismo criterio que en olmOCR:
#   'estratificar' → split 80-10-10
#   'solo_train'   → todas las páginas a train
# La lista debe mantenerse sincronizada con la del pipeline de olmOCR.
PUBLICACIONES = {
    "accion-libertaria":     "estratificar",
    "boletín-de-la-cámara":  "estratificar",
    "excelsior":             "estratificar",
    "fémina":                "estratificar",
    "filipinas":             "estratificar",
    "filipinas-ante-europa": "estratificar",
    "heraldo-de-la-mujer":   "estratificar",
    "hispanidad":            "estratificar",
    "la-malasia":            "estratificar",
    "la-vanguardia":         "estratificar",
    "semana":                "estratificar",
    "sorpresas-chicago":     "estratificar",
    "other":                 "solo_train",
}


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — SPLIT
# ══════════════════════════════════════════════════════════════════════════════

# Estos valores deben coincidir exactamente con los del pipeline de olmOCR
# para que train/val/test sean los mismos conjuntos de páginas.

SPLIT_PATH = DOCLAYOUT_ROOT.parent / "split.json"

SPLIT_TRAIN       = 0.80
SPLIT_VALIDATION  = 0.10
SPLIT_TEST        = 0.10
SPLIT_SEED        = 42    # misma semilla que olmOCR
SPLIT_MIN_PAGINAS = 10


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — MODELO
# ══════════════════════════════════════════════════════════════════════════════

BASE_MODEL  = "doclayout_yolo_docstructbench_imgsz1024.pt"
OUTPUT_NAME = "layout-model"


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — PARÁMETROS DE ENTRENAMIENTO Y PREDICCIÓN
# ══════════════════════════════════════════════════════════════════════════════

# Entrenamiento
EPOCHS   = 30
BATCH    = 2
IMG_SIZE = 1024
DEVICE   = "cuda:0"

# Predicción
PRED_CONF = 0.25    # umbral de confianza para las detecciones