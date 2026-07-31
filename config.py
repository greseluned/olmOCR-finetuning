#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py
=========
Configuración centralizada del proyecto olmocr-finetune.
Todos los scripts importan desde aquí; edita solo este fichero.
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# RUTAS DEL PROYECTO
# ══════════════════════════════════════════════════════════════════════════════

# Raíz del proyecto (donde está este fichero: olmocr-finetune/)
ROOT = Path(__file__).resolve().parent

# Corpus externo — al mismo nivel que olmocr-finetune/
# Estructura:  ../corpus/<publicacion>/jpg/*.jpg
#              ../corpus/<publicacion>/page/*.xml
CORPUS_DIR = ROOT.parent / "corpus"

DATASET_DIR = ROOT / "dataset"
SCRIPTS_DIR = ROOT / "scripts"
MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# Subdirectorios del dataset
DATASET_TRAIN_DIR      = DATASET_DIR / "train"
DATASET_VALIDATION_DIR = DATASET_DIR / "validation"
DATASET_TEST_DIR       = DATASET_DIR / "test"

# JSONL finales por split (el JSONL completo pre-split ya no es necesario)
JSONL_TRAIN      = DATASET_TRAIN_DIR      / "train.jsonl"
JSONL_VALIDATION = DATASET_VALIDATION_DIR / "validation.jsonl"
JSONL_TEST       = DATASET_TEST_DIR       / "test.jsonl"

# ══════════════════════════════════════════════════════════════════════════════
# PUBLICACIONES DEL CORPUS
# ══════════════════════════════════════════════════════════════════════════════
# Cada entrada: nombre_carpeta -> comportamiento en el split
#   'estratificar' : aplica 80-10-10 dentro de esta publicación
#   'solo_train'   : todo va a train (material insuficiente para dividir)
#
# La estructura esperada dentro de cada carpeta es:
#   corpus/<publicacion>/jpg/<stem>.jpg
#   corpus/<publicacion>/page/<stem>.xml

PUBLICACIONES = {
    "accion-libertaria":    "estratificar",
    "boletín-de-la-cámara": "estratificar",
    "excelsior":            "estratificar",
    "fémina":               "estratificar",
    "filipinas":            "estratificar",
    "filipinas-ante-europa":"estratificar",
    "heraldo-de-la-mujer":  "estratificar",
    "hispanidad":           "estratificar",
    "la-malasia":           "estratificar",
    "la-vanguardia":        "estratificar",
    "semana":               "estratificar",
    "sorpresas-chicago":    "estratificar",
    "other":                "solo_train",
}

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT DEL DATASET
# ══════════════════════════════════════════════════════════════════════════════

SPLIT_TRAIN      = 0.80
SPLIT_VALIDATION = 0.10
SPLIT_TEST       = 0.10

SPLIT_SEED = 42

# Número mínimo de páginas para poder aplicar el split 80-10-10
# (necesitamos al menos 1 página en val y 1 en test → mínimo 10 páginas;
#  con menos se pone todo en train con un aviso)
SPLIT_MIN_PAGINAS = 10

# ══════════════════════════════════════════════════════════════════════════════
# MODELO BASE Y FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════

MODELO_BASE = "allenai/olmOCR-7B-0225-preview"

CHECKPOINT_DIR    = MODELS_DIR / "checkpoints"
MODELO_MERGED_DIR = MODELS_DIR / "prensa_hispanica_merged"

# ── Parámetros LoRA ───────────────────────────────────────────────────────────
LORA_RANK    = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Parámetros de entrenamiento ───────────────────────────────────────────────
BATCH_SIZE_TRAIN    = 1
GRAD_ACCUM_STEPS    = 8
NUM_EPOCHS          = 3
LEARNING_RATE       = 2e-5
LR_SCHEDULER        = "cosine"
WARMUP_RATIO        = 0.05
QUANTIZATION_BITS   = 4
MAX_SEQ_LEN         = 4096
IMAGE_MAX_PIXELS    = 1_003_520

EVAL_STEPS          = 50
SAVE_STEPS          = 50
SAVE_TOTAL_LIMIT    = 3

LLAMA_FACTORY_TEMPLATE = "qwen2_vl"

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

INFERENCIA_MODELO_DIR     = MODELO_MERGED_DIR
INFERENCIA_MAX_NEW_TOKENS = 4096
INFERENCIA_DO_SAMPLE      = False

# ══════════════════════════════════════════════════════════════════════════════
# INSTRUCCIONES DE USUARIO — sistema de prompts adaptativos
# ══════════════════════════════════════════════════════════════════════════════
# 02_build_dataset.py selecciona el prompt según los metadatos del PAGE XML:
# tipo de región (heading, caption, image…), ReadingOrder, fracción de imagen.

PROMPTS = {

    # ── Página de texto corrido (una sola columna) ───────────────────────────
    "texto_corrido": (
        "<image>\n"
        "Transcribe esta página de prensa histórica en castellano "
        "(Filipinas, Puerto Rico, República Dominicana, Cuba o España, "
        "siglos XIX-XX).\n\n"
        "El texto ocupa toda la anchura de la página sin dividirse en columnas. "
        "Transcríbelo de arriba a abajo en orden de lectura natural.\n\n"
        "NORMAS:\n"
        "- Conserva erratas tipográficas, ortografía de época y puntuación original.\n"
        "- Texto completamente ilegible → [ilegible]\n"
        "- Texto parcialmente ilegible → [¿palabra?]\n"
        "- Fotografías, grabados e ilustraciones → [Fotografía: descripción breve]\n"
        "- Pies de foto → *[Pie: texto exacto]*\n"
        "- Ignora marcas de agua diagonales.\n"
        "- No añadas comentarios ni texto fuera de la transcripción."
    ),

    # ── Página columnar (dos o más columnas en todo el cuerpo) ───────────────
    "columnar": (
        "<image>\n"
        "Transcribe esta página de prensa histórica en castellano "
        "(Filipinas, Puerto Rico, República Dominicana, Cuba o España, "
        "siglos XIX-XX) respetando el orden de lectura columna a columna, "
        "de izquierda a derecha y de arriba a abajo dentro de cada columna.\n\n"
        "NORMAS:\n"
        "- Usa '---COLUMNA N---' para separar columnas (N = número, empezando en 1).\n"
        "- Conserva erratas tipográficas, ortografía de época y puntuación original.\n"
        "- Texto completamente ilegible → [ilegible]\n"
        "- Texto parcialmente ilegible → [¿palabra?]\n"
        "- Fotografías, grabados e ilustraciones → [Fotografía: descripción breve]\n"
        "- Pies de foto → *[Pie: texto exacto]*\n"
        "- Ignora marcas de agua diagonales.\n"
        "- No añadas comentarios ni texto fuera de la transcripción."
    ),

    # ── Página mixta (cabecera a ancho completo + cuerpo columnar) ───────────
    "mixta": (
        "<image>\n"
        "Transcribe esta página de prensa histórica en castellano "
        "(Filipinas, Puerto Rico, República Dominicana, Cuba o España, "
        "siglos XIX-XX).\n\n"
        "La página combina bloques que ocupan toda la anchura (cabecera, "
        "título principal, sumario…) con un cuerpo dividido en columnas.\n\n"
        "NORMAS DE ESTRUCTURA:\n"
        "- Transcribe primero los bloques de cabecera en orden de arriba a abajo, "
        "  sin marcador de columna.\n"
        "- A continuación transcribe el cuerpo columnar usando '---COLUMNA N---' "
        "  (N empezando en 1), de izquierda a derecha y de arriba a abajo "
        "  dentro de cada columna.\n\n"
        "NORMAS GENERALES:\n"
        "- Conserva erratas tipográficas, ortografía de época y puntuación original.\n"
        "- Texto completamente ilegible → [ilegible]\n"
        "- Texto parcialmente ilegible → [¿palabra?]\n"
        "- Fotografías, grabados e ilustraciones → [Fotografía: descripción breve]\n"
        "- Pies de foto → *[Pie: texto exacto]*\n"
        "- Ignora marcas de agua diagonales.\n"
        "- No añadas comentarios ni texto fuera de la transcripción."
    ),

    # ── Página dominada por imágenes ─────────────────────────────────────────
    "imagen_dominante": (
        "<image>\n"
        "Esta página de prensa histórica en castellano "
        "(Filipinas, Puerto Rico, República Dominicana, Cuba o España, "
        "siglos XIX-XX) está dominada por fotografías, grabados o ilustraciones.\n\n"
        "Transcribe en este orden:\n"
        "1. Títulos o encabezados visibles.\n"
        "2. Para cada imagen, una descripción de su contenido:\n"
        "   [Fotografía: descripción del contenido visual, personas, lugar, acción]\n"
        "3. Pies de foto: *[Pie: texto exacto]*\n"
        "4. Créditos de fotógrafo o grabador.\n\n"
        "Si no hay ningún texto transcribible responde únicamente:\n"
        "[PÁGINA DE IMAGEN - sin texto transcribible]\n\n"
        "NORMAS:\n"
        "- Conserva la puntuación y ortografía originales.\n"
        "- Texto ilegible → [ilegible]\n"
        "- No añadas comentarios ni texto fuera de la transcripción."
    ),

    # ── Portada ───────────────────────────────────────────────────────────────
    "portada": (
        "<image>\n"
        "Esta es la portada de una publicación de prensa histórica en castellano "
        "(Filipinas, Puerto Rico, República Dominicana, Cuba o España, "
        "siglos XIX-XX). El texto visible puede ser escaso.\n\n"
        "Transcribe en este orden:\n"
        "1. Nombre de la publicación (masthead / cabecera).\n"
        "2. Subtítulo o lema de la revista / periódico.\n"
        "3. Fecha, número, año, tomo y precio si aparecen.\n"
        "4. Descripción de la imagen central o ilustración principal:\n"
        "   [Fotografía: descripción detallada del sujeto y composición]\n"
        "5. Pie de imagen si existe: *[Pie: texto exacto]*\n"
        "6. Cualquier otro texto en los márgenes, esquinas o filetes.\n\n"
        "No transcribas los elementos puramente decorativos (cenefas, orlas).\n\n"
        "NORMAS:\n"
        "- Conserva erratas tipográficas, ortografía de época y puntuación original.\n"
        "- Texto ilegible → [ilegible] | Parcialmente ilegible → [¿palabra?]\n"
        "- No añadas comentarios ni texto fuera de la transcripción."
    ),
}

# Alias de compatibilidad para scripts que aún usen INSTRUCCION_USUARIO
INSTRUCCION_USUARIO = PROMPTS["mixta"]