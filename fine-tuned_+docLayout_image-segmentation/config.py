#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py

Configuración común del tercer fine-tuning de olmOCR (segmentación de página
con docLayout): rutas del proyecto, publicaciones del corpus, ruta de
split.json, modelo base, parámetros de LoRA y de entrenamiento (incluido el
tamaño máximo de los recortes) y parámetros de inferencia. PROMPTS contiene los
dos prompts por región, text y graphic. Los demás scripts lo importan; no se
ejecuta directamente.

Uso:
    Editar aquí las rutas, los hiperparámetros o los prompts y volver a lanzar
    el script del pipeline correspondiente.

Nota:
    El split train/val/test se lee de split.json (config.SPLIT_PATH), en la raíz
    del repo. El corpus se espera en ../corpus/<publicacion>/{jpg,page}/.
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
OUTPUTS_DIR = ROOT / "outputs/transcripciones"

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

SPLIT_PATH = ROOT.parent / "split.json"

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
MODELO_MERGED_DIR = MODELS_DIR / "fine-tuned_+docLayout_image-segmentation"

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
GRAD_ACCUM_STEPS    = 8           # batch efectivo = 8, igual que antes
NUM_EPOCHS          = 3
LEARNING_RATE       = 2e-5
LR_SCHEDULER        = "cosine"
WARMUP_RATIO        = 0.05
QUANTIZATION_BITS   = 4
MAX_SEQ_LEN         = 2048         # ← de 4096 a 512
IMAGE_MAX_PIXELS    = 1_003_520   # se mantiene para inferencia de página completa
CROP_IMAGE_MAX_PIXELS = 200_704   # ← nuevo, para el fine-tuning con recortes

EVAL_STEPS          = 500         # ← de 100 a 500
SAVE_STEPS          = 500         # ← de 100 a 500
SAVE_TOTAL_LIMIT    = 5           # ← de 20 a 5

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
# 03_build_dataset.py selecciona el prompt según los metadatos del PAGE XML:
# tipo de región (heading, caption, image…), ReadingOrder, fracción de imagen.

PROMPTS = {

    "text": (
        "<image>\n"
        "Transcribe esta región de una página de prensa histórica en castellano "
        "(Filipinas, Puerto Rico, República Dominicana, Cuba o España, siglos XIX-XX).\n\n"
        "NORMAS:\n"
        "- Conserva erratas tipográficas, ortografía de época y puntuación original.\n"
        "- Texto completamente ilegible → [ilegible]\n"
        "- Texto parcialmente ilegible → [¿palabra?]\n"
        "- Ignora marcas de agua diagonales.\n"
        "- No añadas comentarios ni texto fuera de la transcripción."
    ),

    "graphic": (
        "<image>\n"
        "Describe brevemente el contenido visual de este recorte de prensa histórica "
        "(Filipinas, Puerto Rico, República Dominicana, Cuba o España, siglos XIX-XX).\n\n"
        "Formato: [Fotografía: descripción del contenido visual, personas, lugar, acción]\n\n"
        "NORMAS:\n"
        "- Si hay texto visible (pie de foto, crédito), transcríbelo: "
        "*[Pie: texto exacto]*\n"
        "- Si no hay nada transcribible responde únicamente: "
        "[IMAGEN - sin texto transcribible]\n"
        "- No añadas comentarios ni texto fuera de la descripción."
    ),
}