# olmOCR-finetuning

## Descripción

Este repositorio recoge el pipeline de fine-tuning de [olmOCR-7B](https://huggingface.co/allenai/olmOCR-7B-0225-preview) (Qwen2-VL-7B) para la transcripción automática de prensa histórica en español mediante QLoRA. El objetuvo es adaptar un modelo OCR de propósito general a las características específicas de periódicos publicados entre los años 1850 y 1950 procedentes de Cuba, España, Filipinas, Puerto Rico o República Dominicana.

olmOCR es un VLM (Vision Language Model) desarrollado por el Allen Institute for AI a partir de Qwen2-VL-7B y entrenado para transcribir documentos de todo tipo. Su rendimiento sobre prensa histórica en español se ve limitado por factores que el fine-tuning pretende corregir:

- **Tipografía de época**: fuentes con tipos desgastados, ligaturas, tinta corrida y degradación del soporte.
- **Diseño columnar complejo**: la mayoría de páginas combina una cabecera de ancho completo (masthead, título principal, sumario) con un cuerpo dividido en dos a seis columnas tipográficas.
- **Mezcla de imágenes y texto**: grabados, fotografías con pie de foto, filetes decorativos y marcas de agua que el modelo base tiende a ignorar o a transcribir como ruido.
- **Convenciones ortográficas no estándares**: léxico, acentuación y puntuación de época que difieren del español contemporáneo.
- **Instrucción única**: el modelo base recibe siempre la misma instrucción genérica; el fine-tuning le enseña a comportarse de forma diferente según el tipo estructural de cada página.

El pipeline ha sido desarrollado en el marco del proyecto GRESEL-UNED: "Narrativas poscoloniales en periódicos en español de Asia, España y el Caribe hispánico" (PID2023-151280OB-C22), financiado por el Ministerio de Ciencia e Innovación / AEI.
 
---
 
## Estructura del repositorio
  
```
olmOCR-finetuning/
├── corpus/                          # Corpus de imágenes y transcripciones ground-truth de Transkribus
|   └── <publicación>/
|       ├── jpg/                     # Imágenes de página escaneada en formato JPG
|       └── page/                    # Trascripciones en formato PAGE XML (Transkribus)
├── config.py                        # Configuración centralizada del proyecto
├── scripts/
│   ├── 01_parse_pagexml.py          # Parser PAGE XML + CLI de diagnóstico
│   ├── 02_build_dataset.py          # Construcción de los JSONL de entrenamiento
│   ├── 03_train_lora.py             # Fine-tuning QLoRA con LLaMA-Factory
│   └── 04_inference.py              # Inferencia y evaluación CER
├── dataset/
│   ├── train/train.jsonl
│   ├── validation/validation.jsonl
│   └── test/test.jsonl
└── outputs/                         # Transcripciones generadas
```
 
### Publicaciones del corpus

Las publicaciones incluidas en el dataset de entrenamiento son:

| Publicación | Split |
|---|---|
| *Acción Libertaria* | 80-10-10 |
| *Boletín de la Cámara de Comercio Española de Filipinas* | 80-10-10 |
| *Excelsior* | 80-10-10 |
| *Fémina* | 80-10-10 |
| *Filipinas* | 80-10-10 |
| *Filipinas Ante Europa* | 80-10-10 |
| *Heraldo de la Mujer* | 80-10-10 |
| *Hispanidad* | 80-10-10 |
| *La Malasia* | 80-10-10 |
| *La Vanguardia* | 80-10-10 |
| *Semana* | 80-10-10 |
| *Sorpresas Chicago* | 80-10-10 |
| Miscelánea (*other*) | solo train |

Las publicaciones marcadas como `solo_train` tienen material insuficiente para dividir en splits (< 10 páginas válidas). El split se aplica **dentro de cada publicación** para garantizar que todos los títulos están representados en los conjuntos de validación y test, y se shufflea al final para que el orden de entrenamiento mezcle publicaciones.
 
---
 
## Pipeline
 
### `01_parse_pagexml.py`: parser de PAGE XML
 
Convierte cada PAGE XML de Transkribus en texto plano estructurado listo para usarse como ground-truth. Puede ejecutarse también como CLI de diagnóstico sobre ficheros individuales o directorios.
 
**Estrategia de extracción:**
 
1. **ReadingOrder explícito**: respeta el atributo `readingOrder` de Transkribus sin reordenar las regiones por coordenada Y.

2. **Inclusión de regiones de imagen**: `ImageRegion` y `GraphicRegion` se insertan en el texto de ground-truth como marcadores `[Fotografía: descripción]`. Esto enseña al modelo a indicar explícitamente la presencia de ilustraciones en lugar de ignorarlas o alucinar texto.

3. **Detección de columnas por gap de centroides X**: en lugar de una franja fija, el parser calcula los centroides X de todas las regiones de texto y detecta columnas buscando gaps superiores al 12 % del ancho de página. 

4. **Cabeceras de ancho completo**: las regiones con `type="heading"` se tratan siempre como elementos de ancho completo, independientemente de su posición X, evitando que se asignen a una columna cuando encabezan el cuerpo columnar.

5. **Marcadores de columna en el texto**: cuando hay más de una columna, el texto incluye separadores `---COLUMNA N---` que el modelo aprende a reproducir.

6. **Metadatos para clasificación**: el parser devuelve un diccionario de metadatos (`num_columnas_detectadas`, `fraccion_imagen`, `tiene_cabecera_ancha`, `n_texto`, `n_imagen`, `ancho_pagina`, `alto_pagina`) que el clasificador de tipo de página usa directamente.
---
 
### `02_build_dataset.py`: construcción del dataset
 
Descubre todos los pares JPG + PAGE XML del corpus, parsea cada par con `01_parse_pagexml.py`, clasifica el tipo de página, selecciona el prompt correspondiente y escribe los tres JSONL finales (`train`, `validation`, `test`).

Cada registro JSONL tiene el formato:

```json
{
  "image": "/ruta/absoluta/pagina.jpg",
  "conversations": [
    {"role": "user",      "content": "<instrucción adaptativa según tipo de página>"},
    {"role": "assistant", "content": "<transcripción ground-truth>"}
  ],
  "_meta": {
    "fuente":      "excelsior_1932-09-10_p003",
    "publicacion": "excelsior",
    "tipo_pagina": "mixta",
    "columnas":    3,
    "n_texto":     18,
    "n_imagen":    2,
    "frac_imagen": 0.1
  }
}
```
 
El campo `_meta` no lo consume LLaMA-Factory pero sirve para revisar la distribución del dataset y para diagnóstico post-entrenamiento.
 
---
 
#### Clasificación de tipo de página
 
Cada página se clasifica en uno de cinco tipos usando los metadatos del parser, siguiendo este árbol de decisión:

```
¿nombre de fichero termina en _p001 o _001?   →  portada
¿fracción de regiones de imagen ≥ 50 %?       →  imagen_dominante
¿num_columnas = 1?                            →  texto_corrido
¿tiene cabecera ancha Y num_columnas > 1?     →  mixta
¿num_columnas > 1?                            →  columnar
```

La portada se detecta por convención de nombre de fichero porque la primera página de cada número tiene características visuales muy distintas (masthead tipográfico, logotipo, sumario) que merecen instrucciones propias.
 
---
 
#### Sistema de prompts adaptativos
 
El dataset usa cinco prompts distintos, uno por tipo de página. Durante el entrenamiento, el modelo aprende a interpretar instrucciones con distintos niveles de detalle estructural; durante la inferencia, el prompt actúa como señal de contexto que conduce la generación hacia el formato esperado.

| Tipo | Instrucción característica |
|---|---|
| `portada` | Masthead → subtítulo → fecha/número → imagen central → pie → texto marginal |
| `imagen_dominante` | Titular + descripción de cada imagen + pies de foto + créditos |
| `texto_corrido` | Transcripción de arriba a abajo sin marcadores de columna |
| `columnar` | Columna a columna con separadores `---COLUMNA N---` |
| `mixta` | Bloques de cabecera primero; luego cuerpo columnar con separadores |

Todos los prompts comparten las mismas normas de transcripción:

- Conservar erratas tipográficas, ortografía de época y puntuación original.
- Texto completamente ilegible → `[ilegible]`
- Texto parcialmente ilegible → `[¿palabra?]`
- Fotografías, grabados e ilustraciones → `[Fotografía: descripción breve]`
- Pies de foto → `*[Pie: texto exacto]*`
- Ignorar marcas de agua diagonales.
 
---
 
### `03_train_lora.py`: fine-tuning QLoRA
 
Orquesta el entrenamiento mediante [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), que gestiona el ciclo completo: carga del modelo base, aplicación de LoRA, cuantización QLoRA, evaluación periódica, guardado de checkpoints y exportación del modelo fusionado.
 
**Parámetros LoRA:**

| Parámetro | Valor |
|---|---|
| `lora_rank` | 32 |
| `lora_alpha` | 64 |
| `lora_dropout` | 0.1 |
| Módulos objetivo | q, k, v, o, gate, up, down |
| `quantization_bit` | 4 (QLoRA) |
| Precisión | bfloat16 |

**Parámetros de entrenamiento:**

| Parámetro | Valor |
|---|---|
| `num_train_epochs` | 3 |
| `learning_rate` | 2e-5 |
| `lr_scheduler_type` | cosine |
| `warmup_ratio` | 0.05 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 |
| `max_seq_len` | 4096 tokens |
| `image_max_pixels` | 1 003 520 px |
 
El script genera un `dataset_info.json` local con rutas absolutas a los JSONL y lo pasa a LLaMA-Factory mediante el parámetro `dataset_dir`, evitando modificar la instalación pip. Al terminar el entrenamiento, exporta el modelo fusionado (pesos base + adaptadores LoRA).
 
---
 
### `04_inference.py`: inferencia
 
Carga el modelo fusionado y transcribe imágenes individuales, directorios completos o el test set completo. En modo evaluación calcula el CER (Character Error Rate) respecto a los textos de referencia del test set usando distancia de Levenshtein a nivel de carácter. La salida se guarda como TXT (una página = un fichero) y opcionalmente como JSONL con metadatos de tiempo de procesado.

Durante la inferencia, si el PAGE XML paralelo está disponible, el script reclasifica la página en tiempo real para seleccionar el prompt correcto. Sin XML, usa `mixta` como fallback (el tipo más frecuente en el corpus y el que cubre más casos estructurales).
 
---
 
## Métricas de evaluación

*Esta sección se completará con los resultados del fine-tuning calculando las métricas del framework desarrollado por Macicior-Mitxelena y García-Serrano (2025) (https://github.com/jaionemacicior/from-paper-to-pixel).*

La evaluación sobre el test set mide el CER (Character Error Rate) respecto a las transcripciones manuales de Transkribus mediante distancia de Levenshtein a nivel de carácter. Los resultados se guardan en `outputs/evaluacion/evaluacion_test.json`.

---

## Metodología de transcripción del corpus

[...]

---

## Autoras

[...]

---

## Cita

[...]

---

## Licencia

Este repositorio se distribuye bajo la licencia **Creative Commons Atribución-NoComercial 4.0 Internacional (CC BY-NC 4.0)**.

Puede compartir y adaptar el material para fines no comerciales siempre que se proporcione atribución adecuada.

Más información: https://creativecommons.org/licenses/by-nc/4.0/

---

## Financiación

Este trabajo ha sido financiado por el Ministerio de Ciencia e Innovación / AEI en el marco del proyecto coordinado GRESEL-UNED (PID2023-151280OB-C22).

---

## Referencias

- Hiyouga. (2024). LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ Language Models [Software]. https://github.com/hiyouga/LLaMA-Factory
- Lu, L. et al. (2024). olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models. Allen Institute for AI. https://huggingface.co/allenai/olmOCR-7B-0225-preview
- Macicior-Mitxelena, J. y García-Serrano, A. (2026). From Paper to Pixel: Experimental Framework for Access to Historical Spanish Documents [Software]. https://github.com/jaionemacicior/from-paper-to-pixel
- READ-COOP SCE. (2026). Transkribus [Software]. https://transkribus.eu/
- Wang, P. et al. (2024). Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. https://arxiv.org/abs/2409.12191