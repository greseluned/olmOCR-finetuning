# olmOCR-finetuning

Fine-tuning mediante QLoRA de [olmOCR-7B-0225-preview](https://huggingface.co/allenai/olmOCR-7B-0225-preview) para la transcripción automática de prensa histórica en español de los siglos XIX y XX (Asia, España y el Caribe hispánico).

El modelo base se ve limitado por la tipografía de época, el diseño columnar complejo, la mezcla de imágenes y texto y las convenciones ortográficas no estándar. El repositorio compara **cuatro workflows** que se diferencian en (a) quién decide el tipo de página o el layout y (b) si olmOCR transcribe páginas completas o regiones recortadas.

| # | Workflow | Decide tipo de página / layout | Unidad transcrita |
|---|---|---|---|
| 1 | olmOCR base | olmOCR (prompt clasificador) | Página |
| 2 | olmOCR fine-tuned | olmOCR fine-tuned (etiquetas derivadas del PAGE XML) | Página |
| 3 | olmOCR fine-tuned + docLayout (tipo de página) | docLayout-YOLO fine-tuned | Página |
| 4 | olmOCR fine-tuned + docLayout (segmentación) | docLayout-YOLO fine-tuned (bbox, tipo y orden) | Región |

## Estructura

```
olmOCR-finetuning/
├── corpus/<publicación>/{jpg,page}/            # Imágenes JPG y PAGE XML
├── olmOCR-base/                                # Workflow 1
├── fine-tuned_prompt-type/                     # Workflow 2
├── docLayout_fine-tuned/                       # Fine-tuning de docLayout-YOLO (usado en 3 y 4)
├── fine-tuned_+docLayout_prompt-type/          # Workflow 3
├── fine-tuned_+docLayout_image-segmentation/   # Workflow 4
├── CITATION.cff
├── evaluate.py                                 # Evaluación común
├── split.json                                  # Partición train/val/test compartida
└── README.md
```

Cada workflow tiene su propio `config.py` (rutas, hiperparámetros y prompts), `scripts/`, `dataset/`, `models/` y `outputs/{transcripciones,metricas}/`.

## Corpus y partición

Doce publicaciones más una miscelánea (solo train), con partición estratificada 80-10-10 por cabecera: 736 páginas de train, 90 de validación y 90 de test. Las publicaciones más representadas son *Semana* (334), *Fémina* (204) e *Hispanidad* (154); el resto tiene entre 5 y 50 ejemplares.

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


Todos los modelos entrenados (olmOCR y docLayout-YOLO) leen la misma partición de `split.json` (`{stem: "train"|"val"|"test"}`), de modo que ningún modelo ve en entrenamiento las páginas de test de otro y los cuatro workflows son comparables. Los stems deben ser únicos en todo el corpus.

---

## Workflow 1 · olmOCR base

Línea base sin fine-tuning. olmOCR clasifica la página con un prompt clasificador y la transcribe con el prompt del tipo asignado (`portada`, `texto_corrido`, `columnar`, `mixta`, `imagen_dominante`).

| Script | Función |
|---|---|
| `01_parse_pagexml.py` | PAGE XML → transcripción estructurada y tipo de página |
| `02_build_dataset.py` | Construye los JSONL de train/val/test |
| `03_inference.py` | Inferencia (`--modo-prompt auto\|unico`) y CER sobre test |
| `limpiar_salida.py` | Extrae `natural_text` de la salida JSON de olmOCR |

```bash
python scripts/03_inference.py --evaluar --jsonl
```

**Resultados (test, 90 páginas):** CER normalizado **0.2224** · WER normalizado **0.3837**.

---

## Workflow 2 · olmOCR fine-tuned (tipo de página desde PAGE XML)

Fine-tuning QLoRA con LLaMA-Factory. El tipo de página de cada ejemplo de entrenamiento se obtiene de los metadatos del PAGE XML con este árbol de decisión:

```
¿fichero termina en _p001 / _001?        → portada
¿≥ 50 % de regiones de imagen?           → imagen_dominante
¿1 columna?                              → texto_corrido
¿cabecera ancha y > 1 columna?           → mixta
¿> 1 columna?                            → columnar
```

Todos los prompts comparten las mismas normas: conservar erratas y ortografía de época, `[ilegible]`, `[¿palabra?]`, `[Fotografía: …]`, `*[Pie: …]*`, ignorar marcas de agua y separar columnas con `---COLUMNA N---`.

| Script | Función |
|---|---|
| `01_parse_pagexml.py` | Parser PAGE XML y clasificador de tipo de página |
| `02_build_dataset.py` | JSONL con el prompt de cada tipo |
| `03_train_lora.py` | QLoRA con LLaMA-Factory y exportación del modelo fusionado |
| `04_inference.py` | Inferencia (`--modo-prompt auto\|unico\|xml`) y CER |
| `05_evaluate.py` · `limpiar_salida.py` | Evaluación y limpieza de salida |

```bash
python scripts/03_train_lora.py
python scripts/04_inference.py --evaluar --jsonl
```

**Resultados:** CER normalizado **0.1610** · WER normalizado **0.3285**.

---

## docLayout-YOLO fine-tuned (componente de los workflows 3 y 4)

Fine-tuning de [DocLayout-YOLO](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench) sobre las regiones anotadas en los PAGE XML. Clases: `text` (TextRegion), `graphic` (ImageRegion, GraphicRegion, TableRegion) y `unknown`; se excluyen las presentes en menos de 3 páginas.

| Script | Función |
|---|---|
| `01_pagexml_to_yolo.py` | PAGE XML → etiquetas YOLO (según `split.json`) |
| `02_prepare_dataset.py` | Estructura Ultralytics, `gt_boxes.json` y `data.yaml` |
| `03_finetune.py` | Entrenamiento (30 épocas, batch 2, 1024 px, CodeCarbon opcional) |
| `04_predict.py` | Predicción de cajas (`--modelo baseline\|<nombre>`) |

```bash
python scripts/03_finetune.py
python scripts/04_predict.py --split test --modelo layout-model
```

Pesos resultantes: `models/layout-model/train/weights/best.pt`.

**Resultados (cajas, test):** IoU medio 0.6648 · F1 medio 0.6596 · EMR 0.7169 · composite 0.6622. Por clase: `text` F1 0.695, `graphic` F1 0.624.

Si se reentrena docLayout hay que regenerar las cachés y datasets de los workflows 3 y 4.

---

## Workflow 3 · olmOCR fine-tuned + docLayout (tipo de página)

Igual que el workflow 2, pero el tipo de página lo decide docLayout-YOLO (cajas `text` frente a `graphic`/`table`; las `unknown` se ignoran). La portada se sigue detectando por nombre de fichero. Si una página no está en la caché se usa el clasificador XML.

| Script | Función |
|---|---|
| `01_parse_pagexml.py` | Parser PAGE XML |
| `02_build_cache_layout.py` | docLayout sobre el corpus → `dataset/layout_cache.json` |
| `03_build_dataset.py` | JSONL según la caché y `split.json` (`--sin-cache` para usar el XML) |
| `04_train_lora.py` | QLoRA y exportación del modelo fusionado |
| `05_inference.py` | Fase 1: docLayout clasifica; fase 2: olmOCR transcribe (`--sin-layout`) |
| `limpiar_salida.py` | Limpieza de la salida de olmOCR |

```bash
python scripts/02_build_cache_layout.py
python scripts/03_build_dataset.py
python scripts/04_train_lora.py
python scripts/05_inference.py --evaluar --jsonl
```

**Resultados:** CER normalizado **0.1997** · WER normalizado **0.3652**.

---

## Workflow 4 · olmOCR fine-tuned + docLayout (segmentación en regiones)

docLayout segmenta la página en regiones (bbox, tipo `text`/`graphic` y orden de lectura por bandas horizontales de 20 px). olmOCR se entrena y transcribe **recortes de región** con dos prompts (`text`, `graphic`). En train y val las regiones son las de docLayout, emparejadas con el GT por IoU (0.30); en test se usan las regiones GT del PAGE XML. La salida es un PAGE XML por página con coordenadas, orden y texto.

| Script | Función |
|---|---|
| `01_parse_pagexml.py` | Parser PAGE XML (incluye `xml_a_regiones()`) |
| `02_build_cache_layout.py` | docLayout → `dataset/layout_regions_cache.json` |
| `03_build_dataset.py` | Recortes en `dataset/crops/` y JSONL por región; verifica contra `split.json` |
| `04_train_lora.py` | QLoRA sobre recortes (`--reanudar` para continuar) |
| `05_inference.py` | docLayout → recortes → olmOCR → PAGE XML (`--sin-layout`, reintentos ante OOM) |

```bash
python scripts/02_build_cache_layout.py
python scripts/03_build_dataset.py
python scripts/04_train_lora.py
python scripts/05_inference.py --evaluar --jsonl
```

En Windows conviene fijar `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Resultados:** CER normalizado **0.4408** · WER normalizado **0.5405**.

---

## Entrenamiento QLoRA (workflows 2–4)

| LoRA | Valor | Entrenamiento | Valor |
|---|---|---|---|
| rank / alpha / dropout | 32 / 64 / 0.1 | épocas | 3 |
| módulos | q, k, v, o, gate, up, down | learning rate | 2e-5 (cosine, warmup 0.05) |
| cuantización | 4 bits (bf16) | batch × acumulación | 1 × 8 |

Workflows 2 y 3: `max_seq_len` 4096, `image_max_pixels` 1 003 520, evaluación y guardado cada 50 pasos. Workflow 4: `max_seq_len` 2048, recortes de 200 704 px, evaluación y guardado cada 500 pasos.

## Evaluación

`evaluate.py` calcula CER y WER por página y los agrega (total, macro y micro) sobre todo el test y por publicación; con docLayout añade métricas de cajas (IoU, P/R/F1, EMR, composite).

```bash
python evaluate.py --jsonl ruta/test.jsonl --root ruta/olmOCR-finetuning
```

| Workflow | CER (norm.) | WER (norm.) | CER (prom.) | WER (prom.) |
|---|---|---|---|---|
| 1 · base | 0.2224 | 0.3837 | 1.0440* | 0.8286 |
| 2 · fine-tuned | **0.1610** | **0.3285** | **0.1947** | **0.3601** |
| 3 · + docLayout (tipo) | 0.1997 | 0.3652 | 0.2343 | 0.3947 |
| 4 · + docLayout (regiones) | 0.4408 | 0.5405 | 0.4337 | 0.5160 |

\* Distorsionado por una página atípica (CER 75.19).

El fine-tuning mejora claramente sobre el modelo base. Añadir docLayout no aporta mejora en OCR y la segmentación en regiones empeora los resultados; las cajas de docLayout tienen margen de mejora, sobre todo en precisión de `text` y detección de `graphic`.

## Hardware
 
Todos los experimentos (entrenamiento, inferencia y evaluación) se ejecutaron en un portátil **MSI Vector 16 HX AI A2XW** con procesador **Intel Core Ultra 9** y GPU **NVIDIA GeForce RTX 5090 Laptop**, bajo Windows.

## Notas de entorno (Windows)

- **PyTorch ≥ 2.6:** los scripts de docLayout fijan `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` para cargar sus checkpoints (solo seguro con pesos de confianza).
- **Fusión Conv+BN:** se desactiva `BaseModel.fuse` para evitar `'Conv' object has no attribute 'bn'`; solo afecta a la velocidad.
- **Lanzadores `.exe` del venv:** guardan la ruta absoluta de Python; si se mueve el entorno, `python -m pip install --force-reinstall --no-deps llamafactory`.
- **Rutas:** los scripts derivan las rutas de la ubicación de `config.py`.

## Licencia

[Creative Commons Atribución-NoComercial 4.0 Internacional (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

## Autoría

```text
Nuria Lucia Pirvu Curtusan
```

## Financiación

Trabajo desarrollado en el marco del proyecto GRESEL-UNED «Narrativas poscoloniales en periódicos en español de Asia, España y el Caribe hispánico» (PID2023-151280OB-C22), financiado por el Ministerio de Ciencia e Innovación / AEI.

## Referencias

- Allen Institute for AI. (2025). olmOCR-7B-0225-preview [Modelo de lenguaje de visión]. Hugging Face. https://huggingface.co/allenai/olmOCR-7B-0225-preview
- Garcia-Serrano, A. M., López Adrados, H., & Zabala Ortiz, B. (2026). Prensa histórica: Gold Standard con 9 números de "Acción Libertaria" (V1) (Version V1) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.21076010
- Garcia-Serrano, A. M., Torrente Gómez de la Venta, C., Martín Piquero, A., López Adrados, H., Linda Gaia, Z., & Alonso Aparicio, S. (2026). Gold Standard de "Semana, 1949" prensa histórica (V1) (Version V1) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.21245557
- Hiyouga. (2024). LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ Language Models. https://github.com/hiyouga/LLaMA-Factory
- Macicior-Mitxelena, J. y García-Serrano, A. (2026). From Paper to Pixel: Experimental Framework for Access to Historical Spanish Documents. https://github.com/jaionemacicior/from-paper-to-pixel
- Zhao, Z., Kang, H., Wang, B., & He, C. (2024). Doclayout-yolo: Enhancing document layout analysis through diverse synthetic data and global-to-local adaptive perception. arXiv preprint arXiv:2410.12628. https://arxiv.org/abs/2410.12628
