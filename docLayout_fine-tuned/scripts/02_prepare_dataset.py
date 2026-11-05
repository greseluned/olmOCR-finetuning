#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_prepare_dataset.py
=====================

Prepara el dataset para entrenar y evaluar el modelo docLayout a partir
de las imágenes y las anotaciones YOLO generadas en el paso anterior.

El script reorganiza los archivos en una estructura estandarizada,
convierte las etiquetas YOLO a coordenadas absolutas XYXY y genera
los archivos de configuración y ground truth necesarios para el
entrenamiento y la evaluación del modelo.

FUNCIONAMIENTO:
    1. Carga el mapeo de clases desde classes.json, que relaciona
       los nombres de las clases con sus identificadores numéricos.

    2. Recorre los conjuntos train, val y test del corpus y copia
       las imágenes y sus correspondientes etiquetas YOLO a una
       estructura de directorios estandarizada.

    3. Lee las dimensiones de cada imagen y convierte las
       bounding boxes del formato YOLO normalizado
       (class_id, x_center, y_center, width, height) al formato
       XYXY con coordenadas absolutas en píxeles
       (x1, y1, x2, y2).

    4. Genera un archivo gt_boxes.json para cada split con las
       anotaciones de ground truth de las imágenes, incluyendo
       las coordenadas de cada región y el nombre de su clase.

    5. Genera un archivo data.yaml con las rutas de los conjuntos
       de entrenamiento y validación, el número de clases y sus
       nombres. Este archivo sirve como configuración del dataset
       para el modelo YOLO.

ESTRUCTURA DE ENTRADA:

    data/<NOMBRE_CORPUS>/
        classes.json
        train/
            images/
            yolo_labels/
        val/
            images/
            yolo_labels/
        test/
            images/
            yolo_labels/

ESTRUCTURA DE SALIDA:

    <NOMBRE_CORPUS>/
        data.yaml
        images/
            train/
            val/
            test/
        labels/
            train/
                gt_boxes.json
                *.txt
            val/
                gt_boxes.json
                *.txt
            test/
                gt_boxes.json
                *.txt

REQUISITOS:
    - Python 3.10 o superior.
    - Pillow (PIL).
    - Un dataset generado previamente por 01_pagexml_to_yolo.py,
      con la estructura de carpetas y los archivos esperados.

USO:

    Ejecutar el script indicando el nombre del corpus:

        python docLayout_fine-tuned/02_prepare_dataset.py --corpus data_regiones

    El nombre indicado debe corresponder a una carpeta existente
    dentro del directorio data/ del proyecto.

SALIDA:

    El script genera un directorio con el dataset preparado para
    docLayout, que contiene:

    - Imágenes y etiquetas organizadas por split.
    - Archivos gt_boxes.json con las anotaciones de referencia
      en coordenadas absolutas XYXY.
    - Un archivo data.yaml con la configuración de clases y las
      rutas de entrenamiento y validación.

IMPORTANTE:
    Este script no realiza el entrenamiento del modelo ni modifica
    las anotaciones originales. Su función es adaptar y reorganizar
    los datos generados en el paso anterior para que puedan ser
    utilizados por el pipeline de docLayout.

    Debe ejecutarse después de 01_pagexml_to_yolo.py y antes de
    iniciar el entrenamiento o la evaluación del modelo.
"""

import argparse
import json
import os
import random
import shutil
 
from PIL import Image
 
random.seed(42)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — COPIA Y CONVERSIÓN DE ARCHIVOS
# ══════════════════════════════════════════════════════════════════════════════
 
def copy_files(img_list, images_folder, labels_folder, split, output_dir, id_to_class):
    """
    Copia las imágenes y sus etiquetas YOLO a la estructura de salida.
 
    Parámetros:
        img_list      — nombres de archivo de las imágenes del split
        images_folder — carpeta de origen de las imágenes
        labels_folder — carpeta de origen de las etiquetas YOLO
        split         — nombre del split (train, val o test)
        output_dir    — directorio raíz de salida
        id_to_class   — mapeo de identificador numérico a nombre de clase
    """
    for img_name in img_list:
        # Copiar imagen
        shutil.copy(
            os.path.join(images_folder, img_name),
            os.path.join(output_dir, 'images', split, img_name),
        )
 
        # Copiar archivo de etiquetas
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label = os.path.join(labels_folder, label_name)
        dst_label = os.path.join(output_dir, 'labels', split, label_name)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            print(f"  [AVISO] No se ha encontrado la etiqueta correspondiente a {img_name}.")
 
 
def yolo_to_xyxy(line, img_w, img_h):
    """
    Convierte una línea de etiqueta YOLO a coordenadas absolutas XYXY.
 
    Parámetros:
        line  — 'class_id x_center y_center width height' (valores normalizados)
        img_w — ancho de la imagen en píxeles
        img_h — alto de la imagen en píxeles
 
    Devuelve:
        (cls_id, [x1, y1, x2, y2]) con coordenadas en píxeles
    """
    cls_id, x_c, y_c, bw, bh = map(float, line.strip().split())
    cls_id = int(cls_id)
    x_c, y_c, bw, bh = x_c * img_w, y_c * img_h, bw * img_w, bh * img_h
    x1 = x_c - bw / 2
    y1 = y_c - bh / 2
    x2 = x_c + bw / 2
    y2 = y_c + bh / 2
    return cls_id, [x1, y1, x2, y2]
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — PREPARACIÓN DEL DATASET
# ══════════════════════════════════════════════════════════════════════════════
 
def prepare_dataset(corpus_name):
    """
    Reorganiza el corpus, genera los gt_boxes.json y el data.yaml.
 
    Parámetros:
        corpus_name — nombre de la carpeta del corpus dentro de data/
    """
    BASE_DIR = 'olmOCR-finetuning/docLayout_fine-tuned'
    OUTPUT_DIR = os.path.join(BASE_DIR, corpus_name)
    CORPUS_PATH = os.path.join(BASE_DIR, 'data', corpus_name)
 
    # Cargar correspondencia entre clases e identificadores
    with open(os.path.join(CORPUS_PATH, 'classes.json'), 'r') as f:
        class_to_id = json.load(f)
    id_to_class = {v: k for k, v in class_to_id.items()}
 
    splits = ['train', 'val', 'test']
 
    for split in splits:
        images_folder = os.path.join(CORPUS_PATH, split, 'images')
        labels_folder = os.path.join(CORPUS_PATH, split, 'yolo_labels')
 
        # Crear carpetas de salida
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)
 
        # Obtener lista de imágenes
        imgs = [f for f in os.listdir(images_folder)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
 
        # Copiar imágenes y etiquetas
        copy_files(imgs, images_folder, labels_folder, split, OUTPUT_DIR, id_to_class)
 
        # Convertir etiquetas YOLO a cajas de referencia (ground truth)
        gt_boxes_dict = {}
        for img_name in imgs:
            label_file = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(labels_folder, label_file)
            img_path = os.path.join(images_folder, img_name)
 
            if not os.path.exists(label_path):
                continue
 
            with Image.open(img_path) as im:
                w, h = im.size
 
            boxes = []
            with open(label_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    cls_id, xyxy = yolo_to_xyxy(line, w, h)
                    boxes.append({
                        "xyxy": xyxy,
                        "class_name": id_to_class.get(cls_id, str(cls_id)),
                    })
 
            gt_boxes_dict[os.path.splitext(img_name)[0]] = boxes
 
        # Guardar las cajas de referencia de este split
        gt_json_path = os.path.join(OUTPUT_DIR, 'labels', split, "gt_boxes.json")
        with open(gt_json_path, "w", encoding="utf-8") as f:
            json.dump(gt_boxes_dict, f, indent=4, ensure_ascii=False)
 
        print(f"  Split '{split}' preparado correctamente.")
        print(f"  Anotaciones de referencia guardadas en: {gt_json_path}")
 
    # Generar data.yaml
    num_classes = max(class_to_id.values()) + 1
    class_names = [None] * num_classes
    for cls, idx in class_to_id.items():
        class_names[idx] = cls
 
    output_yaml = os.path.join(OUTPUT_DIR, 'data.yaml')
    yaml_content = f"""
train: {os.path.join('images', 'train')}
val: {os.path.join('images', 'val')}
nc: {num_classes}
names: {class_names}
"""
 
    with open(output_yaml, 'w') as f:
        f.write(yaml_content.strip())
 
    print(f"  data.yaml generado en: {output_yaml}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — CLI
# ══════════════════════════════════════════════════════════════════════════════
 
def main():
    parser = argparse.ArgumentParser(
        description="Prepara el dataset para entrenar y evaluar DocLayout-YOLO.",
    )
    parser.add_argument(
        "--corpus", type=str, required=True,
        help="Nombre de la carpeta del corpus dentro de data/",
    )
    args = parser.parse_args()
 
    prepare_dataset(args.corpus)
 
 
if __name__ == "__main__":
    main()