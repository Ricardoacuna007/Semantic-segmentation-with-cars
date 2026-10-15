#!/usr/bin/env python3
"""
Script para generar máscaras binarias y previsualizaciones superpuestas
a partir de anotaciones JSON de LabelMe en el dataset.
"""

import os
import argparse
import json
from PIL import Image, ImageDraw
import numpy as np

def generate_masks(images_dir, json_dir, masks_out_dir, previews_out_dir=None):
    os.makedirs(masks_out_dir, exist_ok=True)
    if previews_out_dir:
        os.makedirs(previews_out_dir, exist_ok=True)

    img_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg'))
        and not f.endswith(('_mask.png', '_preview.jpg', '_preview.png'))
    ])
    
    print(f"Procesando {len(img_files)} imágenes desde '{images_dir}'...")
        
    processed_count = 0
    for img_name in img_files:
        base_name = os.path.splitext(img_name)[0]
        
        # Buscar el archivo JSON correspondiente
        json_path = os.path.join(json_dir, f"{base_name}.json")
        if not os.path.exists(json_path):
            # Intentar buscar en images_dir si json_dir no lo contiene
            json_path = os.path.join(images_dir, f"{base_name}.json")
        
        if not os.path.exists(json_path):
            print(f"⚠️  Omitiendo {img_name}: No se encontró {base_name}.json")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        w = data.get('imageWidth')
        h = data.get('imageHeight')

        if not w or not h:
            img_full_path = os.path.join(images_dir, img_name)
            with Image.open(img_full_path) as tmp_img:
                w, h = tmp_img.size

        # Crear máscara binaria (0 = fondo, 255 = vehículo)
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)

        # Dibujar polígonos respetando el orden del archivo JSON
        for shape in data.get('shapes', []):
            label = shape.get('label', '').lower()
            pts = [tuple(pt) for pt in shape.get('points', [])]
            if len(pts) >= 3:
                fill_val = 255 if label == 'car' else 0
                draw.polygon(pts, fill=fill_val)

        # Guardar la máscara en formato PNG
        mask_filename = f"{base_name}.png"
        mask_path = os.path.join(masks_out_dir, mask_filename)
        mask.save(mask_path)

        # Si se solicita carpeta de previsualización
        if previews_out_dir:
            img_path = os.path.join(images_dir, img_name)
            with Image.open(img_path) as orig_img:
                orig_rgb = orig_img.convert('RGB')
                mask_arr = np.array(mask)
                img_arr = np.array(orig_rgb)

                color_mask = np.zeros_like(img_arr)
                color_mask[mask_arr > 0] = [255, 0, 0]  # Capa roja sobre vehículos

                overlay_arr = img_arr.copy()
                car_indices = mask_arr > 0
                overlay_arr[car_indices] = (img_arr[car_indices] * 0.6 + color_mask[car_indices] * 0.4).astype(np.uint8)

                overlay_img = Image.fromarray(overlay_arr)
                preview_path = os.path.join(previews_out_dir, f"{base_name}_preview.jpg")
                overlay_img.save(preview_path, quality=90)

        processed_count += 1
        print(f"  ✓ {base_name}: Máscara generada en {mask_path}")

    print(f"\n¡Completado! Se generaron {processed_count} máscaras exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar máscaras desde anotaciones LabelMe")
    parser.add_argument("--images_dir", default="/home/hackriki/Desktop/dual/py/proyect_cars/images", help="Directorio con imágenes")
    parser.add_argument("--json_dir", default="/home/hackriki/Desktop/dual/py/proyect_cars/jsons", help="Directorio con archivos JSON")
    parser.add_argument("--masks_out", default="/home/hackriki/Desktop/dual/py/proyect_cars/masks", help="Directorio de salida para máscaras PNG")
    parser.add_argument("--previews_out", default="/home/hackriki/Desktop/dual/py/proyect_cars/previews", help="Directorio de salida para previsualizaciones")

    args = parser.parse_args()
    generate_masks(args.images_dir, args.json_dir, args.masks_out, args.previews_out)
