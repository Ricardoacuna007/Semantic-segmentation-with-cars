
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from dataset import CarDataset
from model import UNet


# --------------------------------------------------
# 1. Configuración
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
SPLIT_DIR = BASE_DIR / "Dataset" / "dataset_split"
MODEL_PATH = BASE_DIR / "checkpoints" / "best_unet_no_aug_100.pth"
OUTPUT_DIR = BASE_DIR / "predictions_val_no_aug_100_best"

IMAGE_SIZE = 256
BATCH_SIZE = 4
THRESHOLD = 0.5
MAX_PREVIEWS = 8

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Dispositivo:", device)


# --------------------------------------------------
# 2. Cargar el dataset de validación
# --------------------------------------------------

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"No se encontró el modelo: {MODEL_PATH}"
    )

val_dataset = CarDataset(
    images_dir=SPLIT_DIR / "val" / "images",
    masks_dir=SPLIT_DIR / "val" / "masks",
    image_size=IMAGE_SIZE
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 3. Cargar la U-Net entrenada
# --------------------------------------------------

model = UNet(
    in_channels=3,
    out_channels=1,
    base_channels=16
).to(device)

weights = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(weights)
model.eval()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Imágenes de validación:", len(val_dataset))
print("Modelo cargado correctamente.")


# --------------------------------------------------
# 4. Función para superponer los errores
# --------------------------------------------------

def crear_superposicion(image_array, pred, target):
    """
    Verde: coche correctamente detectado (TP).
    Rojo: falso positivo (FP).
    Azul: falso negativo (FN).
    El fondo correctamente clasificado conserva su color.
    """

    overlay = image_array.astype(np.float32).copy()

    tp = pred & target
    fp = pred & ~target
    fn = ~pred & target

    alpha = 0.50

    # Verde: predicción y máscara real coinciden en coche.
    overlay[tp] = (
        (1 - alpha) * overlay[tp]
        + alpha * np.array([0, 255, 0])
    )

    # Rojo: predijo coche, pero era fondo.
    overlay[fp] = (
        (1 - alpha) * overlay[fp]
        + alpha * np.array([255, 0, 0])
    )

    # Azul: había coche en la máscara, pero no se detectó.
    overlay[fn] = (
        (1 - alpha) * overlay[fn]
        + alpha * np.array([0, 0, 255])
    )

    return overlay.clip(0, 255).astype(np.uint8)


# --------------------------------------------------
# 5. Generar comparaciones y calcular IoU
# --------------------------------------------------

iou_scores = []
saved_images = 0

with torch.no_grad():

    for images, masks in val_loader:

        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        probabilities = torch.sigmoid(logits)

        predictions = probabilities >= THRESHOLD
        targets = masks >= 0.5

        for i in range(images.size(0)):

            pred = predictions[i, 0].cpu().numpy()
            target = targets[i, 0].cpu().numpy()

            intersection = np.logical_and(
                pred, target
            ).sum()

            union = np.logical_or(
                pred, target
            ).sum()

            iou = 1.0 if union == 0 else intersection / union
            iou_scores.append(float(iou))

            # Solo guardar las primeras imágenes solicitadas.
            if saved_images >= MAX_PREVIEWS:
                continue

            image_array = (
                images[i]
                .cpu()
                .permute(1, 2, 0)
                .numpy() * 255
            ).clip(0, 255).astype(np.uint8)

            target_array = target.astype(
                np.uint8
            ) * 255

            pred_array = pred.astype(
                np.uint8
            ) * 255

            overlay_array = crear_superposicion(
                image_array, pred, target
            )

            original = Image.fromarray(image_array)
            real_mask = Image.fromarray(
                target_array
            ).convert("RGB")
            predicted_mask = Image.fromarray(
                pred_array
            ).convert("RGB")
            overlay = Image.fromarray(overlay_array)

            # Crear una imagen con cuatro paneles.
            panel_width = IMAGE_SIZE
            header_height = 42

            panel = Image.new(
                "RGB",
                (panel_width * 4, IMAGE_SIZE + header_height),
                "white"
            )

            draw = ImageDraw.Draw(panel)

            headers = [
                "Original",
                "Mascara real",
                "Prediccion",
                f"Errores - IoU: {iou:.3f}"
            ]

            for j, title in enumerate(headers):
                draw.text(
                    (j * panel_width + 8, 8),
                    title,
                    fill="black"
                )

            panel.paste(original, (0, header_height))
            panel.paste(
                real_mask, (panel_width, header_height)
            )
            panel.paste(
                predicted_mask, (panel_width * 2, header_height)
            )
            panel.paste(
                overlay, (panel_width * 3, header_height)
            )

            saved_images += 1

            panel.save(
                OUTPUT_DIR
                / f"validacion_{saved_images:02d}.png"
            )


# --------------------------------------------------
# 6. Mostrar el resumen
# --------------------------------------------------

mean_iou = sum(iou_scores) / len(iou_scores)

print("\nEvaluación visual completada.")
print(f"IoU promedio de validación: {mean_iou:.4f}")
print("Comparaciones guardadas:", saved_images)
print("Carpeta de resultados:", OUTPUT_DIR)
print("Umbral de predicción:", THRESHOLD)
