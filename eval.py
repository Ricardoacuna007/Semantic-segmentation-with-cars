
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
MODEL_PATH = BASE_DIR / "checkpoints" / "best_unet_aug_100.pth"
OUTPUT_DIR = BASE_DIR / "predictions_test" /"best_unet_aug_100"

IMAGE_SIZE = 256
BATCH_SIZE = 4
THRESHOLD = 0.5

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Dispositivo:", device)


# --------------------------------------------------
# 2. Comprobar que existe el modelo entrenado
# --------------------------------------------------

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"No se encontró el modelo entrenado: {MODEL_PATH}"
    )


# --------------------------------------------------
# 3. Cargar el conjunto de prueba
# --------------------------------------------------

test_dataset = CarDataset(
    images_dir=SPLIT_DIR / "test" / "images",
    masks_dir=SPLIT_DIR / "test" / "masks",
    image_size=IMAGE_SIZE
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Imágenes para evaluar:", len(test_dataset))


# --------------------------------------------------
# 4. Cargar la U-Net y sus pesos aprendidos
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

print("Modelo cargado correctamente.")


# --------------------------------------------------
# 5. Preparar métricas y carpeta de resultados
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

iou_scores = []
dice_scores = []
saved_images = 0
MAX_PREVIEWS = 8


# --------------------------------------------------
# 6. Evaluar sin entrenar
# --------------------------------------------------

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        # La red devuelve logits.
        logits = model(images)

        # Convertir logits en probabilidades entre 0 y 1.
        probabilities = torch.sigmoid(logits)

        # Umbral: probabilidad >= 0.5 significa coche.
        predictions = probabilities >= THRESHOLD
        targets = masks >= 0.5

        # Calcular IoU y Dice para cada imagen.
        for i in range(images.size(0)):

            pred = predictions[i, 0]
            target = targets[i, 0]

            intersection = torch.logical_and(
                pred, target
            ).sum().item()

            union = torch.logical_or(
                pred, target
            ).sum().item()

            pred_pixels = pred.sum().item()
            target_pixels = target.sum().item()

            # Si ambas máscaras están vacías, coinciden.
            if union == 0:
                iou = 1.0
            else:
                iou = intersection / union

            denominator = pred_pixels + target_pixels

            if denominator == 0:
                dice = 1.0
            else:
                dice = (
                    2.0 * intersection / denominator
                )

            iou_scores.append(iou)
            dice_scores.append(dice)

            # Guardar comparaciones visuales de los primeros ejemplos.
            if saved_images < MAX_PREVIEWS:

                image_array = (
                    images[i]
                    .detach()
                    .cpu()
                    .permute(1, 2, 0)
                    .numpy() * 255.0
                ).clip(0, 255).astype(np.uint8)

                target_array = (
                    targets[i, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8) * 255
                )

                pred_array = (
                    predictions[i, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8) * 255
                )

                original = Image.fromarray(image_array)
                real_mask = Image.fromarray(
                    target_array
                ).convert("RGB")
                predicted_mask = Image.fromarray(
                    pred_array
                ).convert("RGB")

                # Tres paneles: original, máscara real, predicción.
                panel = Image.new(
                    "RGB",
                    (IMAGE_SIZE * 3, IMAGE_SIZE + 30),
                    "white"
                )

                draw = ImageDraw.Draw(panel)
                draw.text((10, 8), "Original", fill="black")
                draw.text(
                    (IMAGE_SIZE + 10, 8),
                    "Mascara real",
                    fill="black"
                )
                draw.text(
                    (IMAGE_SIZE * 2 + 10, 8),
                    "Prediccion",
                    fill="black"
                )

                panel.paste(original, (0, 30))
                panel.paste(real_mask, (IMAGE_SIZE, 30))
                panel.paste(
                    predicted_mask,
                    (IMAGE_SIZE * 2, 30)
                )

                panel.save(
                    OUTPUT_DIR
                    / f"comparacion_{saved_images + 1:02d}.png"
                )

                saved_images += 1


# --------------------------------------------------
# 7. Mostrar los resultados finales
# --------------------------------------------------

mean_iou = sum(iou_scores) / len(iou_scores)
mean_dice = sum(dice_scores) / len(dice_scores)

print("\nResultados del conjunto de prueba")
print(f"IoU promedio:  {mean_iou:.4f}")
print(f"Dice promedio: {mean_dice:.4f}")

print("\nComparaciones guardadas:", saved_images)
print("Carpeta de resultados:", OUTPUT_DIR)
