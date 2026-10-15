
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CarDataset
from model import UNet


# --------------------------------------------------
# 1. Configuración
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
SPLIT_DIR = BASE_DIR / "Dataset" / "dataset_split"
MODEL_PATH = BASE_DIR / "checkpoints" / "best_unet_aug.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 4

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Dispositivo:", device)


# --------------------------------------------------
# 2. Cargar validación
# --------------------------------------------------

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
# 3. Cargar el mejor modelo guardado
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


# --------------------------------------------------
# 4. Evaluar cada umbral
# --------------------------------------------------

results = {}

for threshold in THRESHOLDS:

    iou_scores = []
    dice_scores = []
    precision_scores = []
    recall_scores = []

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device) >= 0.5

            probabilities = torch.sigmoid(model(images))
            predictions = probabilities >= threshold

            for pred, target in zip(predictions, masks):

                pred = pred[0]
                target = target[0]

                intersection = torch.logical_and(
                    pred, target
                ).sum().item()

                union = torch.logical_or(
                    pred, target
                ).sum().item()

                pred_pixels = pred.sum().item()
                target_pixels = target.sum().item()

                iou = (
                    1.0 if union == 0
                    else intersection / union
                )

                denominator = pred_pixels + target_pixels

                dice = (
                    1.0 if denominator == 0
                    else 2.0 * intersection / denominator
                )

                precision = (
                    1.0 if target_pixels == 0
                    else 0.0
                ) if pred_pixels == 0 else (
                    intersection / pred_pixels
                )

                recall = (
                    1.0 if pred_pixels == 0
                    else 0.0
                ) if target_pixels == 0 else (
                    intersection / target_pixels
                )

                iou_scores.append(iou)
                dice_scores.append(dice)
                precision_scores.append(precision)
                recall_scores.append(recall)

    results[threshold] = {
        "iou": sum(iou_scores) / len(iou_scores),
        "dice": sum(dice_scores) / len(dice_scores),
        "precision": sum(precision_scores) / len(precision_scores),
        "recall": sum(recall_scores) / len(recall_scores),
    }


# --------------------------------------------------
# 5. Mostrar comparación
# --------------------------------------------------

print("\nResultados sobre validación")
print("Umbral | IoU    | Dice   | Precision | Recall")

for threshold, metrics in results.items():
    print(
        f"{threshold:.1f}    | "
        f"{metrics['iou']:.4f} | "
        f"{metrics['dice']:.4f} | "
        f"{metrics['precision']:.4f}    | "
        f"{metrics['recall']:.4f}"
    )

best_threshold = max(
    results,
    key=lambda t: results[t]["iou"]
)

print(
    f"\nMejor umbral por IoU promedio: "
    f"{best_threshold:.1f}"
)
