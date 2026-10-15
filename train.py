
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CarDataset
from model import UNet


# --------------------------------------------------
# 1. Configuración
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
SPLIT_DIR = BASE_DIR / "Dataset" / "dataset_split"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_unet_no_aug_100.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 0.001

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Dispositivo:", device)


# --------------------------------------------------
# 2. Cargar los datasets
# --------------------------------------------------

train_dataset = CarDataset(
    images_dir=SPLIT_DIR / "train" / "images",
    masks_dir=SPLIT_DIR / "train" / "masks",
    image_size=IMAGE_SIZE,
    augment=False
)

val_dataset = CarDataset(
    images_dir=SPLIT_DIR / "val" / "images",
    masks_dir=SPLIT_DIR / "val" / "masks",
    image_size=IMAGE_SIZE
)

test_dataset = CarDataset(
    images_dir=SPLIT_DIR / "test" / "images",
    masks_dir=SPLIT_DIR / "test" / "masks",
    image_size=IMAGE_SIZE
)

print("Imágenes de entrenamiento:", len(train_dataset))
print("Imágenes de validación:", len(val_dataset))
print("Imágenes de prueba:", len(test_dataset))


# --------------------------------------------------
# 3. Crear los DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 4. Crear el modelo y configurar el aprendizaje
# --------------------------------------------------

model = UNet(
    in_channels=3,
    out_channels=1,
    base_channels=16
).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Crear la carpeta donde guardaremos el mejor modelo.
CHECKPOINT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

best_val_loss = float("inf")

print("\nComenzando el entrenamiento...")
print("Épocas:", EPOCHS)
print("Lotes por época:", len(train_loader))


# --------------------------------------------------
# 5. Bucle de entrenamiento
# --------------------------------------------------

for epoch in range(1, EPOCHS + 1):

    epoch_start = time.perf_counter()

    # Modo entrenamiento: permite actualizar el modelo.
    model.train()

    train_loss_total = 0.0
    train_samples = 0

    for batch_idx, (images, masks) in enumerate(
        train_loader, start=1
    ):
        images = images.to(device)
        masks = masks.to(device)

        # Limpiar los gradientes anteriores.
        optimizer.zero_grad(set_to_none=True)

        # Predecir las máscaras.
        outputs = model(images)

        # Comparar predicciones con las máscaras reales.
        loss = criterion(outputs, masks)

        # Calcular gradientes y actualizar parámetros.
        loss.backward()
        optimizer.step()

        # Acumular la pérdida ponderada por el tamaño del lote.
        batch_size = images.size(0)
        train_loss_total += loss.item() * batch_size
        train_samples += batch_size

        # Mostrar progreso durante cada época.
        if batch_idx % 20 == 0 or batch_idx == len(train_loader):
            print(
                f"  Época {epoch}/{EPOCHS} - "
                f"Lote {batch_idx}/{len(train_loader)}",
                flush=True
            )

    train_loss = train_loss_total / train_samples


    # --------------------------------------------------
    # 6. Validación: evaluar sin actualizar parámetros
    # --------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    val_samples = 0

    with torch.no_grad():

        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            batch_size = images.size(0)
            val_loss_total += loss.item() * batch_size
            val_samples += batch_size

    val_loss = val_loss_total / val_samples


    # --------------------------------------------------
    # 7. Guardar el mejor modelo
    # --------------------------------------------------

    is_best = val_loss < best_val_loss

    if is_best:
        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

    elapsed = time.perf_counter() - epoch_start

    print(
        f"\nÉpoca {epoch}/{EPOCHS} | "
        f"Pérdida train: {train_loss:.4f} | "
        f"Pérdida val: {val_loss:.4f} | "
        f"Tiempo: {elapsed:.1f} s"
        + (" | Mejor modelo guardado" if is_best else ""),
        flush=True
    )


# --------------------------------------------------
# 8. Resumen final
# --------------------------------------------------

print("\nEntrenamiento terminado.")
print(f"Mejor pérdida de validación: {best_val_loss:.4f}")
print(f"Modelo guardado en: {CHECKPOINT_PATH}")
