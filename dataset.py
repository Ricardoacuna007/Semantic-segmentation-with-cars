
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
from torch.utils.data import Dataset
import random


class CarDataset(Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        image_size=256,
        augment=False
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.augment = augment

        # Comprobar que ambas carpetas existan.
        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"No existe la carpeta de imágenes: {self.images_dir}"
            )

        if not self.masks_dir.is_dir():
            raise FileNotFoundError(
                f"No existe la carpeta de máscaras: {self.masks_dir}"
            )

        # Buscar las imágenes JPG.
        image_paths = sorted(
            path for path in self.images_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg"}
        )

        # Buscar las máscaras PNG y asociarlas por nombre.
        mask_paths = {
            path.stem: path
            for path in self.masks_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        }

        # Verificar que todas las imágenes tengan máscara.
        image_names = {path.stem for path in image_paths}
        mask_names = set(mask_paths)

        if image_names != mask_names:
            missing_masks = sorted(image_names - mask_names)
            missing_images = sorted(mask_names - image_names)

            raise ValueError(
                f"Máscaras faltantes: {missing_masks[:10]}\n"
                f"Imágenes faltantes: {missing_images[:10]}"
            )

        # Guardar cada imagen junto a su máscara.
        self.samples = [
            (path, mask_paths[path.stem])
            for path in image_paths
        ]

    def __len__(self):
        # Cantidad de parejas imagen-máscara.
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        # Leer la imagen en RGB y la máscara en escala de grises.
        with Image.open(image_path) as file:
            image = file.convert("RGB")

        with Image.open(mask_path) as file:
            mask = file.convert("L")

        # La imagen y la máscara deben coincidir antes de transformarlas.
        if image.size != mask.size:
            raise ValueError(
                f"Dimensiones distintas en {image_path.name}: "
                f"imagen {image.size}, máscara {mask.size}"
            )

        # Adaptar ambas a un cuadrado sin deformar su proporción.
        # Los espacios añadidos serán fondo en la máscara.
        size = (self.image_size, self.image_size)

        image = ImageOps.pad(
            image,
            size,
            method=Image.Resampling.BILINEAR,
            color=(0, 0, 0),
        )

        mask = ImageOps.pad(
            mask,
            size,
            method=Image.Resampling.NEAREST,
            color=0,
        )
        
        # Aplicar transformaciones aleatorias solo al entrenamiento.
        if self.augment:

            # Voltear imagen y máscara juntas.
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                mask = ImageOps.mirror(mask)

            # Cambiar suavemente las condiciones de iluminación.
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            saturation = random.uniform(0.85, 1.15)

            image = ImageEnhance.Brightness(
                image
            ).enhance(brightness)

            image = ImageEnhance.Contrast(
                image
            ).enhance(contrast)

            image = ImageEnhance.Color(
                image
            ).enhance(saturation)


        # --------------------------------------------------
        # 1. Aumentos de datos (solo en entrenamiento)
        # --------------------------------------------------
        # Convertir la imagen a un array y normalizar sus valores a [0, 1].
        image_array = np.asarray(image, dtype=np.float32).copy()
        image_array /= 255.0

        # Convertir la imagen de [alto, ancho, canales]
        # a [canales, alto, ancho], como requiere PyTorch.
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        # Convertir la máscara a las clases binarias:
        # 0 = fondo, 1 = coche.
        mask_array = np.asarray(mask).copy()
        mask_array = (mask_array >= 128).astype(np.float32)

        # Añadir el canal de la máscara: [1, alto, ancho].
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        return image_tensor, mask_tensor
