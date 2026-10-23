# Car Segmentation with PyTorch

Proyecto académico de visión artificial para la **segmentación semántica de vehículos** mediante una red neuronal convolucional U-Net implementada con PyTorch.

## Objetivo

Identificar los píxeles correspondientes a vehículos en imágenes, generando máscaras binarias con dos clases:

- `0`: Fondo.
- `1`: Coche.

## Proceso de desarrollo

1. Recolección y organización de 500 imágenes de vehículos.
2. Anotación manual de los coches mediante polígonos con LabelMe.
3. Conversión de las anotaciones a máscaras PNG binarias.
4. Validación del dataset: correspondencia entre imágenes y máscaras, dimensiones, valores y visualización de contornos.
5. Preparación de imágenes y máscaras para su procesamiento con PyTorch a una resolución de 256 × 256 píxeles.
6. Desarrollo y presentación del proyecto de segmentación.

## Dataset

- **Imágenes:** 500 archivos JPG.
- **Máscaras:** 500 archivos PNG.
- **Resolución de entrada:** 256 × 256 píxeles.
- **Clases:** coche y fondo.

## Tecnologías

- Python
- PyTorch
- NumPy
- Pillow
- LabelMe
- OpenCV *(para posibles tareas de inferencia y visualización)*

## Estructura del proyecto

```text
proyect_cars/
├── checkpoints/              # Pesos de los modelos entrenados
│   ├── best_unet.pth
│   ├── best_unet_aug.pth
│   ├── best_unet_aug_100.pth
│   └── best_unet_no_aug_100.pth
├── Dataset/
│   ├── dataset_split/        # División del dataset
│   ├── images/               # Imágenes originales
│   ├── jsons/                # Anotaciones LabelMe
│   └── masks/                # Máscaras de segmentación
├── docs/
│   └── Presentacion.pptx     # Presentación del proyecto
├── predictions/
│   ├── predictions_val_aug_100_best/
│   ├── predictions_val_aug_best/
│   ├── predictions_val_best/
│   ├── predictions_val_no_aug_100_best/
│   └── previews/             # Previsualizaciones
├── scripts/
│   ├── generar_mascaras.py   # Generación de máscaras desde anotaciones
│   └── treshold_eval.py      # Evaluación de umbrales
├── dataset.py                # Carga y preparación de datos
├── eval.py                   # Evaluación del modelo
├── model.py                  # Arquitectura U-Net
├── train.py                  # Entrenamiento
├── visualize.py              # Visualización de predicciones
├── requeriments.txt          # Dependencias del proyecto
└── README.md                 # Documentación
```

Los checkpoints contienen diferentes versiones de los pesos entrenados, incluyendo experimentos con y sin aumento de datos. Las carpetas de predicciones almacenan los resultados visuales de evaluación.

## Trabajo futuro

- Evaluar el modelo con métricas como IoU y Dice.
- Mejorar la calidad de las anotaciones y la generalización.
- Experimentar con aumento de datos y diferentes arquitecturas.
- Optimizar la inferencia y explorar su ejecución en tiempo real.
- Integrar la segmentación en una aplicación.

## Estado

**Proyecto académico presentado y finalizado.** Las mejoras indicadas quedan como posibles líneas de trabajo futuro.
