import os
import multiprocessing
import random
import shutil
import torch
from ultralytics import YOLO
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, Rotate

dataset_path = os.path.join(os.getcwd(), r"C:\Users\Иван\PycharmProjects\PythonProject\.venv\КУРСОВАЯ\DATASET\340401_Avtobus")
config_path = os.path.join(dataset_path, r"C:\Users\Иван\PycharmProjects\PythonProject\.venv\КУРСОВАЯ\DATASET\340401_Avtobus\data_config.yaml")

val_images_path = os.path.join(dataset_path, r"C:\Users\Иван\PycharmProjects\PythonProject\.venv\КУРСОВАЯ\DATASET\340401_Avtobus\images", r"C:\Users\Иван\PycharmProjects\PythonProject\.venv\КУРСОВАЯ\DATASET\340401_Avtobus\images\val")
train_images_path = os.path.join(dataset_path, r"C:\Users\Иван\PycharmProjects\PythonProject\.venv\КУРСОВАЯ\DATASET\340401_Avtobus\images", r"C:\Users\Иван\PycharmProjects\PythonProject\.venv\КУРСОВАЯ\DATASET\340401_Avtobus\images\train")
val_labels_path = os.path.join(dataset_path, r"C:\Users\Иван\PycharmProjects\PythonProject\.venv\КУРСОВАЯ\DATASET\340401_Avtobus\labels", r"C:\Users\Иван\PycharmProjects\PythonProject\.venv\КУРСОВАЯ\DATASET\340401_Avtobus\labels\val")
train_labels_path = os.path.join(dataset_path, r"C:\Users\Иван\PycharmProjects\PythonProject\.venv\КУРСОВАЯ\DATASET\340401_Avtobus\labels", r"C:\Users\Иван\PycharmProjects\PythonProject\.venv\КУРСОВАЯ\DATASET\340401_Avtobus\labels\train")

device1 = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8s.pt")

transform = Compose([
    RandomBrightnessContrast(p=0.2),
    HorizontalFlip(p=0.5),
    Rotate(limit=45, p=0.5)
])

def augment_image(image):
    augmented = transform(image=image)
    return augmented['image']

def main():
    results = model.train(
        data=config_path,
        epochs=1,
        batch=32,
        imgsz=640,
        device=device1,
        augment=True,
        patience=10
    )
    metrics = results.metrics

    models_dir = r"C:\Users\Иван\PycharmProjects\PythonProject\.venv\КУРСОВАЯ\MODELS"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_save_path = os.path.join(models_dir, model_name)

    model.export(format="torchscript")
    shutil.copy(r"C:\Users\Иван\PycharmProjects\PythonProject\.venv\КУРСОВАЯ\MODELS\runs\dress\train\weights\best.pt", model_save_path)

    print(f"Модель сохранена как {model_save_path}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()