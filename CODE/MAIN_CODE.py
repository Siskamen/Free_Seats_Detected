import os
import multiprocessing
import random
import shutil
import torch
from ultralytics import YOLO
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, Rotate

dataset_path = os.path.join(os.getcwd(), r"Your path to dataset")
config_path = os.path.join(dataset_path, r"Your path to data_config.yaml")

val_images_path = os.path.join(dataset_path, r"Your path to val images data")
train_images_path = os.path.join(dataset_path, r"Your path to train images data")
val_labels_path = os.path.join(dataset_path, r"Your path toval labels data")
train_labels_path = os.path.join(dataset_path, r"Your path to train labels data")

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


if name == 'main':
    from multiprocessing import freeze_support
    freeze_support()  
    main()
