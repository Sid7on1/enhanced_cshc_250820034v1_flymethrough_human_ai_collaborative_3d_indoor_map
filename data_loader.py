import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import json
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants and configuration
@dataclass
class Config:
    data_dir: str
    batch_size: int
    num_workers: int
    image_size: Tuple[int, int]
    num_classes: int

class DataLoaderConfig(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

class DataFormatException(Exception):
    pass

class DataLoadingError(Exception):
    pass

class DataBatchingError(Exception):
    pass

class DataLoader(Dataset):
    def __init__(self, config: Config, data_format: DataLoaderConfig):
        self.config = config
        self.data_format = data_format
        self.data_dir = config.data_dir
        self.image_size = config.image_size
        self.num_classes = config.num_classes
        self.image_paths = self._load_image_paths()
        self.labels = self._load_labels()

    def _load_image_paths(self) -> List[str]:
        image_paths = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _load_labels(self) -> List[int]:
        labels = []
        with open(os.path.join(self.data_dir, "labels.json"), "r") as f:
            label_data = json.load(f)
            for image_path in self.image_paths:
                label = label_data[image_path]
                labels.append(label)
        return labels

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        return image

    def _load_image(self, image_path: str) -> np.ndarray:
        try:
            image = self._preprocess_image(image_path)
            return image
        except Exception as e:
            logger.error(f"Error loading image: {image_path}")
            raise DataLoadingError(f"Error loading image: {image_path}") from e

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        image_path = self.image_paths[index]
        image = self._load_image(image_path)
        label = self.labels[index]
        return image, label

    def __len__(self) -> int:
        return len(self.image_paths)

class BatchLoader:
    def __init__(self, data_loader: DataLoader, batch_size: int):
        self.data_loader = data_loader
        self.batch_size = batch_size

    def _batch_images(self, images: List[np.ndarray]) -> np.ndarray:
        batch_size = self.batch_size
        num_images = len(images)
        num_batches = num_images // batch_size
        batch_images = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch = images[start:end]
            batch_images.append(np.stack(batch, axis=0))
        return np.stack(batch_images, axis=0)

    def _batch_labels(self, labels: List[int]) -> np.ndarray:
        batch_size = self.batch_size
        num_labels = len(labels)
        num_batches = num_labels // batch_size
        batch_labels = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch = labels[start:end]
            batch_labels.append(np.array(batch))
        return np.array(batch_labels)

    def _batch_data(self, data: List[Tuple[np.ndarray, int]]) -> Tuple[np.ndarray, np.ndarray]:
        images, labels = zip(*data)
        batch_images = self._batch_images(images)
        batch_labels = self._batch_labels(labels)
        return batch_images, batch_labels

    def __iter__(self):
        data = []
        for image, label in self.data_loader:
            data.append((image, label))
            if len(data) == self.batch_size:
                yield self._batch_data(data)
                data = []
        if data:
            yield self._batch_data(data)

def create_data_loader(config: Config, data_format: DataLoaderConfig) -> DataLoader:
    data_loader = DataLoader(config, data_format)
    return data_loader

def create_batch_loader(data_loader: DataLoader, batch_size: int) -> BatchLoader:
    batch_loader = BatchLoader(data_loader, batch_size)
    return batch_loader

def main():
    config = Config(
        data_dir="/path/to/data",
        batch_size=32,
        num_workers=4,
        image_size=(224, 224),
        num_classes=10,
    )
    data_format = DataLoaderConfig.TRAIN
    data_loader = create_data_loader(config, data_format)
    batch_loader = create_batch_loader(data_loader, config.batch_size)
    for batch in batch_loader:
        images, labels = batch
        print(images.shape, labels.shape)

if __name__ == "__main__":
    main()