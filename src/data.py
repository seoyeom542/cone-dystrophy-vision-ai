"""데이터 로딩 및 augmentation.

Kaggle "Retinal OCT Images"는 ImageFolder 구조(클래스명 = 폴더명)라
torchvision.datasets.ImageFolder로 바로 읽을 수 있습니다.
"""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

# ImageNet 사전학습 가중치에 맞춘 정규화 값
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool) -> A.Compose:
    if train:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class AlbumentationsImageFolder(Dataset):
    """ImageFolder의 폴더 구조를 그대로 쓰되 albumentations 변환을 적용."""

    def __init__(self, root: Path, transform: A.Compose):
        self._base = ImageFolder(str(root))  # 라벨/클래스 매핑만 활용
        self.transform = transform
        self.classes = self._base.classes
        self.samples = self._base.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = cv2.imread(path, cv2.IMREAD_COLOR)        # OCT는 흑백이지만 백본 입력은 3채널
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=np.asarray(image))["image"]
        return image, label


def build_dataloaders(data_root: Path, image_size: int, batch_size: int, num_workers: int):
    loaders = {}
    for split in ("train", "val", "test"):
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        is_train = split == "train"
        ds = AlbumentationsImageFolder(split_dir, build_transforms(image_size, train=is_train))
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders
