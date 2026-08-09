"""데이터 로딩 및 augmentation.

Kaggle "Retinal OCT Images"는 ImageFolder 구조(클래스명 = 폴더명)라
torchvision.datasets.ImageFolder로 (경로, 라벨) 목록을 바로 얻을 수 있습니다.

주의: 공식 val/ 폴더는 클래스당 8장(총 32장)뿐이라 epoch마다의 val_acc가
거의 노이즈입니다. 그래서 기본값은 train/ 에서 검증 셋을 떼어내는 방식입니다.
"""
import random
import re
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

# ImageNet 사전학습 가중치에 맞춘 정규화 값
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Kermany 데이터셋 파일명 규칙: "CNV-9911627-1.jpeg" → 가운데 숫자가 환자 ID.
# 한 환자의 여러 스캔이 train/val 양쪽에 흩어지면 val 정확도가 과대평가되므로
# 검증 셋은 이미지가 아니라 환자 단위로 분리합니다.
_PATIENT_ID_RE = re.compile(r"^[A-Za-z]+-(\d+)-\d+")


def _patient_id(path: str) -> str:
    match = _PATIENT_ID_RE.match(Path(path).name)
    return match.group(1) if match else Path(path).name  # 규칙에 안 맞으면 이미지 단위로 취급


def geometry_ops(image_size: int) -> list:
    """모델 입력 기하 변환 — 종횡비를 유지한 채 정사각형으로 여백을 채웁니다(letterbox).

    OCT 원본은 512×496, 768×496 처럼 정사각형이 아니라서 단순 resize 와 결과가 다릅니다.
    Grad-CAM 히트맵은 이 변환을 거친 좌표계 위에서 계산되므로, 시각화 배경도
    반드시 같은 변환을 써야 히트맵 위치가 어긋나지 않습니다. 그래서 한 곳에 모아둡니다.
    """
    return [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT),
    ]


def build_transforms(image_size: int, train: bool) -> A.Compose:
    if train:
        return A.Compose([
            *geometry_ops(image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            # ShiftScaleRotate 는 albumentations 2.x 에서 deprecated — Affine 이 같은 역할
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-10, 10),
                     border_mode=cv2.BORDER_CONSTANT, p=0.4),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        *geometry_ops(image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_display_transform(image_size: int) -> A.Compose:
    """Grad-CAM 오버레이 배경용 — 정규화/텐서화 없이 모델 입력과 동일한 기하 변환만 적용."""
    return A.Compose(geometry_ops(image_size))


class AlbumentationsImageFolder(Dataset):
    """ImageFolder가 만든 (경로, 라벨) 목록에 albumentations 변환을 적용."""

    def __init__(self, samples: list[tuple[str, int]], classes: list[str], transform: A.Compose):
        self.samples = samples
        self.classes = classes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = cv2.imread(path, cv2.IMREAD_COLOR)  # OCT는 흑백이지만 백본 입력은 3채널
        if image is None:
            raise FileNotFoundError(f"이미지를 읽지 못했습니다: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image=image)["image"], label


def split_by_patient(
    samples: list[tuple[str, int]], val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """클래스별로 환자 그룹을 섞어 val_ratio 만큼을 검증 셋 인덱스로 떼어냅니다.

    같은 환자의 스캔은 train/val 중 한쪽에만 들어갑니다.
    """
    patients: dict[tuple[int, str], list[int]] = defaultdict(list)
    for idx, (path, label) in enumerate(samples):
        patients[(label, _patient_id(path))].append(idx)

    by_label: dict[int, list[list[int]]] = defaultdict(list)
    for (label, _), indices in patients.items():
        by_label[label].append(indices)

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for label in sorted(by_label):
        groups = sorted(by_label[label])  # 순서를 고정한 뒤 셔플해야 seed로 재현됨
        rng.shuffle(groups)
        # 클래스마다 최소 1명은 val 로, 단 train 이 비지 않도록 상한을 둠
        n_val = min(max(1, round(len(groups) * val_ratio)), max(len(groups) - 1, 0))
        for group in groups[:n_val]:
            val_idx.extend(group)
        for group in groups[n_val:]:
            train_idx.extend(group)
    return sorted(train_idx), sorted(val_idx)


def _make_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # CPU/MPS 에서는 지원되지 않아 경고만 남김
    )


def build_dataloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.1,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """train/val/test 로더를 만듭니다.

    val_split > 0 이면 train/ 에서 환자 단위로 검증 셋을 떼어내고,
    0 이면 공식 val/ 폴더(32장)를 그대로 씁니다.
    """
    loaders: dict[str, DataLoader] = {}
    train_dir = data_root / "train"

    if train_dir.exists():
        base = ImageFolder(str(train_dir))  # 라벨/클래스 매핑만 활용
        classes = base.classes
        train_samples = base.samples

        if val_split > 0:
            train_idx, val_idx = split_by_patient(base.samples, val_split, seed)
            train_samples = [base.samples[i] for i in train_idx]
            val_samples = [base.samples[i] for i in val_idx]
            loaders["val"] = _make_loader(
                AlbumentationsImageFolder(val_samples, classes, build_transforms(image_size, train=False)),
                batch_size, num_workers, shuffle=False,
            )

        loaders["train"] = _make_loader(
            AlbumentationsImageFolder(train_samples, classes, build_transforms(image_size, train=True)),
            batch_size, num_workers, shuffle=True,
        )

    for split in ("val", "test"):
        if split in loaders:  # train 에서 떼어낸 val 이 공식 val/ 보다 우선
            continue
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        base = ImageFolder(str(split_dir))
        loaders[split] = _make_loader(
            AlbumentationsImageFolder(base.samples, base.classes, build_transforms(image_size, train=False)),
            batch_size, num_workers, shuffle=False,
        )

    return loaders
