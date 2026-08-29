"""학습 진입점.

로컬:  python -m src.train
Colab: 노트북에서 `from src.train import main; main()` 또는 `!python -m src.train`

본 학습 전에 파이프라인이 끝까지 도는지 먼저 확인하세요 (GPU 시간 낭비 방지):
    python -m src.train --smoke

학습이 끝나면 artifacts/best_model.pt 와 클래스 라벨이 저장됩니다.
"""
from __future__ import annotations  # 구버전 파이썬에서도 `X | None` 표기가 동작하도록

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from .config import cfg
from .data import IMAGENET_MEAN, IMAGENET_STD, build_dataloaders
from .model import build_model, set_backbone_trainable


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OCT 분류 모델 학습")
    p.add_argument("--smoke", action="store_true",
                   help="스모크 테스트: 1 epoch · 배치 3개만 돌려 파이프라인 전체를 빠르게 검증")
    p.add_argument("--limit-batches", type=int, default=None, metavar="N",
                   help="각 split 당 배치 N개만 사용 (디버그용)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--backbone", type=str, default=None, help="timm 백본 이름 (예: resnet18)")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--val-split", type=float, default=None,
                   help="train 에서 떼어낼 검증 셋 비율. 0 이면 공식 val/ 폴더 사용")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no-pretrained", action="store_true",
                   help="ImageNet 가중치 다운로드 없이 랜덤 초기화 (오프라인 스모크 테스트용)")
    return p.parse_args(argv)


def apply_overrides(args: argparse.Namespace) -> int | None:
    """CLI 인자로 cfg 를 덮어쓰고, 사용할 limit_batches 를 돌려줍니다."""
    for name in ("epochs", "batch_size", "backbone", "data_root",
                 "output_dir", "num_workers", "val_split", "lr"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.no_pretrained:
        cfg.pretrained = False

    limit_batches = args.limit_batches
    if args.smoke:
        # --epochs 를 함께 준 경우엔 사용자 지정을 존중
        if args.epochs is None:
            cfg.epochs = 1
        if limit_batches is None:
            limit_batches = 3
    return limit_batches


def build_checkpoint(model: nn.Module, class_names: list[str]) -> dict:
    """학습 가중치와 추론에 필요한 전처리 설정을 함께 저장합니다."""
    return {
        "state_dict": model.state_dict(),
        "backbone": cfg.backbone,
        "class_names": class_names,
        "image_size": cfg.image_size,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    }


def run_epoch(model, loader, criterion, optimizer, device, train: bool, limit_batches=None):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader) if limit_batches is None else min(limit_batches, len(loader))
    with torch.set_grad_enabled(train):  # 전역 상태를 남기지 않도록 context manager 사용
        for step, (images, labels) in enumerate(
            tqdm(loader, desc="train" if train else "eval", total=n_batches, leave=False)
        ):
            if limit_batches is not None and step >= limit_batches:
                break
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
    if total == 0:
        return float("nan"), float("nan")
    return total_loss / total, correct / total


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    limit_batches = apply_overrides(args)

    set_seed(cfg.seed)
    device = select_device()
    print(f"Device: {device} | Backbone: {cfg.backbone}")
    if limit_batches is not None:
        print(f"🧪 스모크 모드: {cfg.epochs} epoch · split 당 배치 {limit_batches}개 "
              f"(pretrained={cfg.pretrained}) — 지표는 무의미하고 파이프라인 동작만 확인합니다.")

    loaders = build_dataloaders(
        cfg.data_root, cfg.image_size, cfg.batch_size, cfg.num_workers,
        val_split=cfg.val_split, seed=cfg.seed,
    )
    if "train" not in loaders:
        raise FileNotFoundError(
            f"학습 데이터를 찾을 수 없습니다: {cfg.data_root}/train\n"
            "Kaggle 'Retinal OCT Images'를 내려받아 data/OCT2017 에 풀어주세요."
        )

    # 폴더 순서(=실제 라벨 순서)와 config 의 CLASS_NAMES 가 어긋나면 리포트 라벨이 통째로 밀립니다.
    class_names = loaders["train"].dataset.classes
    if len(class_names) != cfg.num_classes:
        raise ValueError(
            f"데이터 클래스 수({len(class_names)})와 모델 출력 수({cfg.num_classes})가 다릅니다: "
            f"{class_names}"
        )
    if class_names != cfg.class_names:
        print(f"⚠️  config.CLASS_NAMES {cfg.class_names} != 폴더 순서 {class_names} — 폴더 순서를 사용합니다.")

    val_source = f"train 에서 분리 ({cfg.val_split:.0%}, 환자 단위)" if cfg.val_split > 0 else "공식 val/ 폴더"
    for split in ("train", "val", "test"):
        if split in loaders:
            print(f"  {split}: {len(loaders[split].dataset):,}장" + (f"  [{val_source}]" if split == "val" else ""))

    model = build_model(cfg.backbone, cfg.num_classes, cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.project_name, config=vars(cfg))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = float("-inf")
    best_model_path = cfg.output_dir / "best_model.pt"
    best_model_saved = False

    for epoch in range(cfg.epochs):
        # 전이학습 워밍업: 초반 N에폭은 분류 헤드만 학습
        set_backbone_trainable(model, trainable=epoch >= cfg.freeze_backbone_epochs)

        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, optimizer, device, train=True, limit_batches=limit_batches)
        val_loss, val_acc = (
            run_epoch(model, loaders["val"], criterion, optimizer, device, train=False,
                      limit_batches=limit_batches)
            if "val" in loaders else (float("nan"), float("nan"))
        )
        print(f"[{epoch+1}/{cfg.epochs}] train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if cfg.use_wandb:
            import wandb
            wandb.log({"train_loss": train_loss, "train_acc": train_acc,
                       "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch})

        if not np.isnan(val_acc) and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(build_checkpoint(model, class_names), best_model_path)
            best_model_saved = True
            print(f"  ↳ best 모델 저장 (val_acc={val_acc:.4f})")

    # 테스트셋 최종 평가 리포트
    if "test" in loaders:
        if best_model_saved:
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
            print(f"\nBest 모델로 테스트 평가 (val_acc={best_val_acc:.4f})")
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for step, (images, labels) in enumerate(loaders["test"]):
                if limit_batches is not None and step >= limit_batches:
                    break
                preds.extend(model(images.to(device)).argmax(1).cpu().tolist())
                gts.extend(labels.tolist())
        names = loaders["test"].dataset.classes
        print("\n=== Test classification report ===")
        # labels 를 명시해야 일부 클래스가 등장하지 않는 경우(스모크 모드)에도 라벨이 안 밀림
        print(classification_report(gts, preds, labels=list(range(len(names))),
                                    target_names=names, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(gts, preds, labels=list(range(len(names)))))

    print("\n✅ 완료" + (" (스모크 테스트 — 실제 학습은 --smoke 없이 실행하세요)" if limit_batches else ""))


if __name__ == "__main__":
    main()
