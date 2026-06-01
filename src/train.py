"""학습 진입점.

로컬:  python -m src.train
Colab: 노트북에서 `from src.train import main; main()` 또는 `!python -m src.train`

학습이 끝나면 artifacts/best_model.pt 와 클래스 라벨이 저장됩니다.
"""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from .config import cfg
from .data import build_dataloaders
from .model import build_model, set_backbone_trainable


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for images, labels in tqdm(loader, desc="train" if train else "eval", leave=False):
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
    return total_loss / total, correct / total


def main() -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Backbone: {cfg.backbone}")

    loaders = build_dataloaders(cfg.data_root, cfg.image_size, cfg.batch_size, cfg.num_workers)
    if "train" not in loaders:
        raise FileNotFoundError(
            f"학습 데이터를 찾을 수 없습니다: {cfg.data_root}/train\n"
            "Kaggle 'Retinal OCT Images'를 내려받아 data/OCT2017 에 풀어주세요."
        )

    model = build_model(cfg.backbone, cfg.num_classes, cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.project_name, config=vars(cfg))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(cfg.epochs):
        # 전이학습 워밍업: 초반 N에폭은 분류 헤드만 학습
        set_backbone_trainable(model, trainable=epoch >= cfg.freeze_backbone_epochs)

        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = (
            run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
            if "val" in loaders else (float("nan"), float("nan"))
        )
        print(f"[{epoch+1}/{cfg.epochs}] train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if cfg.use_wandb:
            import wandb
            wandb.log({"train_loss": train_loss, "train_acc": train_acc,
                       "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch})

        if not np.isnan(val_acc) and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"state_dict": model.state_dict(), "backbone": cfg.backbone,
                 "class_names": loaders["train"].dataset.classes},
                cfg.output_dir / "best_model.pt",
            )
            print(f"  ↳ best 모델 저장 (val_acc={val_acc:.4f})")

    # 테스트셋 최종 평가 리포트
    if "test" in loaders:
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for images, labels in loaders["test"]:
                preds.extend(model(images.to(device)).argmax(1).cpu().tolist())
                gts.extend(labels.tolist())
        names = loaders["test"].dataset.classes
        print("\n=== Test classification report ===")
        print(classification_report(gts, preds, target_names=names))
        print("Confusion matrix:\n", confusion_matrix(gts, preds))


if __name__ == "__main__":
    main()
