import torch

from src.config import cfg
from src.model import build_model
from src.train import build_checkpoint, select_device


def test_checkpoint_contains_reproducible_inference_metadata():
    model = build_model("resnet18", num_classes=4, pretrained=False)
    original_backbone = cfg.backbone
    try:
        cfg.backbone = "resnet18"
        checkpoint = build_checkpoint(model, ["CNV", "DME", "DRUSEN", "NORMAL"], training_mode="smoke")
    finally:
        cfg.backbone = original_backbone

    assert checkpoint["backbone"] == "resnet18"
    assert checkpoint["image_size"] == cfg.image_size
    assert checkpoint["normalization"]["mean"] == (0.485, 0.456, 0.406)
    assert checkpoint["training_mode"] == "smoke"

    restored = build_model(checkpoint["backbone"], num_classes=4, pretrained=False)
    restored.load_state_dict(checkpoint["state_dict"])
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))


def test_select_device_returns_supported_accelerator_or_cpu():
    assert select_device().type in {"cuda", "mps", "cpu"}
