"""전이학습 모델 정의.

timm을 쓰면 ResNet/EfficientNet 등 수백 개 백본을 사전학습 가중치와 함께
한 줄로 불러올 수 있습니다. num_classes만 지정하면 분류 헤드도 자동 교체됩니다.
"""
import timm
import torch.nn as nn


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """전이학습 워밍업용: 분류 헤드를 제외한 백본을 동결/해제."""
    classifier = model.get_classifier()
    classifier_params = set(classifier.parameters())
    for param in model.parameters():
        if param not in classifier_params:
            param.requires_grad = trainable
