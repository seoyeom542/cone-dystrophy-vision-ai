"""Streamlit 시연 데모.

실행:  streamlit run app.py

OCT 망막 이미지를 업로드하면 (1) 4클래스 분류 확률과
(2) Grad-CAM 히트맵(모델이 어디를 보고 판단했는지)을 보여줍니다.
학습으로 만들어진 artifacts/best_model.pt 가 필요합니다.
"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data import build_transforms
from src.model import build_model

CKPT_PATH = Path("artifacts/best_model.pt")
IMAGE_SIZE = 224


def resolve_target_layer(model: nn.Module) -> nn.Module:
    """Grad-CAM 을 걸 마지막 특징 맵 레이어를 백본에 맞춰 찾습니다.

    모듈 목록의 끝에서 세는 방식은 pooling/flatten 을 집어버려서 히트맵이 망가집니다.
    """
    if hasattr(model, "conv_head"):        # timm EfficientNet / MobileNet 계열
        return model.conv_head
    if hasattr(model, "layer4"):           # ResNet 계열: 마지막 residual block
        return model.layer4[-1]
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("Grad-CAM 대상 conv 레이어를 찾지 못했습니다.")
    return convs[-1]


@st.cache_resource
def load_model():
    if not CKPT_PATH.exists():
        return None, None
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = build_model(ckpt["backbone"], num_classes=len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["class_names"]


def preprocess(image_rgb: np.ndarray) -> torch.Tensor:
    tf = build_transforms(IMAGE_SIZE, train=False)
    return tf(image=image_rgb)["image"].unsqueeze(0)


st.set_page_config(page_title="망막 OCT 분류 데모", layout="centered")
st.title("🩺 망막 OCT 이미지 분류 데모")
st.caption("EfficientNet 전이학습 모델 · Grad-CAM 설명 포함")

model, class_names = load_model()
if model is None:
    st.warning("학습된 모델(artifacts/best_model.pt)이 없습니다. 먼저 `python -m src.train` 으로 학습하세요.")
    st.stop()

uploaded = st.file_uploader("OCT 이미지 업로드", type=["jpg", "jpeg", "png"])
if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    tensor = preprocess(image_rgb)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    pred_idx = int(probs.argmax())

    st.subheader(f"예측: **{class_names[pred_idx]}** ({probs[pred_idx]*100:.1f}%)")
    st.bar_chart({class_names[i]: float(probs[i]) for i in range(len(class_names))})

    # Grad-CAM: 마지막 합성곱 특징 맵 기준으로 예측 클래스에 대한 히트맵
    cam = GradCAM(model=model, target_layers=[resolve_target_layer(model)])
    grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]
    vis_base = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
    overlay = show_cam_on_image(vis_base, grayscale_cam, use_rgb=True)

    col1, col2 = st.columns(2)
    col1.image(image_rgb, caption="원본", use_container_width=True)
    col2.image(overlay, caption="Grad-CAM (모델 주목 영역)", use_container_width=True)
