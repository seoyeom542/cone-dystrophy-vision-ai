"""Streamlit 시연 데모.

실행:  streamlit run app.py

OCT 망막 이미지를 업로드하면 (1) 4클래스 분류 확률과
(2) Grad-CAM 히트맵(모델이 어디를 보고 판단했는지)을 보여줍니다.
학습으로 만들어진 artifacts/best_model.pt 가 필요합니다.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data import build_display_transform, build_transforms
from src.model import build_model

CKPT_PATH = Path("artifacts/best_model.pt")
IMAGE_SIZE = 224


def resolve_target_layer(model: nn.Module) -> nn.Module:
    """Grad-CAM 을 걸 마지막 특징 맵 레이어를 백본에 맞춰 찾습니다.

    모듈 목록의 끝에서 세는 방식은 pooling/flatten 을 집어 히트맵이 망가집니다.
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
def load_model() -> tuple[nn.Module | None, list[str] | None]:
    if not CKPT_PATH.exists():
        return None, None
    # 체크포인트에 담긴 건 텐서/문자열뿐이라 weights_only 로드로 충분합니다.
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model = build_model(ckpt["backbone"], num_classes=len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["class_names"]


def decode_upload(uploaded) -> np.ndarray | None:
    """업로드 파일을 RGB 배열로. 디코딩 실패 시 None.

    read() 가 아니라 getvalue() 를 쓰는 이유: Streamlit 은 위젯을 조작할 때마다
    스크립트를 다시 실행하면서 같은 파일 객체를 넘겨줍니다. read() 는 커서를 소모해
    두 번째 실행부터 빈 바이트가 나옵니다.
    """
    buffer = np.frombuffer(uploaded.getvalue(), np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_uint8(image_float: np.ndarray) -> np.ndarray:
    return (np.clip(image_float, 0.0, 1.0) * 255).astype(np.uint8)


st.set_page_config(page_title="망막 OCT 분류 데모", layout="centered")
st.title("🩺 망막 OCT 이미지 분류 데모")
st.caption("EfficientNet 전이학습 모델 · Grad-CAM 설명 포함")
st.warning("연구·시연용 프로토타입입니다. 진단 목적으로 사용할 수 없습니다.", icon="⚠️")

model, class_names = load_model()
if model is None:
    st.info("학습된 모델(artifacts/best_model.pt)이 없습니다. 먼저 `python -m src.train` 으로 학습하세요.")
    st.stop()

uploaded = st.file_uploader("OCT 이미지 업로드", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.stop()

image_rgb = decode_upload(uploaded)
if image_rgb is None:
    st.error("이미지를 읽지 못했습니다. 손상되지 않은 jpg/png 파일인지 확인해주세요.")
    st.stop()

tensor = build_transforms(IMAGE_SIZE, train=False)(image=image_rgb)["image"].unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(tensor), dim=1)[0]
pred_idx = int(probs.argmax())

st.subheader(f"예측: {class_names[pred_idx]} ({probs[pred_idx] * 100:.1f}%)")
for idx in torch.argsort(probs, descending=True).tolist():
    st.progress(float(probs[idx]), text=f"{class_names[idx]} — {probs[idx] * 100:.1f}%")

st.divider()
st.subheader("Grad-CAM — 모델이 주목한 영역")
target_name = st.selectbox(
    "설명할 클래스", class_names, index=pred_idx,
    help="기본값은 예측 클래스입니다. 다른 클래스를 고르면 모델이 그 클래스의 근거를 어디서 찾는지 볼 수 있습니다.",
)

with GradCAM(model=model, target_layers=[resolve_target_layer(model)]) as cam:  # with: 등록한 hook 정리
    grayscale_cam = cam(
        input_tensor=tensor,
        targets=[ClassifierOutputTarget(class_names.index(target_name))],
    )[0]

# 히트맵은 letterbox 좌표계에서 계산되므로 배경도 같은 변환을 거쳐야 위치가 맞습니다.
vis_base = build_display_transform(IMAGE_SIZE)(image=image_rgb)["image"].astype(np.float32) / 255.0
overlay = show_cam_on_image(vis_base, grayscale_cam, use_rgb=True)

col1, col2 = st.columns(2)
col1.image(to_uint8(vis_base), caption="모델 입력 (letterbox)", use_container_width=True)
col2.image(overlay, caption=f"Grad-CAM · {target_name}", use_container_width=True)
st.caption(f"원본 {image_rgb.shape[1]}×{image_rgb.shape[0]} → 모델 입력 {IMAGE_SIZE}×{IMAGE_SIZE}")
