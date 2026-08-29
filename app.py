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
from src.explain import CLASS_INFO, assess_confidence, class_label
from src.model import build_model

CKPT_PATH = Path("artifacts/best_model.pt")
DEFAULT_IMAGE_SIZE = 224


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
def load_model() -> tuple[nn.Module | None, list[str] | None, dict]:
    if not CKPT_PATH.exists():
        return None, None, {"image_size": DEFAULT_IMAGE_SIZE}
    # 체크포인트에 담긴 건 텐서/문자열뿐이라 weights_only 로드로 충분합니다.
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model = build_model(ckpt["backbone"], num_classes=len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    metadata = {
        "image_size": ckpt.get("image_size", DEFAULT_IMAGE_SIZE),
        "backbone": ckpt["backbone"],
        "is_smoke": ckpt.get("training_mode") == "smoke" or "smoke" in str(CKPT_PATH.resolve()).lower(),
    }
    return model, ckpt["class_names"], metadata


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


st.set_page_config(page_title="망막 OCT AI 분석", page_icon="👁️", layout="wide")
st.title("망막 OCT AI 분석")
st.caption("OCT 이미지를 네 가지 범주로 분류하고, 모델이 참고한 영역을 시각화합니다.")

model, class_names, model_info = load_model()
if model is None:
    st.info("학습된 모델(artifacts/best_model.pt)이 없습니다. 먼저 `python -m src.train` 으로 학습하세요.")
    st.stop()
image_size = model_info["image_size"]

with st.sidebar:
    st.header("이용 안내")
    st.markdown("1. OCT 이미지 한 장을 업로드합니다.\n2. 결과 요약과 후보별 확률을 확인합니다.\n3. Grad-CAM에서 모델의 주목 영역을 확인합니다.")
    st.divider()
    st.subheader("모델 정보")
    st.write(f"백본: `{model_info['backbone']}`")
    st.write(f"입력 크기: `{image_size} × {image_size}`")
    st.caption("지원 형식: JPG, JPEG, PNG")

st.warning("연구·교육용 프로토타입입니다. 이 결과만으로 질환을 진단하거나 치료를 결정하지 마세요.", icon="⚠️")
if model_info["is_smoke"]:
    st.error(
        "현재는 파이프라인 확인용 스모크 모델입니다. 일부 배치만 학습했으므로 예측 결과와 확률은 성능 평가에 사용할 수 없습니다.",
        icon="🧪",
    )

uploaded = st.file_uploader(
    "분석할 OCT 이미지",
    type=["jpg", "jpeg", "png"],
    help="개인정보가 포함되지 않은 망막 OCT 단면 이미지를 한 장 선택하세요.",
)
if not uploaded:
    st.info("이미지를 업로드하면 분석 결과가 이곳에 표시됩니다.")
    st.stop()

image_rgb = decode_upload(uploaded)
if image_rgb is None:
    st.error("이미지를 읽지 못했습니다. 손상되지 않은 jpg/png 파일인지 확인해주세요.")
    st.stop()

tensor = build_transforms(image_size, train=False)(image=image_rgb)["image"].unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(tensor), dim=1)[0]
pred_idx = int(probs.argmax())
prob_values = [float(value) for value in probs]
ranked_indices = torch.argsort(probs, descending=True).tolist()
pred_name = class_names[pred_idx]
confidence_label, confidence_level, confidence_message = assess_confidence(prob_values)

summary_tab, cam_tab, guide_tab = st.tabs(["결과 요약", "주목 영역(Grad-CAM)", "용어와 주의사항"])

with summary_tab:
    st.subheader("분석 결과")
    image_col, result_col = st.columns([1, 1.35], gap="large")
    with image_col:
        st.image(image_rgb, caption="업로드한 원본 OCT", use_container_width=True)
    with result_col:
        with st.container(border=True):
            st.caption("가장 높은 모델 후보")
            st.markdown(f"### {class_label(pred_name)}")
            metric_col, confidence_col = st.columns(2)
            metric_col.metric("모델 출력 확률", f"{prob_values[pred_idx] * 100:.1f}%")
            confidence_col.metric("결과 해석", confidence_label)
            info = CLASS_INFO.get(pred_name)
            if info:
                st.write(info["summary"])
        getattr(st, confidence_level)(confidence_message)

    st.subheader("전체 후보 비교")
    st.caption("네 확률의 합은 100%입니다. 수치가 서로 비슷하면 모델이 한 범주를 구분하지 못한 상태입니다.")
    for rank, idx in enumerate(ranked_indices, start=1):
        st.progress(prob_values[idx], text=f"{rank}위  {class_label(class_names[idx])} — {prob_values[idx] * 100:.1f}%")

    with st.expander("확률을 어떻게 읽어야 하나요?"):
        st.markdown(
            "- 이 값은 입력 영상을 네 범주 중 어디에 가깝게 보는지 나타내는 **모델 내부 점수**입니다.\n"
            "- 실제 질환 보유 가능성이나 진단 정확도를 직접 의미하지 않습니다.\n"
            "- 1위와 2위가 비슷하면 단일 결과보다 `판단 보류`로 해석하는 것이 안전합니다.\n"
            "- 촬영 장비, 영상 품질, 학습 데이터와의 차이에 따라 결과가 달라질 수 있습니다."
        )

with cam_tab:
    st.subheader("모델이 판단에 참고한 영역")
    st.write("Grad-CAM은 선택한 클래스 점수를 높이는 데 상대적으로 영향을 준 위치를 색으로 표시합니다.")
    target_name = st.selectbox(
        "확인할 클래스",
        class_names,
        index=pred_idx,
        format_func=class_label,
        help="기본값은 가장 높은 후보입니다. 다른 클래스를 선택하면 해당 후보에 대한 주목 영역을 볼 수 있습니다.",
    )

    with GradCAM(model=model, target_layers=[resolve_target_layer(model)]) as cam:
        grayscale_cam = cam(
            input_tensor=tensor,
            targets=[ClassifierOutputTarget(class_names.index(target_name))],
        )[0]

    vis_base = build_display_transform(image_size)(image=image_rgb)["image"].astype(np.float32) / 255.0
    overlay = show_cam_on_image(vis_base, grayscale_cam, use_rgb=True)

    col1, col2 = st.columns(2)
    col1.image(to_uint8(vis_base), caption="모델에 입력된 이미지", use_container_width=True)
    col2.image(overlay, caption=f"주목 영역 · {class_label(target_name)}", use_container_width=True)
    st.caption(f"원본 {image_rgb.shape[1]}×{image_rgb.shape[0]} → 모델 입력 {image_size}×{image_size} (비율 유지 후 여백 추가)")
    st.info("색상 해석: 빨강·노랑은 상대적으로 영향이 큰 영역, 파랑은 영향이 작은 영역입니다. 이는 병변 경계나 질환 위치를 직접 표시하는 검출 결과가 아닙니다.")

with guide_tab:
    st.subheader("분류 범주")
    for name in class_names:
        info = CLASS_INFO.get(name)
        if info:
            with st.expander(class_label(name)):
                st.write(info["summary"])

    st.subheader("사용 시 주의사항")
    st.markdown(
        "- 본 도구는 연구·교육 목적의 프로토타입이며 의료기기가 아닙니다.\n"
        "- 낮은 품질, 잘못된 방향, 다른 촬영 방식의 이미지는 신뢰할 수 없는 결과를 만들 수 있습니다.\n"
        "- Grad-CAM은 모델의 관심을 설명하는 보조 자료이며 병변 분할이나 원인 증명이 아닙니다.\n"
        "- 건강 관련 판단은 안과 전문의의 진료와 정식 검사 결과를 따라야 합니다."
    )
