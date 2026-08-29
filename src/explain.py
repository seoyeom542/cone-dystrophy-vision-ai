"""데모 화면에서 사용하는 클래스 설명과 예측 확신도 해석."""

CLASS_INFO = {
    "CNV": {
        "ko": "맥락막 신생혈관",
        "summary": "망막 아래에 비정상 혈관이 자라 삼출이나 출혈을 일으킬 수 있는 소견입니다.",
    },
    "DME": {
        "ko": "당뇨 황반부종",
        "summary": "당뇨병과 관련해 황반에 액체가 고이고 망막이 두꺼워질 수 있는 소견입니다.",
    },
    "DRUSEN": {
        "ko": "드루젠",
        "summary": "망막 아래에 노폐물이 쌓여 작은 침착물로 보이는 소견입니다.",
    },
    "NORMAL": {
        "ko": "정상",
        "summary": "이 데이터셋의 질환 범주에 해당하는 특징이 뚜렷하지 않은 OCT 영상입니다.",
    },
}


def class_label(class_name: str) -> str:
    info = CLASS_INFO.get(class_name)
    return f"{class_name} · {info['ko']}" if info else class_name


def assess_confidence(probabilities: list[float]) -> tuple[str, str, str]:
    """최고 확률과 1·2위 격차로 사용자용 해석을 반환합니다.

    이는 모델 확률의 임상적 보정을 대신하지 않으며 UI에서 과신을 막기 위한 휴리스틱입니다.
    """
    if not probabilities:
        raise ValueError("확률 목록이 비어 있습니다.")
    ranked = sorted(probabilities, reverse=True)
    top = ranked[0]
    margin = top - ranked[1] if len(ranked) > 1 else top
    if top < 0.4 or margin < 0.1:
        return (
            "판단 보류",
            "warning",
            "상위 후보들의 확률이 비슷해 한 클래스를 신뢰하기 어렵습니다. 다른 영상 또는 학습된 모델로 다시 확인하세요.",
        )
    if top < 0.65:
        return (
            "낮은 확신",
            "warning",
            "가장 높은 후보가 뚜렷하지 않습니다. 결과를 참고용으로만 보고 전문가 검토가 필요합니다.",
        )
    return (
        "상대적으로 높은 확신",
        "success",
        "모델 내부에서는 한 후보가 비교적 뚜렷합니다. 다만 확률은 진단 정확도나 실제 질환 가능성을 뜻하지 않습니다.",
    )
