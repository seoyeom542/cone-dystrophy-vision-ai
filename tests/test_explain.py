import pytest

from src.explain import assess_confidence, class_label


def test_class_label_includes_plain_language_name():
    assert class_label("CNV") == "CNV · 맥락막 신생혈관"
    assert class_label("UNKNOWN") == "UNKNOWN"


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ([0.26, 0.251, 0.245, 0.244], "판단 보류"),
        ([0.52, 0.31, 0.10, 0.07], "낮은 확신"),
        ([0.80, 0.10, 0.06, 0.04], "상대적으로 높은 확신"),
    ],
)
def test_confidence_assessment(probabilities, expected):
    label, level, message = assess_confidence(probabilities)
    assert label == expected
    assert level in {"warning", "success"}
    assert message


def test_confidence_assessment_rejects_empty_values():
    with pytest.raises(ValueError, match="비어"):
        assess_confidence([])
