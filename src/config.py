"""프로젝트 전역 설정.

Colab과 로컬 양쪽에서 동작하도록 경로/하이퍼파라미터를 한 곳에 모았습니다.
값만 바꾸면 학습/데모 전체에 반영됩니다.
"""
from dataclasses import dataclass, field
from pathlib import Path


# Retinal OCT Images 데이터셋의 4개 클래스
# (맥락막신생혈관 / 당뇨황반부종 / 드루젠 / 정상)
#
# 순서 주의: ImageFolder 는 폴더명을 알파벳순으로 정렬해 라벨을 매기므로
# 여기도 반드시 알파벳순이어야 합니다. 순서가 어긋나면 예측 라벨이 통째로 밀립니다.
# (train.py 가 실제 폴더 순서와 대조해 경고를 띄웁니다.)
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


@dataclass
class Config:
    # --- 경로 ---
    # Kaggle "Retinal OCT Images" 압축을 풀면 train/ val/ test/ 하위에 클래스별 폴더가 생깁니다.
    data_root: Path = Path("data/OCT2017")
    output_dir: Path = Path("artifacts")

    # --- 모델 ---
    # timm 백본 이름. 가볍게 시작하려면 "resnet18", 성능 위주면 "efficientnet_b0".
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    num_classes: int = len(CLASS_NAMES)

    # --- 학습 ---
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2          # Colab 기본값 기준
    freeze_backbone_epochs: int = 1  # 첫 N에폭은 분류기만 학습(전이학습 워밍업)

    # --- 검증 셋 ---
    # 공식 OCT2017 val/ 은 클래스당 8장(총 32장)뿐이라 best 모델 선택 기준으로 쓰면
    # 사실상 무작위 선택이 됩니다. 기본값은 train/ 에서 환자 단위로 떼어내는 방식이고,
    # 0 으로 두면 공식 val/ 폴더를 사용합니다.
    val_split: float = 0.1

    # --- 실험 추적 ---
    use_wandb: bool = False       # wandb 로그인했다면 True
    project_name: str = "cone-dystrophy-vision-ai"

    seed: int = 42

    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))


cfg = Config()
