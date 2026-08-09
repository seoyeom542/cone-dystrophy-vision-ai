"""스모크 테스트용 가짜 OCT 데이터셋 생성기.

5GB짜리 Kaggle 데이터를 받기 전에 학습 파이프라인이 끝까지 도는지 확인하는 용도입니다.
실제 데이터셋과 같은 폴더 구조 + 같은 파일명 규칙(CLASS-환자ID-번호.jpeg)으로 만들어서
환자 단위 val 분리 로직까지 함께 검증됩니다.

    python scripts/make_dummy_data.py
    python -m src.train --data-root data/dummy --smoke --no-pretrained

※ 내용은 난수 노이즈라 정확도 수치는 아무 의미가 없습니다. "에러 없이 끝나는가"만 봅니다.
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def make_split(root: Path, split: str, patients_per_class: int, scans_per_patient: int, size: int, rng: random.Random) -> int:
    count = 0
    for label, cls in enumerate(CLASSES):
        out_dir = root / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in range(patients_per_class):
            # split 마다 환자 ID 대역을 다르게 줘서 실제 데이터처럼 환자가 겹치지 않게 함
            patient_id = f"{label}{'0' if split == 'train' else '9'}{p:04d}"
            for k in range(scans_per_patient):
                # 클래스마다 밝기를 살짝 다르게 → 모델이 학습할 신호가 아예 없지는 않게
                noise = rng.randrange(0, 2**31)
                arr = np.random.default_rng(noise).integers(0, 90, (size, size), dtype=np.uint8)
                arr = np.clip(arr.astype(np.int16) + label * 30, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(out_dir / f"{cls}-{patient_id}-{k}.jpeg", quality=80)  # 2D uint8 → 흑백
                count += 1
    return count


def main() -> None:
    p = argparse.ArgumentParser(description="스모크 테스트용 가짜 OCT 데이터 생성")
    p.add_argument("--out", type=Path, default=Path("data/dummy"))
    p.add_argument("--patients", type=int, default=6, help="클래스당 환자 수 (train 기준)")
    p.add_argument("--scans", type=int, default=4, help="환자당 스캔 수")
    p.add_argument("--size", type=int, default=256, help="이미지 한 변 픽셀")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    total = 0
    for split, patients in (("train", args.patients), ("test", max(2, args.patients // 3))):
        n = make_split(args.out, split, patients, args.scans, args.size, rng)
        print(f"  {split}: {n}장 ({patients}명/클래스 × {args.scans}장)")
        total += n

    print(f"\n✅ {total}장 생성 → {args.out}")
    print(f"   다음: python -m src.train --data-root {args.out} --smoke --no-pretrained")


if __name__ == "__main__":
    main()
