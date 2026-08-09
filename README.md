# cone-dystrophy-vision-ai

망막 OCT 이미지를 AI로 분류하는 전이학습 프로젝트.
추체이영양증(Cone dystrophy) 연구로 확장하기 위한 베이스라인입니다.

## 컨셉
- **목표**: 망막 OCT 영상을 정상/질환으로 분류하는 딥러닝 모델
- **데이터**: Kaggle [Retinal OCT Images (Kermany 2018)](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) — `NORMAL / CNV / DME / DRUSEN` 4클래스
- **방법**: ImageNet 사전학습 백본(EfficientNet/ResNet)을 전이학습
- **확장**: 모델 구조 검증 후 세브란스 코호트 등 추체이영양증 데이터로 확장

## 기술 스택
| 레이어 | 도구 |
|--------|------|
| 언어 | Python 3.11+ |
| 딥러닝 | PyTorch |
| 전이학습 백본 | timm (EfficientNet-B0 기본) |
| 전처리/증강 | albumentations |
| 모델 해석 | Grad-CAM (grad-cam) |
| 실험 추적 | Weights & Biases (선택) |
| 학습 환경 | Google Colab (무료 GPU) |
| 시연 데모 | Streamlit |

## 프로젝트 구조
```
src/
  config.py   # 경로·하이퍼파라미터 한 곳에 모음
  data.py     # 데이터 로딩 + albumentations 증강
  model.py    # timm 전이학습 모델
  train.py    # 학습 루프 + 평가 리포트
app.py        # Streamlit 데모 (분류 + Grad-CAM)
scripts/make_dummy_data.py      # 스모크 테스트용 가짜 데이터 생성
notebooks/colab_quickstart.md   # Colab 실행 가이드
```

## 빠른 시작
1. **스모크 테스트** — 데이터 없이 파이프라인만 1분 안에 검증:
   ```bash
   pip install -r requirements.txt
   python scripts/make_dummy_data.py
   python -m src.train --data-root data/dummy --smoke --no-pretrained
   ```
2. **학습 (Colab)**: [notebooks/colab_quickstart.md](notebooks/colab_quickstart.md) 참고.
   본 학습 전에 `python -m src.train --smoke` 로 한 번 확인하세요.
3. **데모 (로컬)**:
   ```bash
   streamlit run app.py   # artifacts/best_model.pt 필요
   ```

## 참고: 검증 셋
공식 OCT2017 `val/` 은 클래스당 8장(총 32장)뿐이라 best 모델 선택 기준으로 쓰면
사실상 무작위 선택이 됩니다. 기본값(`cfg.val_split=0.1`)은 `train/` 에서 **환자 단위**로
검증 셋을 떼어냅니다 — 같은 환자의 스캔이 train/val 에 섞이면 val 정확도가 부풀려지기 때문입니다.

## 다음 단계
- [ ] 베이스라인 학습 → 테스트 정확도 확인
- [ ] Grad-CAM으로 모델 주목 영역 검증
- [ ] 클래스 불균형 처리 / 추가 증강 실험
- [ ] 추체이영양증 데이터로 fine-tuning
