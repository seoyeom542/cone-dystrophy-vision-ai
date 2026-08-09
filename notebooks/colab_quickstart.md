# Colab 빠른 시작

Google Colab 새 노트북에서 아래 셀을 순서대로 실행하세요.
(런타임 → 런타임 유형 변경 → **GPU(T4)** 선택)

```python
# 1. 저장소 클론 & 의존성 설치
!git clone https://github.com/<YOUR_ID>/cone-dystrophy-vision-ai.git
%cd cone-dystrophy-vision-ai
!pip install -q -r requirements.txt
```

```python
# 2. Kaggle 'Retinal OCT Images' 데이터 다운로드
import kagglehub
path = kagglehub.dataset_download("paultimothymooney/kermany2018")
print("다운로드 경로:", path)

# OCT2017 폴더를 프로젝트 data/ 아래로 연결
!mkdir -p data
!ln -s {path}/OCT2017 data/OCT2017 2>/dev/null || true
!ls data/OCT2017
```

```python
# 3. 스모크 테스트 — 본 학습 전에 파이프라인이 끝까지 도는지 1분 안에 확인
#    (1 epoch · split 당 배치 3개만. 정확도 수치는 무의미하고 "에러 없이 끝나는가"만 봅니다)
!python -m src.train --smoke
```

> 여기서 터지면 **GPU 시간을 쓰기 전에** 고치세요. 데이터를 아직 안 받았다면
> 가짜 데이터로도 같은 검증이 가능합니다:
> ```
> !python scripts/make_dummy_data.py
> !python -m src.train --data-root data/dummy --smoke --no-pretrained
> ```

```python
# 4. (선택) wandb 로그인 — 실험 그래프 자동 기록
# import wandb; wandb.login()
# src/config.py 의 use_wandb=True 로 변경
```

```python
# 5. 본 학습 시작 (T4 기준 수 시간 소요)
!python -m src.train
```

주요 옵션 (config.py 를 고치지 않고 실험할 때):

| 옵션 | 설명 |
|---|---|
| `--smoke` | 1 epoch · 배치 3개로 파이프라인만 검증 |
| `--limit-batches N` | split 당 배치 N개만 사용 |
| `--epochs N` / `--batch-size N` / `--lr` | 하이퍼파라미터 |
| `--backbone resnet18` | 더 가벼운 백본으로 빠르게 실험 |
| `--val-split 0.1` | train 에서 떼어낼 검증 비율 (0 = 공식 val/ 폴더 32장 사용) |
| `--no-pretrained` | ImageNet 가중치 없이 랜덤 초기화 |

```python
# 6. 학습 결과(artifacts/best_model.pt) 로컬로 내려받기 → Streamlit 데모에 사용
from google.colab import files
files.download("artifacts/best_model.pt")
```

## 데모 실행 (로컬 PC)
```bash
pip install -r requirements.txt
# Colab에서 받은 best_model.pt 를 artifacts/ 에 넣고:
streamlit run app.py
```
