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
# 3. (선택) wandb 로그인 — 실험 그래프 자동 기록
# import wandb; wandb.login()
# src/config.py 의 use_wandb=True 로 변경
```

```python
# 4. 학습 시작
!python -m src.train
```

```python
# 5. 학습 결과(artifacts/best_model.pt) 로컬로 내려받기 → Streamlit 데모에 사용
from google.colab import files
files.download("artifacts/best_model.pt")
```

## 데모 실행 (로컬 PC)
```bash
pip install -r requirements.txt
# Colab에서 받은 best_model.pt 를 artifacts/ 에 넣고:
streamlit run app.py
```
