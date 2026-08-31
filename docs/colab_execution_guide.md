# Google Colab 실측 실행 가이드 (Source=Measured-GPU)

> ⚡ **바로 실행**: [`notebooks/PCAG_Measured_GPU.ipynb`](../notebooks/PCAG_Measured_GPU.ipynb)
> 를 Colab 에 업로드(또는 GitHub 에서 `Open in Colab`) → 런타임을 T4 로 설정 →
> 첫 셀부터 Run all. 아래는 그 원리와 수동 실행 절차입니다.

무료 Colab **T4 GPU(16GB VRAM)** 환경에서 이 연구의 모든 실측 데이터
(`Source=Measured-GPU`)를 추출하고, 17종 그림과 분석 산출물을 갱신하기 위한
단계별 실행 안내입니다.

> ⚠️ **세션 주의**: 무료 Colab 은 실행 시간·유휴 제한이 있어 4모델×5정밀도(20건)를
> 한 번에 돌리면 도중에 세션이 끊깁니다. **세션당 1모델**(5정밀도) 실행을 원칙으로 하고,
> 각 정밀도가 끝나는 즉시 CSV/체크포인트가 Drive 로 저장되므로 끊겨도
> `--resume` 으로 이어서 측정할 수 있습니다.

---

## 0. 사전 준비

### 0-1. 런타임 확인 (GPU 할당)

```python
# GPU 확인: 아래에 "Tesla T4" 등이 보여야 실측 가능
!nvidia-smi
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

> GPU가 없으면 `런타임 > 런타임 유형 변경 > T4 GPU` 를 선택하고 다시 시작합니다.

### 0-2. Hugging Face 토큰 (gated 모델)

`meta-llama/Llama-3-8B`, `google/gemma-2-9b` 는 라이선스 동의가 필요합니다.
각 모델 카드에서 **Accept** 후 아래 셀로 토큰을 설정합니다. (없으면
`--models Qwen/Qwen2.5-7B mistralai/Mistral-7B` 등 공개 모델만 실측해도 됩니다.)

```python
import getpass
HF_TOKEN = getpass.getpass("HF read token: ")
```

### 0-3. Google Drive 마운트 + 저장소 클론

```python
from google.colab import drive
drive.mount('/content/drive')   # 인증 팝업 허용

# (필수) 실측 결과·체크포인트 보존 디렉터리
!mkdir -p /content/drive/MyDrive/pcag_results
```

```python
# 저장소 클론 (이미 있으면 스킵)
%cd /content
if not __import__('os').path.exists('/content/llm-pcag-research'):
    !git clone https://github.com/<your-repo>/llm-pcag-research.git
%cd /content/llm-pcag-research
```

---

## 1. 패키지 설치

Colab 기본에는 `torch`(CUDA 빌드)가 이미 들어 있으므로, 실측에 필요한
나머지를 설치합니다.

```python
# bitsandbytes 버전 충돌 대비: Colab 기본 torch 와 맞춰 최신 안정판 설치
!pip install -q --upgrade transformers accelerate bitsandbytes pynvml
!pip install -q datasets scipy sympy matplotlib   # ARC-Easy 평가·분석·그림용
```

설치 확인:

```python
import torch, transformers, accelerate, bitsandbytes
from transformers import BitsAndBytesConfig
import datasets
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("bnb OK", bitsandbytes.__version__, "| datasets OK", datasets.__version__)
```

> - `bitsandbytes` 는 표준 방법 검증 세트(`--quant-method bnb`, INT8/INT4/FP4)에서만 필요합니다.
> - **기본 실측(균일 RTN PCAG 곡선)** 은 INT6/INT5/INT4/INT3/INT2 를 bitsandbytes 없이
>   `experiments/lowbit.py` 의 per-channel RTN 으로 측정합니다.
> - `datasets` 가 없으면 ARC-Easy 평가가 자동으로 내장 합성 태스크로 폴백됩니다.

---

## 2. 실측 실행 — 모델별 분할 (핵심)

### 2-1. ★ 최적 실측 프로토콜 (균일 RTN PCAG 곡선)

학술적 목표: **모든 비트폭을 한 가지 양자화 방법(per-channel RTN)으로 측정**해
방법론 혼합 없이 미세 비트 PCAG 곡선 `{16,8,6,5,4,3,2}`을 만든다.
(문헌 GPTQ/AWQ 와의 직접 비교를 피하고, 대신 "동일 방법 하에서의 붕괴 구조"를
자기-일관적으로 실측한다.)

```python
%cd /content/llm-pcag-research/experiments

!python benchmark_driver.py \
    --models meta-llama/Llama-3-8B \
    --quant-method rtn \
    --precisions FP16 INT8 INT6 INT5 INT4 INT3 INT2 \
    --eval arc_easy --eval_questions 100 \
    --repeats 3 --energy_norm \
    --update-main \
    --drive-dir /content/drive/MyDrive/pcag_results \
    --resume
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--models` | Llama-3-8B | **세션당 1모델** 지정 (타임아웃 대비) |
| `--quant-method` | `rtn` | `rtn`: lowbit 균일 RTN(PCAG 곡선) / `bnb`: bitsandbytes 표준(검증 세트) |
| `--precisions` | rtn: 7종 전체 | 원하는 정밀도만 골라 실행 가능 |
| `--eval` | `arc_easy` | 표준 0-shot 객관식(ARC-Easy). `datasets` 없으면 synthetic 폴백 |
| `--eval_questions` | 100 | 평가 질문 수 (시간 절약용) |
| `--prompts` | 20 | 지연/처리량 벤치마크 프롬프트 수 |
| `--repeats` | 3 | 정밀도당 반복 → 평균·표준편차(실측 신뢰구간) |
| `--energy_norm` | off | 측정 에너지를 **J/1,000 tokens** 로 정규화 (문헌 앵커와 단위 일치) |
| `--update-main` | off | `results_raw.csv`(Llama PCAG 기준) 갱신 — **rtn 에서만** 유효 |
| `--drive-dir` | 없음 | 매 측정 즉시 결과 CSV·체크포인트를 Drive 로 미러링 |
| `--resume` | off | 체크포인트에 완료된 (모델,정밀도) 는 건너뜀 |
| `--force` | off | `--resume` 이어도 완료분 재측정 |
| `--batch_size` | 4 | 추론 배치 chunk (T4 OOM 방지, 2로 줄이면 더 안전) |
| `--no_eval` | off | 정확도 평가 생략 → Accuracy는 **빈 값** (분석에서 자동 제외) |
| `--tdp` | 자동 | 전력 추정 폴백용 TDP(W) 오버라이드 |

### 2-2. 4개 모델 개별 실행 (권장 순서)

각 모델을 **별도 Colab 세션**에서 실행하거나, 한 세션에서 `--resume` 과 함께
순차 실행합니다. (중간에 끊겨도 다음에 `--resume` 로 이어짐)

```python
# (1) Llama-3-8B — 기본 모델 + results_raw.csv 갱신 (--update-main)
!python benchmark_driver.py --models meta-llama/Llama-3-8B \
    --quant-method rtn --eval arc_easy --eval_questions 100 \
    --repeats 3 --energy_norm --update-main \
    --drive-dir /content/drive/MyDrive/pcag_results --resume

# (2) Qwen-2.5-7B  (results_raw.csv 는 Llama 전용이므로 --update-main 없음)
!python benchmark_driver.py --models Qwen/Qwen2.5-7B \
    --quant-method rtn --eval arc_easy --eval_questions 100 \
    --repeats 3 --energy_norm \
    --drive-dir /content/drive/MyDrive/pcag_results --resume

# (3) Gemma-2-9B  (HF 토큰 필요)
import os; os.environ["HF_TOKEN"] = HF_TOKEN
!python benchmark_driver.py --models google/gemma-2-9b \
    --quant-method rtn --eval arc_easy --eval_questions 100 \
    --repeats 3 --energy_norm \
    --drive-dir /content/drive/MyDrive/pcag_results --resume

# (4) Mistral-7B
!python benchmark_driver.py --models mistralai/Mistral-7B \
    --quant-method rtn --eval arc_easy --eval_questions 100 \
    --repeats 3 --energy_norm \
    --drive-dir /content/drive/MyDrive/pcag_results --resume
```

각 실행이 끝나면 `results_multimodel_raw.csv`(4모델 × 7정밀도 균일 RTN 곡선)가
실측치로 교체·누적됩니다. Llama-3-8B 의 경우 `--update-main` 으로
`results_raw.csv`(분석 기준 파일)도 함께 갱신됩니다.

### 2-3. 표준 방법 검증 세트 (선택, 별도 CSV)

균일 RTN 곡선의 경향이 표준 양자화(bitsandbytes)와 질적으로 일치하는지 확인하는
검증 실행입니다. **반드시 별도 CSV로** 저장해 PCAG 곡선과 섞지 않습니다.

```python
# Llama-3-8B 의 INT8/INT4 를 bitsandbytes 로 측정 → results_multimodel_bnb_raw.csv
!python benchmark_driver.py --models meta-llama/Llama-3-8B \
    --quant-method bnb --precisions INT8 INT4 \
    --eval arc_easy --eval_questions 100 --repeats 3 --energy_norm \
    --drive-dir /content/drive/MyDrive/pcag_results --resume
```

### 2-4. 부분 실행 / 이어서 실행 예시

```python
# INT6 하나만 먼저 시험 (빠른 검증)
!python benchmark_driver.py --models Qwen/Qwen2.5-7B --precisions INT6 \
    --drive-dir /content/drive/MyDrive/pcag_results --resume

# 세션이 끊긴 뒤 — 남은 정밀도만 자동 이어서
!python benchmark_driver.py --models Qwen/Qwen2.5-7B \
    --drive-dir /content/drive/MyDrive/pcag_results --resume
```

> 측정 결과물:
> - `experiments/results_multimodel_raw.csv` (균일 RTN PCAG 곡선, 4모델×7정밀도)
> - `experiments/results_multimodel_bnb_raw.csv` (bnb 표준 방법 검증 세트)
> - `experiments/results_raw.csv` (Llama RTN 곡선 — 분석 기준, `--update-main` 시)
> - `<출력>.checkpoint.json` (재개용)
> → 모두 `--drive-dir` 로 지정한 Drive 경로에 동일하게 복사됩니다.

---

## 3. 세션 팅김·타임아웃 대처

### 3-1. 원칙

1. **세션당 1모델** — 20건 전체를 한 세션에 돌리지 않는다.
2. **매 정밀도 즉시 저장** — 끊겨도 진행분은 Drive 에 남는다.
3. **재개** — 새 세션에서 `--resume` 으로 미완료 건만 이어받는다.

### 3-2. 예상 소요 시간 (T4, Llama-3-8B 기준, 대략)

| Precision | 예상 소요 |
|---|---|
| FP16 | ~3분 (T4 에서 8B fp16 은 VRAM 거의 한계 — INT8 이하가 실용적) |
| INT8 | ~2.5분 |
| INT6 / INT5 | ~2분 |
| INT4 | ~2분 |
| INT3 | ~4–6분 (on-the-fly dequant 연산 오버헤드) |
| INT2 | ~4–6분 |

모델 1개(7정밀도 × 3회 반복 + ARC-Easy 100문항) ≈ **35–50분**.
무료 세션의 연속 실행 한도 내에서 무난하지만, `--repeats 1` 로 줄이면
**15–20분** 안에 완료됩니다 (표준편차 대신 평균만 기록).

### 3-3. (선택) 브라우저 Keep-alive

긴 실행 중 세션이 유휴로 판정되는 것을 늦추려면 브라우저 개발자 콘솔(F12)에서:

```js
function ClickConnect(){console.log("keep-alive");document.querySelector("colab-connect-button")?.click()}
setInterval(ClickConnect, 60000)
```

> Colab 정책이 수시로 바뀌므로, **결과 자동 저장 + 재개**를 항상 기본 방어선으로 삼으세요.

---

## 4. 실측 후 파이프라인 (17개 차트 + 최종 결과 갱신)

모든 실측이 끝나면 Drive 에서 결과를 저장소로 복사한 뒤 분석 파이프라인을
순서대로 실행합니다. `make_figures.py` 가 실측 데이터 존재를 감지해
그림 하단 출처를 `Source=Measured-GPU` 로 자동 갱신합니다.

```python
# Drive 결과를 로컬 experiments/ 로 복원 (중요: Drive 가 기록원)
!cp /content/drive/MyDrive/pcag_results/results_raw.csv experiments/ 2>/dev/null || true
!cp /content/drive/MyDrive/pcag_results/results_multimodel_raw.csv experiments/ 2>/dev/null || true
!cp /content/drive/MyDrive/pcag_results/results_multimodel_bnb_raw.csv experiments/ 2>/dev/null || true

%cd /content/llm-pcag-research/experiments

# 1) PCAG 분석 + Power Wall 판정 + Fig1/Fig2 (실측치 기준, 7비트폭 곡선)
!python analysis.py

# 2) 해석적 유도 + Jevons 기호 증명 (GPU 무관, 실측치 재피팅)
!python analytical_proof.py

# 3) 민감도·강건성 (θ 스윕 + Jevons 그리드 + Monte Carlo N=3000)
!python sensitivity.py

# 4) Jevons 매크로 시뮬레이션
!python jevons_model.py

# 5) 그림 17종 갱신 (PNG 200dpi + PDF, footnote = Source=Measured-GPU,
#    하드코딩 앵커 제거 — 벽 구간/기울기/변곡점을 실측 analysis_summary 에서 계산)
!python make_figures.py

# 6) (선택) 통합 파이프라인 검증
!python dry_run.py
```

실행 순서가 핵심입니다 — `analysis.py` → `analytical_proof.py` → `sensitivity.py`
→ `make_figures.py` 를 반드시 이 순서로 수행해야 `analysis_summary.json`,
`analysis_proof.json`, `sensitivity_summary.json` 이 최신 실측치로 동기화된 뒤
그림이 생성됩니다. (그림 fig1/fig2/fig11/fig14/fig17 은 이 JSON 들의 Power Wall
구간·기울기·변곡점을 직접 읽어 그립니다.)

생성물:

- `docs/figures/fig{1..17}_*.{png,pdf}` — 17종 그림 (실측 출처 표기)
- `experiments/analysis_summary.json` — 실측 PCAG·Power Wall 판정 (7비트폭)
- `experiments/analysis_proof.json` — 실측 재피팅 해석 모델 + 변곡점 b\*
- `experiments/sensitivity_summary.json` — θ 스윕·Jevons·Monte Carlo
- `experiments/results_raw.csv` (Llama RTN 곡선), `results_multimodel_raw.csv`
  (4모델 RTN), `results_multimodel_bnb_raw.csv` (bnb 검증) — 실측 원시 데이터

---

## 5. 데이터 무결성 주의사항

1. **Source 표기**: 실측 행은 `Source=Measured-GPU`, 문헌 앵커는
   `Source=Reference-Literature`. 같은 (모델, 정밀도) 는 실측이 문헌을 **덮어씁니다**.
2. **부분 혼합 경고**: 한 모델의 정밀도 일부만 실측하면 같은 모델 안에
   문헌·실측이 섞입니다. **한 모델은 7정밀도(FP16/INT8/INT6/INT5/INT4/INT3/INT2)
   전부를 실측**해 완전 교체하세요. 문헌 앵커와 방법·단위가 달라도 PCAG 는
   상대 비율이라 모델 내부에서 일관됩니다.
3. **방법 분리**: 균일 RTN 곡선(`results_multimodel_raw.csv`)과 bnb 검증 세트
   (`results_multimodel_bnb_raw.csv`)는 **별도 파일**로 유지하세요. 둘을 섞으면
   PCAG 곡선의 방법론 일관성이 깨집니다. `--update-main` 은 rtn 에서만 동작합니다.
4. **에너지 단위**: 실측 `Total_Energy_J` 는 "벤치마크 배치 전체 소모 에너지"이며,
   문헌 앵커의 "J/1,000 tokens"와 절대 기준이 다릅니다. **`--energy_norm` 을 켜면**
   J/1,000 tokens 로 정규화되어 문헌과 단위가 일치합니다. PCAG 는 **상대 비율** 지표이므로
   같은 모델 안에서는 정규화 여부와 무관합니다.
5. **정확도는 표준 벤치마크(ARC-Easy)**: 기본 `--eval arc_easy` 는 0-shot 객관식
   표준 벤치마크로 문헌과 비교 가능한 수준의 프록시입니다. `datasets` 미설치 시
   내장 **합성 논리 태스크**로 폴백되며, 이 경우 MMLU-style 수치와 직접 비교하면 안 됩니다.
6. **균일 RTN(모든 저비트 공통)**: `--quant-method rtn` 은 FP16 을 제외한 전 비트를
   `lowbit.py` 의 **보정 없는 per-channel RTN** 대칭 양자화로 측정합니다.
   GPTQ/AWQ 등 보정 기반 문헌 수치와 절대 정확도를 직접 비교하면 안 되며,
   대신 "동일 방법 하 미세 비트 곡선의 붕괴 구조"를 실측합니다.
   각 행 Notes 에 `quant=uniform per-channel RTN (lowbit, uncalibrated)` 가 기록됩니다.
7. **전력 계측 폴백**: pynvml/nvidia-smi 가 모두 안 되는 환경에서는
   torch.cuda 기반 **TDP 추정**(T4=70W)으로 기록되며 Notes 에
   `power=torch.cuda TDP estimate (not real power draw)` 로 명시됩니다.
8. **FP16 한계**: T4 16GB 에서 Llama-3-8B FP16(≈16GB)은 VRAM이 빠듯합니다.
   OOM 시 `--batch_size 2 --prompts 10 --max_new_tokens 64` 로 줄이거나
   INT8 을 실측 기준으로 삼는 것을 권장합니다.

---

## 6. 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| `OutOfMemoryError` | `--batch_size 2`, `--prompts 10`, `--max_new_tokens 64` 로 축소 |
| `bitsandbytes` import 실패 | `!pip install -q --upgrade bitsandbytes`, 커널 재시작 |
| INT8/INT4 가 비트 매칭 오류 | transformers·accelerate·bnb 버전을 최신으로 통일 후 재시작 |
| `pynvml` 초기화 실패 | 자동 폴백(nvidia-smi → torch.cuda TDP 추정) 적용, 무시 가능 |
| gated 모델 401 | `os.environ["HF_TOKEN"]=...` 설정 + 모델 카드에서 라이선스 동의 |
| gemma-2 로딩 느림/오류 | `attn_implementation` 충돌 시 transformers 를 최신으로 업그레이드 |
| 세션 끊김 | `--resume` + `--drive-dir` 확인 후 재실행 (진행분 보존됨) |
| 그림이 여전히 Reference 표기 | `experiments/` 에 실측 CSV 가 있는지 확인 후 `make_figures.py` 재실행 |

---

## 7. 빠른 요약 (체크리스트)

- [ ] T4 GPU 런타임 확인
- [ ] HF 토큰 설정 (Llama/Gemma 실측 시)
- [ ] Drive 마운트 + 저장소 클론
- [ ] `transformers/accelerate/bitsandbytes/pynvml/datasets` 설치
- [ ] 모델 1개씩 `benchmark_driver.py` 실행: rtn 7정밀도 × `--repeats 3` (`--drive-dir --resume`)
- [ ] (선택) bnb 표준 방법 검증 세트 별도 실행 (`--quant-method bnb`)
- [ ] Drive → `experiments/` 로 결과 복원
- [ ] `analysis.py → analytical_proof.py → sensitivity.py → jevons_model.py → make_figures.py`
- [ ] `dry_run.py` 로 산출물 검증
- [ ] `docs/figures/` 17종 + `*.json` 실측 갱신 확인