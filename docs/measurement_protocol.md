# 실험 프로토콜 및 데이터 출처 기록 (Measurement Protocol & Data Provenance)

본 문서는 실측 데이터(`Source=Measured-GPU`)를 얻기 위한 **실험 조건·절차·단위·검증**을
고정한다. 논문/그림의 모든 수치는 이 프로토콜 하에서 생성된 원시 데이터에서 전사한다.

> 관련 문서: 설계 결정 `docs/design_decisions.md`, 실행 가이드
> `docs/colab_execution_guide.md`, 일정 `research_journal.md`.

---

## 1. 실험 환경 (고정)

| 항목 | 값 |
|---|---|
| GPU | NVIDIA T4 (16GB VRAM, 무료 Colab) |
| 런타임 | Python 3 (Colab 기본, CUDA torch) |
| 패키지 | torch(Colab 기본), transformers, accelerate, bitsandbytes, pynvml, datasets, scipy, sympy, matplotlib |
| CUDA 메모리 | `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` |
| 측정 온도/전원 | 가상화 GPU → 실전력 계측 불가 시 **TDP 추정**(DD-08) |

환경 해시(재현용): 실측 실행 시 노트북/가이드의 `!pip install` 과 동일 버전 사용.

---

## 2. 측정 대상 (모델 × 정밀도)

### 2-1. 균일 RTN PCAG 곡선 (기본, `--quant-method rtn`)

| 모델 | 정밀도 | 방법 |
|---|---|---|
| meta-llama/Llama-3-8B | FP16, INT8, INT6, INT5, INT4, INT3, INT2 | FP16=tf16, 나머지=lowbit per-channel RTN |
| Qwen/Qwen2.5-7B | 동일 7종 | 동일 |
| google/gemma-2-9b | 동일 7종 | 동일 |
| mistralai/Mistral-7B | 동일 7종 | 동일 |

산출: `results_multimodel_raw.csv` (4모델 × 7 = 28행), Llama 는 `results_raw.csv` 에
`--update-main` 으로 복제.

### 2-2. bnb 표준 방법 검증 세트 (선택, `--quant-method bnb`)

| 모델 | 정밀도 | 방법 |
|---|---|---|
| meta-llama/Llama-3-8B | INT8, INT4 | bitsandbytes (INT8, NF4 double-quant) |

산출: `results_multimodel_bnb_raw.csv` — **곡선과 분리** (DD-02).

---

## 3. 측정 절차 (드라이버 기준)

각 (모델, 정밀도) 조합에 대해:

1. **로딩**: `load_model(method, precision)` — FP16/저비트 스트리밍 로드
   (`max_memory=총 VRAM 80%` 상한) → Linear 만 packed 저비트 변환(RTN) → GPU 이동.
   원본 fp16 해제 후 `empty_cache`.
2. **Warmup**: 1개 프롬프트, `max_new_tokens=16`.
3. **정확도 평가** (결정적, 1회): `--eval arc_easy --eval_questions 100`
   (고정 seed, test split). `--no_eval` 시 Accuracy 는 빈 값 → 분석 스킵(ISSUE-CODE-07).
4. **성능·전력 반복 측정** (`--repeats N`, 기본 3): 각 반복에서
   - 단일 `_bench()` 호출로 지연/처리량/에너지-전력 동시 측정 (ISSUE-CODE-09).
   - 전력: PyNVML → nvidia-smi → TDP 추정 폴백 체인.
   - `reset_peak_memory_stats` 후 VRAM peak 기록.
   - 반복 평균 기록, 표준편차는 Notes(`std_*`)에.
5. **단위 정규화** (`--energy_norm`): 배치 전체 J → `J/1,000 tokens`
   (= `energy × 1000 / total_new`).
6. **즉시 저장**: CSV upsert((모델,정밀도) 기준 교체) + 체크포인트 + Drive 미러링.

### 벤치마크 부하 (기본)

| 항목 | 값 |
|---|---|
| `--prompts` | 20 (한국어 지시문) |
| `--max_new_tokens` | 128 |
| `--batch_size` | 4 (chunk 단위) |
| `--repeats` | 3 |
| `--eval_questions` | 100 |

---

## 4. 컬럼 정의 (스키마: `schema.py`)

| 컬럼 | 의미 | 단위/형식 |
|---|---|---|
| Precision | 정밀도 라벨 | FP16/INT8/INT6/INT5/INT4/INT3/INT2/FP4 |
| Model_Name | HF 모델 ID | 문자열 |
| Latency_ms | 토큰당 평균 지연 | ms/token |
| Throughput_tps | 초당 생성 토큰 | tokens/s |
| Avg_Power_W | 평균 전력 | W (실측 or TDP 추정) |
| Total_Energy_J | 총 에너지 | J (배치 전체) 또는 **J/1,000 tokens**(`--energy_norm` 시) |
| Accuracy_Score | 정확도 | % (ARC-Easy 0-shot) — `--no_eval` 시 빈 값 |
| VRAM_GB | 최대 할당 VRAM | GB |
| Source | 데이터 출처 | Measured-GPU / Reference-Literature |
| Notes | 측정 조건 | 방법·평가·단위·추정 여부·std |

---

## 5. 데이터 무결성 규칙

1. **출처 라벨링**: 실측 행은 반드시 `Source=Measured-GPU`. 문헌 앵커와 동일 스키마를
   공유하되 같은 (모델, 정밀도) 에서 실측이 문헌을 **덮어쓴다**.
2. **방법 동질성**: 곡선 CSV 는 한 방법(RTN)으로만. bnb 는 별도 파일(DD-02).
3. **완전 교체 원칙**: 한 모델은 7정밀도 전부 실측 → 부분 혼합(문헌+실측) 금지.
4. **미완료 행 배제**: accuracy/energy 누락 행은 analysis/sensitivity/make_figures 에서
   자동 스킵 (ISSUE-CODE-07).
5. **단위 고정**: 에너지 단위를 Notes 에 명시. PCAG 는 상대 비율이라 단위 무관.
6. **재현**: 노트북 Run all → `--resume` 이 진행분을 이어받아 idempotent.

---

## 6. 측정 품질 (샘플 수 / 신뢰 구간)

- 정밀도당 `--repeats 3` → 지연·에너지의 표준편차를 Notes 로 보고.
- 정확도는 ARC-Easy 고정 seed 샘플(100문항)의 점 추정 — 표본 변동은 sensitivity.py 의
  Monte Carlo(σ=3%, N=3000)로 간접 평가.
- Power Wall 위치 판정은 경험적 PCHIP / Monte Carlo / 해석적 모델 **3 독립 경로**
  교차 검증 (analysis.py / sensitivity.py / analytical_proof.py).

---

## 7. 실측 후 파이프라인 (고정 순서)

```
analysis.py → analytical_proof.py → sensitivity.py → jevons_model.py → make_figures.py
```

- 각 단계가 이전 단계의 JSON 을 입력으로 사용 (analysis_summary.json →
  analysis_proof.json → sensitivity_summary.json → 그림).
- 그림은 이 JSON 들을 직접 읽음 (DD-05).
- `dry_run.py` 로 산출물 존재·일치 검증.

---

## 8. 알려진 한계 (실측 데이터 해석 시)

1. **전력 추정 가능성**: pynvml/nvidia-smi 불가 환경에서는 TDP 추정(DD-08) —
   절대 W·J 는 근사치, PCAG 상대 비율은 유효.
2. **RTN vs 보정 방법**: 절대 정확도는 GPTQ/AWQ 문헌과 직접 비교 불가.
3. **ARC-Easy 도메인**: MMLU 등 다른 벤치마크와의 절대 비교는 별도 검증 필요.
4. **FP16 한계**: T4 16GB 에서 8B FP16 은 VRAM 여유가 작음 — OOM 시
   `--batch_size 2 --prompts 10 --max_new_tokens 64` 로 축소 또는 INT8 기준.
5. **세션 제한**: 무료 Colab 세션 수명 — 세션당 1모델 + 체크포인트 재개.