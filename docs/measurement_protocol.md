# GPU 실측 측정 프로토콜 (Measurement Protocol)

> 본 문서는 참고 데이터(Reference-Literature) 단계를 **실측(Measured-GPU) 단계로 전환**할 때
> 적용하는 표준 측정 프로토콜이다. `docs/main_paper.md` 3.1~3.2절 및 `docs/reproducibility.md` 와 함께 읽는다.
> 목적: 동일 스키마에서 재현 가능하고, 학술적으로 방어 가능한 실측 수치를 생산한다.

---

## 1. 목적 및 범위

- FP16 / INT8 / INT4 (및 탐색적 INT3/INT2) 가중치 양자화 모델의 **추론 에너지·전력·성능·정확도**를 실측한다.
- 산출물은 `experiments/results_raw.csv` (주 모델)와 `experiments/results_multimodel_raw.csv` (다중 모델)를
  `Source=Measured-GPU` 로 채운다.
- 이 데이터로 `analysis.py → sensitivity.py → make_figures.py → verify_numbers.py` 전 단계를 재실행해
  논문의 모든 수치·그림을 실측치로 갱신한다.

## 2. 환경 및 하드웨어 요구사항

| 항목 | 요구사항 |
|---|---|
| GPU | 1개 이상, VRAM ≥ 16 GB (Llama-3-8B FP16 ≈ 15.8 GB 상주) |
| 드라이버/도구 | `nvidia-smi` 또는 PyNVML (전력 계측), CUDA |
| Python | ≥ 3.10, venv |
| 의존성 | `requirements.txt` + 실측 전용: `torch`, `transformers`, `bitsandbytes`, `pynvml` |
| 계측 주기 | 기본 50 ms 샘플링 (`telemetry.PowerMeter(sample_interval=0.05)`) |

**환경 기록 필수**: GPU 모델/수량, 드라이버 버전, CUDA 버전, torch/transformers/bitsandbytes 버전,
클럭·전압 상태(고정 여부), 주변 온도를 `research_journal.md` 와 CSV `Notes` 에 남긴다.

## 3. 모델 × 정밀도 행렬

| 모델 | 파라미터 | FP16 | INT8 | INT4 | INT3 | INT2 |
|---|---|---|---|---|---|---|
| Llama-3-8B (주) | 8B | ✓ | ✓ | ✓ | (✓) | (✓) |
| Qwen-2.5-7B | 7B | ✓ | ✓ | ✓ | (✓) | (✓) |
| Gemma-2-9B | 9B | ✓ | ✓ | ✓ | (✓) | (✓) |
| Mistral-7B | 7B | ✓ | ✓ | ✓ | (✓) | (✓) |

- `✓` = 표준 실행, `(✓)` = 탐색적(가능 시). INT3/INT2는 bitsandbytes가 지원하지 않으므로
  `experiments/lowbit.py` 의 packed 정수 + 채널별 scale 방식으로 실측한다.
- 모든 정밀도는 **동일 프롬프트·동일 배치·동일 max_new_tokens** 로 측정한다.

## 4. 계측 방법론

### 4.1 전력·에너지 (telemetry.py)
- **1차**: PyNVML `nvmlDeviceGetPowerUsage` (mW)를 50 ms 주기로 샘플링.
- **폴백**: `nvidia-smi --query-gpu=power.draw` 파싱.
- **에너지**: 샘플 시계열을 **사다리꼴 적분**하여 누적 에너지(J) 산출:
  `E = Σ (w[i-1]+w[i])/2 · (t[i]-t[i-1])`.
- 계측 불가 시 `Total_Energy_J = avg_power × elapsed` 폴백을 쓰고 `Notes` 에 **추정값**임을 명시.
- **측정 오버헤드 분리**: `benchmark_driver.py` 는 계측 하의 1회 실행과 계측 없는 성능 실행을
  분리 호출하여 계측 쓰레드가 지연/처리량 측정을 오염시키지 않도록 한다.

### 4.2 에너지 정규화
- 토큰당 에너지 **E_tok = E_total / new_tokens**, 천 토큰당 **E = E_tok × 1000** (스키마 `Total_Energy_J` 의 의미).
- 배치 규모를 고정하고 **배치 스케일 정합 규칙** `E ≈ BATCH_AMORT × P × latency` 를 적용해
  서로 다른 배치/모델 간 에너지가 일관되게 스케일링되는지 교차검증한다 (`multimodel_data.py` 참조).

### 4.3 지연·처리량·VRAM
- 지연: `latency_ms_per_token = elapsed_ms / new_tokens`.
- 처리량: `throughput_tps = new_tokens / elapsed_s`.
- VRAM: `torch.cuda.max_memory_allocated()/1024^3` (GB), 또는 `nvidia-smi` 최대 사용량.
- **워밍업(warmup)** 후 측정해 초기 할당·캐시 영향 제거.

## 5. 정확도 평가
- **우선순위 1**: `experiments/datasets/` 에 MMLU(dev) 또는 GSM8K 형태 로드 시 실제 벤치마크.
- **우선순위 2**: 부재 시 `eval_harness.SyntheticLogicTask` (합성 논리 태스크, seed 고정)로 proxy 정확도.
- 평가는 모든 정밀도에 **동일 하네스·동일 시드·동일 질문 수** 적용.

## 6. 반복 측정 및 통계 보고
- 각 (모델, 정밀도) 셀은 **최소 3회 반복** 측정하고 평균 ± 표준편차를 기록.
- 논문/부록에는 n(반복 수), 평균, 95% CI(가능 시)를 명시.
- 이상치(outlier)는 원시값과 함께 아카이브하고, 제외 시 근거를 기록.

## 7. 데이터 스키마 및 출처 라벨링
- 스키마: `experiments/schema.py` 의 `RESULTS_COLUMNS`
  (`Precision, Model_Name, Latency_ms, Throughput_tps, Avg_Power_W, Total_Energy_J, Accuracy_Score, VRAM_GB, Source, Notes`).
- 실측 행의 `Source = Measured-GPU`, `Notes` 에 하드웨어·반복 수·max_new_tokens·계측 방법 명시.

## 8. 참고 → 실측 교체 절차 (Swap Procedure)
```bash
# 0) 환경 준비 (GPU + 실측 의존성)
# 1) 실측 실행 → results_raw.csv 를 Source=Measured-GPU 로 덮어씀
python benchmark_driver.py --model meta-llama/Llama-3-8B --prompts 40 --max_new_tokens 128
python benchmark_driver.py --model Qwen/Qwen2.5-7B ...
#    (각 모델·정밀도에 대해 반복. INT3/INT2 는 lowbit 경로)

# 2) 다중 모델 실측 파일 갱신 (같은 스키마)
# 3) 분석·민감도·시각화 재실행
python analysis.py
python sensitivity.py        # MC 는 시드 42 로 재현
python make_figures.py

# 4) 무결성 게이트: 실패 항목 = 논문에서 갱신할 수치
python verify_numbers.py
```
- 실측치로 교체한 뒤 **반드시** `verify_numbers.py` 를 실행해 어떤 논문 수치가 낡았는지 파악하고,
  `docs/main_paper.md` 의 표·본문·부록 B 를 갱신한다.
- Power Wall 정밀화를 위해 **INT6/INT5 등 세밀 비트 샘플링**을 추가 권장.

## 9. 무결성 체크리스트
- [ ] 환경(하드웨어·드라이버·라이브러리 버전) 기록
- [ ] 동일 프롬프트·배치·max_new_tokens
- [ ] 워밍업 후 측정, 반복 ≥ 3회
- [ ] 전력/에너지 계측 방법 및 폴백 여부 명시
- [ ] `Source=Measured-GPU` 라벨링
- [ ] 정규화 규칙(E/1k, 배치 스케일) 적용
- [ ] `verify_numbers.py` 실행 및 실패 항목 정리
- [ ] 논문 표·본문·부록 B, 그림 footnote 갱신
- [ ] 연구 저널(`research_journal.md`) 및 변경 로그(`logs/change_log.md`) 갱신

---
*생성 도구: PCAG Research Pipeline. 작성일 2026-09-01. GPU 실측 단계의 표준 절차.*
