# PCAG 연구 저널 (Research Journal)

## 프로젝트 개요
- **주제**: LLM 양자화(Quantization)의 물리적 한계 검증, 신규 지표 PCAG(Power Cost per Accuracy Gain) 정식화, 그리고 Jevons Paradox 관점의 매크로 전력망 연계 분석.
- **산출물**: 제출 가능한 학술 논문 원고(docs/main_paper.md), 실험 파이프라인(experiments/), 문헌 기반 참고 데이터(results_raw.csv), 고해상도 그림(docs/figures/).
- **핵심 정의 (참조 문서 [논문 3.3절 V2])**:
  - PCAG(이산형) = ΔAk / ΔPk = (Ak - A0) / (P0 - Pk)
  - PCAG(연속형) = -dA/dP = -(dA/db)/(dP/db)
  - Power Wall 변곡점 조건식:
    - [조건식 3.1] d²PCAG(b)/db² = 0 and d³PCAG(b)/db³ ≠ 0
    - [조건식 3.2] |(PCAGk+1 - PCAGk)/(bk+1 - bk)| > θ

---

## 2026-08-27 — Phase 0: 환경 점검 및 연구 전략 수립

### 환경 사양
- OS: Linux
- CPU: Intel Core Ultra 7 155H (22 스레드)
- RAM: 7.6 GB (제한적), Swap 2 GB
- GPU: **없음** (`nvidia-smi` 미존재)
- Python: 3.14.4 (시스템 pip/ensurepip 부재)

### 주요 제약 (사실 기록)
1. GPU를 직접 빌려 실험할 수 없는 환경 → 벤치마크 드라이버를 실제 GPU 실행용으로 구현하되, 현재 단계의 수치 분석은 **문헌 기반 참고 데이터(reference data)** 로 수행해야 함.
2. 시스템 pip가 없음 → `get-pip.py`로 pip 26.2.1 부트스트랩, `/tmp/opencode/venv` 에 numpy/pandas/matplotlib/scipy/sympy 설치 완료.
3. RAM이 7.6 GB로 제한 → 실제 GPU에서도 소형 모델(Llama-3-8B 등) 기준 실험 설계 권장.

### 연구 무결성 원칙 (결정)
- **데이터 원칙**: 실제 GPU 실측을 대체하는 수치는 **절대 "실측"으로 위장하지 않는다.** 모든 참고 수치는 발표된 문헌(예: GPTQ/AWQ, MMLU 표준 벤치마크, BitsAndBytes)에서 관찰된 전형적 경향을 앵커로 합성하며, `results_raw.csv` 에 `Source=Reference-Literature` 명시. GPU 실측 시 같은 스키마로 덮어쓰는 파이프라인을 제공.
- **실행 경로**: (A) 코드는 GPU 실행 가능 상태로 완성 → (B) 현재는 참고 데이터로 전체 분석·그림·논문 완성 → (C) GPU 확보 시 `benchmark_driver.py` 한 번으로 실측 교체 가능.

### 계획 수립
Phase 1~5 순차 수행. 상세는 todo 참조.

---
## 2026-08-27 — Phase 1~4 실행 (코드베이스 + 참고 데이터 분석)

### Phase 1: 코드 완성
- `experiments/telemetry.py`: PyNVML/nvidia-smi 전원 계측, 사다리꼴 에너지 적분, CPU 폴백 경고.
- `experiments/schema.py`: 공용 CSV 스키마 (의존성 없는 경량 모듈).
- `experiments/eval_harness.py`: MMLU/GSM8K 로딩(없으면 내장 합성 논리 태스크) 평가기.
- `experiments/benchmark_driver.py`: FP16/INT8/INT4 로딩(bitsandbytes), 전원 계측, 지연/처리량/VRAM, 정확도 평가, results_raw.csv 기록. torch 지연 import.
- `experiments/generate_results.py`: 문헌 앵커 참고 데이터 생성 (Source=Reference-Literature).
- `experiments/dry_run.py`: 통합 파이프라인 검증.

### Phase 2: 참고 데이터 구성 (Llama-3-8B 앵커)
| Prec | Acc% | E J/1k | P W | VRAM GB |
|---|---|---|---|---|
| FP16 | 66.60 | 140.0 | 385 | 15.8 |
| INT8 | 65.50 | 92.0 | 312 | 8.4 |
| INT4 | 63.00 | 61.0 | 245 | 5.9 |
| INT3 | 58.00 | 54.0 | 228 | 4.8 |
| INT2 | 47.00 | 49.5 | 220 | 4.2 |

### Phase 3: PCAG 분석 결과
- PCAG(운영 정의) = (상대 전력 절감)/(상대 정확도 손실). 이유: 원문 [공식 3.1] ΔA/ΔP는 항상 ≤0 → 3.3.2의 PCAG>0 구간과 모순(ISSUE-PCAG-01). 재정의.
- PCAG: INT8=20.76, INT4=10.44, INT3=4.76, INT2=2.20 (단조 감소).
- **[조건식 3.2]** 기울기: INT4→INT3=5.68 (>θ=3) → Power Wall 구간.
- **[조건식 3.1]** 연속형 변곡점: b≈3.51 (이산 판정과 일치).
- PCAG cliff: bits4→3 = 54.4% 급락.
- Fig1/Fig2 생성.

### Phase 4: Jevons 시뮬레이션 결과
- E_d=1.5(탄력적 수요) 조건, 수요곡선 Q(P)=Q0(P/P0)^(-E_d).
- INT4(토큰당 에너지 -56%): 수요 +231%, 총 그리드 부하 +49%.
- E_d>1이면 모든 절감률에서 부하 증가(Jevons 역치 0%) → 역설 성립.
- Fig3 생성.

### 수치 무결성 검증
- 논문 표/본문의 PCAG(20.76/10.44/4.76/2.20), slope(5.68), cliff(54.4%), inflection(b≈3.51), Jevons(+231%/+49%) 모두 analysis_summary.json / jevons_summary.json 과 일치 확인.

### 산출물
- docs/main_paper.md (국문/영문 초록 + 6개 장 + 참고문헌)
- docs/figures/ fig1~fig3 (png+pdf)
- logs/troubleshooting_archive.md (ISSUE-ENV-01/02, CODE-01/02/03, PCAG-01, DATA-01)

---

## 2026-08-28 — Phase 6: GPU 무관 확장 (일반화·강건성·해석적 증명·시각화 17종)

### 배경
GPU 여전히 미확보 → GPU 없이 가능한 작업을 전부 수행하기로 결정. 특히 시각 자료 대량 요청.

### 환경 복구
- /tmp venv 소멸 확인 (ISSUE-ENV-03). get-pip → virtualenv → venv 재구축, deps OK.
- 참고: 논문 v2 갱신까지 총 소요 세션 1회.

### 신규 코드 (experiments/)
- `multimodel_data.py`: 4개 모델(Llama-3-8B, Qwen-2.5-7B, Gemma-2-9B, Mistral-7B) × 5 정밀도 = 20행 앵커.
  물리 정합성 규칙 E ≈ 0.0119×P×latency(Llama 앵커 역산 배치 스케일) 적용. Source=Reference-Literature.
- `sensitivity.py`: (A) θ 스윕 [0.5,10] — 4개 모델 모두 θ<5.68에서 wall=INT4→INT3 불변.
  (B) Jevons 그리드 E_d×s → 폐형 (1−s)^(1−E_d) 확인, 부하증가 영역 65.6%.
  (C) Monte Carlo N=3000, σ=3% — 변곡점 3.398±0.252 (90% CI [2.99,3.66]),
  wall 전이 최빈값 INT4→INT3 (기록된 실행 중 66.8%).
- `analytical_proof.py`: 조건식 3.1 해석적 유도.
  모델: S(x)=S_max(1−e^-(λx)^β) Weibull 포화(RMSE 2.2e−3),
  Lr(x)=c·x+Lr_max·σ(k(x−xc)) 선형+로지스틱 가속(RMSE 7.4e−10).
  핵심 정리: h(x)=g'(x)+g(x)²=0 의 근은 진폭(S_max, Lr_max)에 무관 → Power Wall은 구조적.
  해석적 b*=4.19 (관측 도메인 내 유일근, d³≠0). 경험적 3.51 / MC 3.40±0.25와 INT4 인접 수렴.
  Jevons 폐형 dL/ds=(E_d−1)(1−s)^−E_d 를 sympy로 기호 증명(True) → "부하 증가 ⟺ E_d>1".
  산출: `analysis_proof.json`, `docs/proof_3_1_derivation.md`.
- `make_figures.py`: 그림 17종 생성 (PNG 200dpi + PDF). 기존 fig1~3 개선 +
  fig04~fig17 신규 (자원 사용량, 정확도 절벽, 발산 메커니즘, 다중 모델 PCAG/유지율,
  Jevons 히트맵/위상도/3D 곡면, MC 강건성, θ 민감도, 해석 모델 3패널, iso-PCAG 프론티어,
  대시보드, 벽 수렴 비교). 모든 그림에 Reference-Literature 각주 명시.

### 기술 장애 기록
- ISSUE-CODE-04: 복소수 스텝 미분에서 np.expm1(복소수 미지원)·np.real(Im 대신 Re) 오용.
- ISSUE-CODE-05: sympy simplify 기호 지수(Weibull β) 조합폭발 → 폐형 직접 유도+수치 검증 전환.
- ISSUE-CODE-06: MC에서 float() 스칼라 래퍼 TypeError가 except:pass로 매몰 → 0샘플 버그.
- 상세: logs/troubleshooting_archive.md.

### 논문 v2 갱신 (docs/main_paper.md)
- 국문/영문 초록: 다중 모델 일반화, 해석적 정리, MC 강건성, Jevons 폐형 반영.
- 3.1: 4개 모델 + 보조 파이프라인 명시. 3.2: 다중 모델 앵커 구성 규칙 추가.
- 3.3.4 신설: 조건식 3.1 폐형 유도 + 구조적 무관성 정리.
- 4.3 신설: 다중 모델 일반화 (4개 모델 모두 wall=INT4→INT3, 표는 multimodel CSV 실계산값).
- 4.4 신설: θ 스윕/Monte Carlo/경로 간 교차 검증.
- 5.1: Jevons 폐형 유도 + "증가 ⟺ E_d>1" 증명 + 민감도 그리드 결과.
- 6.1/6.3 요약·한계 갱신, 부록 A 그림 색인(17종) 추가.

### 수치 무결성 검증
- 다중 모델 PCAG 표: 스크립트 실계산값으로 채움(20.0/10.4/4.8/2.2 등) — 추정값 아님.
- Jevons 폐형 +49.07% ↔ jevons_summary.json +49% 일치 확인.
- 해석 모델 PCAG vs 앵커 이산 PCAG: 최대 |Δ|=0.045 (RMSE≈0.03).
- 논문 인용 수치는 모두 JSON/CSV 산출물에서 직접 전사.

### 현재 상태
- GPU 무관 수행 가능한 작업: 완료. 남은 것은 GPU 실측뿐.
- 다음 단계(GPU 확보 시): benchmark_driver.py 실측 실행 → results_raw.csv 교체(Source=Measured-GPU)
  → analysis/sensitivity/figures 재실행 → 논문 수치 갱신. 세밀 비트(INT6/INT5) 샘플링 권장.

---
## 2026-09-01 — Phase 7: 학술 완성도 강화 + 재현성 하드닝 (pre-Colab 원래 연구 방향)

### 배경
저장소를 "Colab T4 최적화" 이전 상태로 되돌려 원래 연구(문헌 앵커 참고 데이터 기반)를 main으로 재편.
Colab 최적화 상태는 별도 브랜치(colab-t4-optimization)로 격리. 이후 main에서 학술 가치 제고 작업 수행.

### 환경 복구 (ISSUE-ENV-03 재발)
- /tmp venv 소멸 확인. 이번에는 재현성을 위해 **저장소 내 persistent `.venv`** 를 구축.
- ensurepip 부재로 `get-pip.py` 부트스트랩 재적용 (ISSUE-ENV-02와 동일 경로).
- numpy 2.5.2 / pandas 3.0.5 / matplotlib 3.11.1 / scipy 1.18.1 / sympy 1.14.0 설치.
- `requirements.txt` 에 검증 버전 고정.

### 재현성 하드닝
- **비결정성 제거**: `generate_results.py`·`multimodel_data.py` 가 `datetime.date.today()` 로 Notes에
  생성일을 박아 재실행 시마다 CSV가 더러워지는 문제 발견(ISSUE-REPROD-01). 고정 상수
  `ANCHOR_GEN_DATE`(2026-08-27 / 2026-08-28)로 교체 → **byte-identical 재생성 확인**.
- 골든 해시(CSV/JSON 6종)를 `docs/reproducibility.md` 에 기록.

### 학술 문서 v3
- `main_paper.md` v3: 정의 3.1~3.3·정리 3.4·5.1·계 3.5·5.2 번호 부여, 기호표(3.3.0),
  관련연구 확장+인용 [1]~[14], 위협 요소 절(4.5), 재현성·데이터/코드 가용성 진술,
  통계 부록 B(다중 모델·MC·θ스윕·Jevons 표).
- `docs/references.bib` 신설 (BibTeX 14건).
- `docs/measurement_protocol.md` 신설 (실측 표준 절차·스왑 프로토콜·체크리스트).
- `docs/reproducibility.md` 신설 (환경·명령열·골든 해시·검증 매트릭스).
- `logs/change_log.md` 신설 (v1/v2/v3 이력).

### 무결성 게이트 도구
- `experiments/verify_numbers.py` 신설 — 논문 인용 수치 60개를 CSV/JSON 산출물과 자동 대조
  (PCAG, Power Wall, 변곡점 3경로, MC, θ스윕, 다중 모델, Jevons). **60 PASS / 0 FAIL**.
- GPU 실측 교체 시 이 게이트의 실패 항목 = 논문에서 갱신할 수치 식별 장치.

### 수치 무결성 검증
- 전체 파이프라인 재실행: dry_run OK, analysis/analytical_proof 재실행 결과 JSON 무변경 확인.
- 다중 모델 PCAG 실계산값(부록 B.2) 전사: Llama 20.02/10.40/4.80/2.19, Qwen 30.48/15.36/5.58/2.38,
  Gemma 21.57/11.52/5.03/2.22, Mistral 24.40/11.49/4.54/2.07.

### 현재 상태
- main = pre-Colab 원래 연구 + v3 문서/도구. Colab 최적화는 `colab-t4-optimization` 브랜치에 보존.
- 다음 단계(GPU 확보 시): measurement_protocol.md Swap Procedure → 실측 교체 → verify_numbers.py
  로 갱신 수치 식별 → 논문·부록·해시 갱신.
- 참고: 원격(origin) 반영은 로컬 자격증명(credential.helper=-l) 문제로 보류 — `git push --force origin main`
  및 `git push -u origin colab-t4-optimization` 수동 실행 필요.

---
## 2026-09-01 — Phase 8: 실측 전 학술 가치 제고 (v4: 통계 추론·모델-형 강건성·이산-연속 일관성·외부 타당도)

### 배경
main(원래 연구)에서 GPU 실측 전에 학술적 가치를 더 높일 수 있는 분석·문서 작업을 수행.
푸시 자격증명(credential.helper=-l)은 `store` 헬퍼로 교체해 원격에도 반영 완료.

### 부트스트랩 통계 추론 (`statistics.py`)
- 앵커 로그정규 잡음(σ=3%, N=3000, 시드 20260901) 하에서 각 PCAG·이산 기울기·연속 변곡점의
  90% CI와 가설검정 산출.
- **강건 정량**: 연속 변곡점 mean 3.39, 90% CI [2.99, 3.65] (MC 3.40±0.25와 일치);
  심층 양자화 PCAG(INT2) 2.22 CI [1.88, 2.66] — 안정.
- **정직한 한계 (estimability)**: 근접-기준 PCAG(INT8)는 잡음에 극히 취약(mean 52.1,
  CI [3.8, 93.7]); 이산 기울기 유의성은 불충분(INT4→3>θ=3: P=0.60, 유의하지 않음).
  → Power Wall의 방어 가능한 증거는 연속 변곡점·심층 붕괴·해석 모델에 두어야 함을 명시.

### 모델-형 강건성 (`model_form.py`)
- 대안 함수형(절감: Weibull/Exp/Tanh/Hill × 손실: LinLog/Power/Logistic/LinPow)에 대한 b\*.
- 표준 포화형 4종 모두에서 b\* ∈ [4.19, 4.27] — 정리 3.4 구조적 무관성과 정합.
- 가속 손실 성분 부재 시 도메인 내 변곡점 없음 → 변곡 존재에 가속 구조가 필수.

### 조건식 3.2 ↔ 3.1 일관성 (`analytical_proof.py` 확장)
- 적합 모델 단위 비트 드롭 최대가 4→3(+6.68), 연속 붕괴율 최대 b≈4.19, 변곡점 b\*=4.19
  → 이산 판정과 연속 판정이 동일 위치(명제 3.6).

### 외부 타당도 (4.8)
- 문헌(GPTQ/AWQ) 전형 PTQ INT4 유지율(~95~99%) 대비 앵커 유지율(94.6~96.4%) 정합 확인.

### 수치 무결성
- verify_numbers.py 60→78 항목 확장. **78 PASS / 0 FAIL.**
- 골든 해시 갱신(analysis_proof/statistics/model_form), 재현 명령열 갱신.

### 시각화·파이프라인 확장
- `make_figures.py` : Fig 18(부트스트랩 추론 — PCAG 90% CI + 변곡점 분포),
  Fig 19(모델-형 강건성 — 대안 함수형 b\*) 추가. 17→19종.
- `dry_run.py` : statistics.py·model_form.py·analytical_proof.py·verify_numbers.py 포함
  전체 파이프라인 통합 검증으로 확장. 실행 결과 전 단계 OK, 산출물(JSON 7종+fig1/2/3/18/19) 확인.

### 현재 상태
- main = pre-Colab 원래 연구 + v4 문서/도구. Colab 최적화는 `colab-t4-optimization` 브랜치.
- 다음 단계(GPU 확보 시): measurement_protocol.md Swap Procedure → 실측 교체 →
  verify_numbers.py 로 갱신 수치 식별 → 논문·부록·해시 갱신.
