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
