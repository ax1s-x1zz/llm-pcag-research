# PCAG Research — LLM 양자화의 전력 벽(Power Wall)과 Jevons 역설

![PCAG Research Overview](docs/figures/fig15_dashboard.png)

**The Power Wall of LLM Quantization and the Macro-Grid Paradox: PCAG Metric and the Jevons Effect**

[English README](README.md)

> **데이터 출처 고지 (Data Provenance)**
> 본 연구는 GPU 실측이 불가한 환경에서 수행되었으며, 모든 수치는 **문헌 앵커 기반 참고 데이터**(`Source=Reference-Literature`)입니다. 실측으로 위장하지 않으며, GPU 확보 시 `experiments/benchmark_driver.py` 한 번의 실행으로 동일 스키마의 실측치(`Source=Measured-GPU`)로 전체 교체 가능하도록 설계되었습니다.

---

## 연구 개요

LLM 추론 비용 폭증에 대응하는 표준 최적화인 **가중치 양자화(FP16→INT8→INT4)** 에 대해, 전력 절감 효율이 정확도 손실에 비해 **빠르게 포화·붕괴**함을 신규 지표 **PCAG(Power Cost per Accuracy Gain)** 로 정량화하고, 소프트웨어 최적화만으로 넘을 수 없는 물리적 **전력 벽(Power Wall)** 을 수학적으로 증명합니다. 나아가 효율 개선이 오히려 총 전력 소비를 늘리는 **Jevons 역설**을 폐형으로 정식화하여 매크로 전력망 관점까지 확장합니다.

### 핵심 발견

| 지표 | 결과 |
|---|---|
| PCAG 붕괴 | INT8 **20.8** → INT4 **10.4** → INT3 **4.8** (단조 감소) |
| **Power Wall** | INT4→INT3 구간, 기울기 **5.68/bit** (> θ=3), PCAG **−54.4%** 절벽 |
| 연속형 변곡점 (조건식 3.1) | 경험적 b\*≈3.51, Monte Carlo **3.40±0.25** (90% CI [2.99, 3.66]), 해석적 b\*≈**4.19** — 세 경로 모두 INT4 인접 수렴 |
| 다중 모델 일반화 | Llama-3-8B, Qwen-2.5-7B, Gemma-2-9B, Mistral-7B **전부** wall=INT4→INT3 |
| 구조적 무관성 정리 | 변곡점 방정식 h(x)=g′(x)+g(x)²=0 의 근은 **진폭(S_max, Lr_max)에 무관** → Power Wall은 구조적 결과 |
| Jevons 역설 (폐형) | TotalLoad/L₀=(1−s)^(1−E_d), **부하 증가 ⟺ E_d>1** (sympy 기호 증명) |
| INT4 (−56% 에너지) @ E_d=1.5 | 수요 **+231%**, 총 그리드 부하 **+49%** |

---

## PCAG 정의

원문 수식 [공식 3.1] `PCAG_k = ΔA_k/ΔP_k`는 항상 ≤0이 되어 해석 구간(PCAG>0)과 모순됨을 발견(ISSUE-PCAG-01)하고, 다음의 **운영(interpretable) 정의**를 채택했습니다:

```
PCAG_k = (상대 전력 절감) / (상대 정확도 손실)
       = ((P₀ − P_k)/P₀) / ((A₀ − A_k)/A₀)
```

**Power Wall 판정 조건:**
- **[조건식 3.1]** d²PCAG(b)/db² = 0 and d³PCAG(b)/db³ ≠ 0 (변곡점)
- **[조건식 3.2]** |ΔPCAG/Δb| > θ (한계효용 붕괴)

![PCAG Power Wall](docs/figures/fig2_pcag_power_wall.png)

---

## 주요 시각 자료

### PCAG 붕괴 — 에너지 vs 정확도 트레이드오프

![Accuracy vs Energy](docs/figures/fig1_accuracy_vs_energy.png)

### 다중 모델 일반화 (4개 모델 × 5 정밀도)

![Multimodel PCAG](docs/figures/fig07_multimodel_pcag.png)

### 해석적 모델: Weibull 포화(전력 절감) vs 로지스틱 가속(정확도 손실)

![Continuous Model](docs/figures/fig13_continuous_model.png)

### Monte Carlo 강건성 (σ=3%, N=3000)

![MC Robustness](docs/figures/fig11_mc_robustness.png)

### Jevons 역설: 히트맵 · 위상도 · 3D 곡면

| 그리드 부하 변화 히트맵 (E_d × s) | 위상도 — 부하 증가 영역 (65.6%) |
|:---:|:---:|
| ![Jevons Heatmap](docs/figures/fig09_jevons_heatmap.png) | ![Jevons Phase](docs/figures/fig10_jevons_phase.png) |

![Jevons Surface](docs/figures/fig16_jevons_surface.png)

### 효율 프론티어 (iso-PCAG)

![Efficiency Frontier](docs/figures/fig14_efficiency_frontier.png)

> 전체 그림 19종(PNG 200dpi + PDF 벡터)은 [`docs/figures/`](docs/figures/) 에서, 논문 부록 A의 그림 색인과 대응됩니다.

---

## 저장소 구조

```text
llm-pcag-research/
├── README.md                        # 영문 README
├── README_kr.md                     # 본 문서 (한국어)
├── INSTRUCTIONS.md                  # 연구 실행 프로토콜 (에이전트 지시서)
├── requirements.txt                 # 고정 의존성 버전 (재현성)
├── research_journal.md              # 연구 저널 (시간순 의사결정·장애 기록)
├── docs/
│   ├── main_paper.md                # 학술 논문 원고 v3 (국문/영문 초록 + 6장 + 부록)
│   ├── proof_3_1_derivation.md      # 조건식 3.1 해석적 유도 (부록 자료)
│   ├── measurement_protocol.md      # GPU 실측 표준 측정 프로토콜
│   ├── reproducibility.md           # 재현 보고서 (환경·명령열·골든 해시·검증 매트릭스)
│   ├── references.bib               # BibTeX 참고문헌 (14건)
│   ├── references/                  # 참조 문서 (PCAG 수식·변수 정의서 등)
│   └── figures/                     # 그림 19종 (PNG 200dpi + PDF)
├── experiments/                     # 재현 가능한 실험 파이프라인
│   ├── schema.py                    # 공용 CSV 스키마 (의존성 없음)
│   ├── telemetry.py                 # PyNVML/nvidia-smi 전력 계측 + 에너지 적분
│   ├── eval_harness.py              # MMLU/GSM8K (없으면 합성 논리 태스크) 평가기
│   ├── benchmark_driver.py          # FP16/INT8/INT4/INT3/INT2 실측 드라이버 (GPU용)
│   ├── lowbit.py                    # packed INT3/INT2 저비트 양자화 엔진 (실측 단계)
│   ├── generate_results.py          # 문헌 앵커 참고 데이터 (Reference-Literature, 결정적)
│   ├── multimodel_data.py           # 4모델 × 5정밀도 = 20행 앵커 (결정적)
│   ├── analysis.py                  # PCAG 산출 + Power Wall 판정
│   ├── analytical_proof.py          # 폐형 유도 + Jevons 증명 + 조건식 3.2↔3.1 일관성
│   ├── sensitivity.py               # θ 스윕 · Jevons 그리드 · Monte Carlo N=3000 (시드 42)
│   ├── statistics.py                # 부트스트랩 CI + 가설검정 (시드 20260901)
│   ├── model_form.py                # 모델-형 강건성 (b*)
│   ├── jevons_model.py              # Jevons 매크로 전력망 시뮬레이션
│   ├── make_figures.py              # 그림 19종 생성 (PNG+PDF)
│   ├── verify_numbers.py            # 학술 무결성 게이트 — 논문 수치 ↔ 산출물 (78개 항목)
│   ├── dry_run.py                   # 통합 파이프라인 검증
│   └── *.csv / *.json               # 원시 데이터 + 분석 산출물
└── logs/
    ├── troubleshooting_archive.md   # 엔지니어링 회고록 (ISSUE 11건)
    └── change_log.md                # 원고·파이프라인 버전 이력 (v1→v4)
```

---

## 재현 방법

### 요구 사항
- Python ≥ 3.10, `numpy pandas matplotlib scipy sympy` (버전은 `requirements.txt` 에 고정)
- **GPU 실측** 시: `torch`, `transformers`, `bitsandbytes`, `pynvml` 추가

### 참고 데이터 기반 전체 재현 (GPU 불필요)

```bash
pip install -r requirements.txt

cd experiments
python generate_results.py      # 1. 참고 데이터 생성 (Llama-3-8B 앵커, 결정적)
python multimodel_data.py       # 2. 다중 모델 앵커 생성 (4모델 × 5정밀도, 결정적)
python analysis.py              # 3. PCAG 분석 + Power Wall 판정
python analytical_proof.py      # 4. 해석적 유도 + Jevons 증명 + 조건식 3.2↔3.1 일관성
python sensitivity.py           # 5. θ 스윕 + Jevons 그리드 + Monte Carlo (시드 42)
python statistics.py            # 6. 부트스트랩 CI + 가설검정 (시드 20260901)
python model_form.py            # 7. 모델-형 강건성 (b*)
python jevons_model.py          # 8. Jevons 매크로 시뮬레이션
python make_figures.py          # 9. 그림 19종 생성
python verify_numbers.py        # 10. 무결성 게이트 — 논문 수치 ↔ 산출물 (78 PASS 기대)
python dry_run.py               #     (선택) 통합 파이프라인 검증
```

전체 재현 보고서(환경·골든 해시·검증 매트릭스)는 [`docs/reproducibility.md`](docs/reproducibility.md) 참조.

### GPU 실측으로 교체 (GPU 확보 시)

```bash
python benchmark_driver.py      # 실측 실행 → results_raw.csv 덮어쓰기 (Source=Measured-GPU)
# 이후 analysis.py → sensitivity.py → make_figures.py 재실행으로 전체 수치·그림 갱신
# 마지막으로 verify_numbers.py 재실행 → 어떤 논문 수치를 갱신할지 식별
```

표준 측정 절차는 [`docs/measurement_protocol.md`](docs/measurement_protocol.md) 를 따른다.

---

## 데이터 무결성 원칙

1. **출처 라벨링**: 모든 데이터 행에 `Source` 컬럼 명시 (`Reference-Literature` / `Measured-GPU`)
2. **수치 전사**: 논문 본문·표의 모든 수치는 `experiments/*.json`, `*.csv` 산출물에서 직접 전사 — 수작업 추정값 없음
3. **교차 검증**: Power Wall 위치를 경험적 PCHIP / Monte Carlo / 해석적 모델의 **3개 독립 경로**로 검증
4. **정직한 한계**: 원문 수식의 부호 모순(ISSUE-PCAG-01)을 은폐하지 않고 근거와 함께 재정의, 참고 데이터의 한계를 논문 최상단에 고지

---

## 로드맵

- [x] Phase 1: 측정·평가 파이프라인 구현 (GPU 실행 가능 상태)
- [x] Phase 2: 참고 데이터 구성 (Llama-3-8B + 4모델 확장)
- [x] Phase 3: PCAG 정식화 + Power Wall 판정 + 해석적 증명
- [x] Phase 4: Jevons 매크로 시뮬레이션 + 폐형 증명
- [x] Phase 5: 논문 원고 완성 (v2) + 그림 19종
- [x] Phase 6: 학술 완성도 강화 (v3) — 정리/정의, 기호표, 관련연구, 위협 요소, 통계 부록, 재현성, 무결성 게이트
- [x] Phase 7: 실측 전 강화 (v4) — 부트스트랩 추론·추정 가능성, 모델-형 강건성, 조건식 일관성, 외부 타당도, 사회적 영향·선언
- [ ] **GPU 실측**: `benchmark_driver.py` 실행 → 실측 데이터 교체 → 세밀 비트(INT6/INT5) 샘플링

---

## 핵심 문서

- **논문 원고 v3**: [`docs/main_paper.md`](docs/main_paper.md)
- **조건식 3.1 해석적 유도**: [`docs/proof_3_1_derivation.md`](docs/proof_3_1_derivation.md)
- **측정 프로토콜**: [`docs/measurement_protocol.md`](docs/measurement_protocol.md)
- **재현 보고서**: [`docs/reproducibility.md`](docs/reproducibility.md)
- **참고문헌 (BibTeX)**: [`docs/references.bib`](docs/references.bib)
- **연구 저널**: [`research_journal.md`](research_journal.md)
- **변경 로그**: [`logs/change_log.md`](logs/change_log.md)
- **트러블슈팅 아카이브**: [`logs/troubleshooting_archive.md`](logs/troubleshooting_archive.md) — 환경·코드·수학·데이터 이슈 11건의 회고록
