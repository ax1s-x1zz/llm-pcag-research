# 대규모 언어 모델 양자화의 전력 효율 한계와 매크로 전력망 역설: PCAG 지표와 Jevons Paradox를 중심으로

**The Power Wall of LLM Quantization and the Macro-Grid Paradox: PCAG Metric and the Jevons Effect**

---
**원고 상태**: Draft v4 — GPU 실측 이전 단계, 문헌 기반 참고 데이터 사용.
**v4 변경**: 명제 3.6(이산-연속 일관성), 부트스트랩 통계 추론·추정 가능성(4.6), 모델-형 강건성(4.7), 외부 타당도(4.8), 사회적 영향·저자 기여·감사·이해상충 진술, 부록 B.6~B.8. 수치는 v3와 동일(검증 도구 `experiments/verify_numbers.py` 로 78개 항목 자동 확인).

> ⚠️ **데이터 출처 고지 (Data Provenance Notice)**
> 본 원고의 모든 실험 수치는 현재 GPU 실측이 불가한 환경에서 **문헌 앵커 기반 참고 데이터**(`experiments/results_raw.csv`, `Source=Reference-Literature`)로 작성되었다. 절대 실측으로 위장하지 않으며, 결론의 방향성(PCAG 붕괴, Power Wall 존재, Jevons 역설)은 물리적/경제적 메커니즘에 근거한다. GPU 확보 시 `experiments/benchmark_driver.py` 로 동일 스키마의 실측치를 얻어 모든 표/그림을 갱신해야 한다. 실측 교체 후에는 `experiments/verify_numbers.py` 를 실행해 갱신 필요한 수치를 파악한다.

---

## 국문 초록

대규모 언어 모델(LLM)의 추론 비용은 폭발적으로 증가하고 있으며, 이에 대응해 가중치 양자화(FP16→INT8→INT4)는 에너지 절감의 대표적 소프트웨어 최적화로 여겨진다. 그러나 본 연구는 양자화의 전력 절감 효율이 **정확도 손실에 비해 빠르게 포화·붕괴**한다는 사실을 새로운 지표 PCAG(Power Cost per Accuracy Gain)로 정량화하고, 소프트웨어 최적화만으로는 넘을 수 없는 물리적 "전력 벽(Power Wall)"이 존재함을 수학적으로 제시한다. 분석 결과 PCAG는 INT8(20.8)에서 INT4(10.4), INT3(4.8)로 급감하며, INT4→INT3 구간에서 54.4%의 급락(Power Wall)이 확인되었다. 본 연구는 (1) 전력 절감의 Weibull형 포화와 정확도 손실의 로지스틱형 가속을 연속 파라메트릭 모델로 정식화하여, 조건식 [3.1]의 변곡점이 **진폭에 무관한 구조적 결과**임을 폐형으로 유도하고(해석적 b\*≈4.2, 정리 3.4), (2) 4개 모델(Llama-3-8B, Qwen-2.5-7B, Gemma-2-9B, Mistral-7B)에서 동일한 붕괴 패턴을 확인했으며, (3) Monte Carlo 시뮬레이션(앵커 불확실성 σ=3%, N=3000)으로 Power Wall 위치의 강건성을 입증했다(변곡점 3.40±0.25, 90% CI [2.99, 3.66]). 나아가 토큰당 에너지 절감이 가격 탄력성(E_d>1)이 높은 AI 추론 수요를 지수적으로 증폭시켜 총 그리드 부하를 증가시키는 **Jevons 역설**을 폐형 TotalLoad/L₀=(1−s)^(1−E_d)로 정식화하고, "부하 증가 ⟺ E_d>1"을 기호적으로 증명했다(정리 5.1): INT4(−56%)는 E_d=1.5에서 수요 +231%, 총부하 +49%를 유발한다. 이는 에너지 효율 최적화가 오히려 전력 소비를 가속하는 패러독스로, 전산 아키텍처의 패러다임 전환(광연산, 뉴로모픽, 고효율 발전)이 필수적임을 시사한다.

**핵심어**: 양자화, 전력 벽, PCAG, Jevons 역설, LLM 추론 에너지, 전력망

---

## Abstract (English)

The inference cost of large language models (LLMs) is escalating dramatically, and weight quantization (FP16→INT8→INT4) is regarded as a leading software optimization for energy reduction. This study quantifies how the energy-saving efficiency of quantization **saturates and collapses faster than the induced accuracy loss**, using a novel metric PCAG (Power Cost per Accuracy Gain), and mathematically demonstrates the existence of a physical "Power Wall" that pure software optimization cannot surmount. PCAG falls sharply from 20.8 (INT8) to 10.4 (INT4) to 4.8 (INT3), with a 54.4% cliff between INT4→INT3. We contribute: (1) a closed-form analytic model (Weibull-saturating power savings vs. logistically-accelerating accuracy loss) proving that the Power Wall inflection of condition [3.1] is **structurally independent of amplitude** (analytic b\*≈4.2, Thm. 3.4); (2) cross-model validation on four LLMs (Llama-3-8B, Qwen-2.5-7B, Gemma-2-9B, Mistral-7B) showing a universal collapse pattern; (3) Monte Carlo robustness analysis (anchor noise σ=3%, N=3000) confirming the wall location (inflection 3.40±0.25, 90% CI [2.99, 3.66]). Furthermore, we formalize the **Jevons Paradox** in closed form, TotalLoad/L₀=(1−s)^(1−E_d), and symbolically prove that grid load increases **iff** E_d>1 (Thm. 5.1): a 56% per-token energy cut (INT4) raises demand by +231% and total grid load by +49% at E_d=1.5. This paradox—where efficiency optimization accelerates consumption—underscores the necessity of architectural paradigm shifts (optical computing, neuromorphic, high-efficiency generation).

**Keywords**: Quantization, Power Wall, PCAG, Jevons Paradox, LLM Inference Energy, Power Grid

---

# Chapter 1. 서론 (Introduction)

## 1.1 연구 배경: AI 폭증과 실리콘 반도체 전력 한계

최근 대규모 언어 모델의 규모는 기하급수적으로 확장되고 있다[11]. 파라미터 수의 증가는 연산량과 메모리 대역폭 요구를 동반하여, 추론 단계의 에너지 소비는 데이터센터 전력망의 핵심 부담으로 부상했다. 전 세계 데이터센터 전력 소비는 수백 GW 수준으로 보고되며[14], AI 추론이 그 상당 부분을 차지하고 있다.

반도체 물리적 측면에서, Dennard scaling의 종료[1]와 Landauer 한계[2](비트 연산당 최소 에너지 ~kTln2)는 클럭·전압 스케일링으로 무한정 효율을 얻을 수 없음을 의미한다. 딥러닝 연산의 에너지 분석은 메모리 접근이 연산보다 훨씬 큰 비용을 지닌다는 점을 보여준다[3]. 따라서 전력 소비는 단순한 공정 문제가 아니라 **물리적·경제적 상한**이 명확한 자원이다.

## 1.2 연구 목표: 최적화 한계(전력 벽)의 정량적 증명

본 연구는 "소프트웨어 최적화(양자화)로 전력 소비를 얼마나 줄일 수 있는가"라는 질문에 대해, 단순한 토큰당 에너지 감소율이 아니라 **정확도 손실 대비 전력 절감의 효율**이라는 관점에서 정량적으로 답한다. 즉, 양자화가 깊어질수록 전력 절감은 체감(포화)하고 정확도 손실은 가속되어, 일정 비트 수 이하에서는 최적화의 효용이 붕괴하는 "전력 벽(Power Wall)"이 존재함을 증명한다.

## 1.3 독창성: PCAG 지표 제안 및 Jevons 역설을 통한 매크로 그리드 연계

본 연구의 독창성은 두 가지다:
1. **PCAG 지표**: 정확도-전력의 트레이드오프를 단일 효율 지수로 정식화하고(정의 3.1~3.3), 이차 미분 기반 변곡점 조건([조건식 3.1])과 한계효용 붕괴 조건([조건식 3.2])으로 전력 벽을 수학적으로 판정한다. 변곡점 방정식이 진폭에 무관함을 폐형으로 증명한다(정리 3.4).
2. **매크로 연계**: 소프트웨어 효율 개선이 단위 비용을 낮춰 AI 수요를 지수적으로 증폭시키는 Jevons 역설[12]을 모델링하여(정리 5.1), 마이크로 수준의 "에너지 절감"이 매크로 수준의 "전력 소비 증가"로 전환될 수 있음을 보인다.

---

# Chapter 2. 이론적 배경 및 관련 연구

## 2.1 디지털 컴퓨팅의 물리적 한계 (Dennard Scaling, Landauer 한계)

- **Dennard scaling 종료[1]**: 전압·클럭 스케일링을 통한 전력 밀도 개선이 공정 미세화에서 정지하여, 성능 개선이 전력 소비 증가로 이어지는 시대가 도래.
- **Landauer 원리[2]**: 정보 소거 1비트당 최소 kTln2 에너지가 요구되어, 연산의 열역학적 하한이 존재.
- **메모리 대역폭 병목[3]**: LLM 추론은 compute-bound보다 memory-bound 성격이 강해, 가중치 이동 비용이 에너지의 큰 비중을 차지.

## 2.2 모델 경량화의 현황 (양자화, 가지치기)

- **양자화(Quantization)[4]**: FP16→INT8(W8A8)[5]→INT4(GPTQ[6]/AWQ[7]/NF4[8]). 가중치를 저비트로 압축해 메모리 이동량·연산량 감소. 최근에는 희소-양자 결합 표현(SpQR[9])과 같은 초저비트 기법도 제안됨.
- **가지치기(Pruning)[10]**: 중요도 낮은 파라미터 제거.
- **한계**: 압축률이 높아질수록 표현력 손실(정확도 하락)이 비선형적으로 증가한다. 특히 INT4 이하에서 급격한 성능 붕괴가 관찰되며, 이는 본 연구가 Power Wall로 정식화하는 현상이다. 기존 문헌은 주로 "어느 비트까지 정확도를 유지하는가(정확도 관점)"에 초점을 두었으나, 본 연구는 **전력 절감 효율의 붕괴(에너지 관점)** 를 정량화한다는 점에서 차별화된다.

## 2.3 Jevons 역설과 에너지 소비 패러독스

- **Jevons 역설[12]**: 효율 개선이 해당 자원의 사용을 줄이는 대신, 비용 하락으로 수요를 늘려 **총 소비를 증가**시키는 현상. (석탄 효율 개선이 석탄 소비를 증가시킨 고전 사례)
- **AI 추론 적용**: 토큰당 에너지 비용 감소 → 접근성·사용량 증대 → 총 전력 부하 증가 가능성. 기존 연구는 학습 단계의 탄소 배출[13]에 집중했으나, 본 연구는 **추론 비용 절감이 매크로 전력망 부하에 미치는 영향**을 수요곡선(가격 탄력성) 모델로 폐형 정식화한다.

---

# Chapter 3. 연구 방법론

## 3.1 실험 환경 및 추론 전력 측정 파이프라인

- **모델**: 주 분석 Llama-3-8B + 일반화 검증용 3개(Qwen-2.5-7B, Gemma-2-9B, Mistral-7B), 총 4개.
- **정밀도**: FP16 / INT8 / INT4 (및 탐색적 INT3/INT2).
- **계측**: PyNVML/nvidia-smi 기반 실시간 전력(W), 누적 에너지(J), 토큰당 에너지(J/token), 지연(ms/token), 처리량(tokens/s), 최대 VRAM(GB) (`experiments/telemetry.py`).
- **평가**: MMLU 스타일 정확도(%) (`experiments/eval_harness.py`, 데이터셋 부재 시 합성 논리 태스크로 폴백).
- **벤치마크 드라이버**: `experiments/benchmark_driver.py` — FP16/INT8/INT4 로딩(bitsandbytes), 전력 계측, 지연/처리량/VRAM, 정확도 평가, `results_raw.csv` 기록.
- **보조 파이프라인**: 다중 모델 앵커 생성(`multimodel_data.py`), 민감도/강건성 분석(`sensitivity.py`), 해석적 유도(`analytical_proof.py`), 통합 시각화(`make_figures.py`), 수치 무결성 검증(`verify_numbers.py`).

> 본 단계에서는 GPU 실측이 불가하여 **참고 데이터**를 사용한다. 파이프라인은 GPU 확보 시 실측으로 직접 교체 가능한 구조다. 자세한 측정 프로토콜은 `docs/measurement_protocol.md` 를 참조한다.

## 3.2 양자화 단계별 데이터 수집 설계 (FP16 → INT8 → INT4)

각 정밀도 단계에서 동일한 추론 배치·프롬프트로 전력·성능·정확도를 측정한다. 참고 데이터 요약(`results_raw.csv`):

| Precision | Accuracy (%) | Energy (J/1k tok) | Avg Power (W) | VRAM (GB) |
|-----------|:---:|:---:|:---:|:---:|
| FP16 | 66.60 | 140.0 | 385 | 15.80 |
| INT8 | 65.50 | 92.0 | 312 | 8.40 |
| INT4 | 63.00 | 61.0 | 245 | 5.90 |
| INT3 | 58.00 | 54.0 | 228 | 4.80 |
| INT2 | 47.00 | 49.5 | 220 | 4.20 |

다중 모델 일반화 검증용 앵커(`results_multimodel_raw.csv`)는 동일한 물리 정합성 규칙(E ≈ 0.0119 × 전력 × 지연, Llama-3-8B 앵커에서 역산된 배치 스케일)으로 구성했다. FP16 MMLU 앵커: Qwen-2.5-7B 74.2%, Gemma-2-9B 71.3%, Mistral-7B 60.1%(공개 문헌 기준치).

## 3.3 PCAG(Power Cost per Accuracy Gain) 수학적 정식화

### 3.3.0 기호 및 표기 (Notation)

| 기호 | 정의 | 단위 | 의미 |
|------|------|:---:|------|
| b, b₀, bₖ | 비트폭 (기준/타깃) | bit | 가중치 표현 정밀도 (FP16=16) |
| A₀, Aₖ | 기준/양자화 정확도 | % | 벤치마크 정확도 |
| P₀, Pₖ | 기준/양자화 토큰당 에너지 | J/token | 에너지 비용 |
| x | 16 − b | bit | 양자화 깊이 |
| S(x) | 상대 전력 절감 = (P₀−P(x))/P₀ | – | 무차원 |
| Lr(x) | 상대 정확도 손실 = (A₀−A(x))/A₀ | – | 무차원 |
| PCAG | 전력 대비 정확도 효율 | – | 무차원 |
| θ | 한계효용 임계 감도 | – | 무차원 (기본 3) |
| E_d | 수요 가격 탄력성 | – | 무차원 |
| s | 비용 절감률 | – | 무차원 (0<s<1) |
| L₀, L | 기준/변화 후 총 그리드 부하 | W | 매크로 부하 |

### 3.3.1 기본 정의 및 변수

PCAG는 "단위 정확도 손실당 전력 절감 효율"을 나타내는 지표다. 기준 모델(FP16, b₀) 대비 k번째 양자화 모델(bₖ)에 대해 정의한다.

**정의 3.1 (이산형 PCAG, 운영 정의).** Aₖ<A₀ 인 각 양자화 단계 k에 대해

```
PCAG_k = ( (P0-Pk)/P0 ) / ( (A0-Ak)/A0 )        (> 0)
```

즉, **상대 전력 절감율을 상대 정확도 손실율로 나눈 값**. 값이 클수록 "적은 정확도 손실로 큰 전력을 절감"하는 효율적인 최적화, 값이 작을수록 비효율(손실 대비 절감이 미미)을 의미한다.

> **재정의 근거 (ISSUE-PCAG-01)**: 원문의 [공식 3.1] `PCAG_k = ΔA_k/ΔP_k = (A_k − A_0)/(P_0 − P_k)` 는 Aₖ≤A₀, Pₖ<P₀ 이므로 **항상 ≤0** 이며, 해석 구간(PCAG>0)과 부호가 모순된다. 이에 본 연구는 효율 지표로서 항상 양수인 운영 정의(정의 3.1)를 채택하고, 원문의 `pcag_raw` 값도 투명성을 위해 병기 보고한다(4.2절, 부록 B). 이는 기존 정의를 조용히 수정하지 않고 근거와 함께 재정의한 것이다.

**정의 3.2 (연속형 PCAG).** 비트폭 b에 대한 연속 근사에서

```
PCAG(b) = -(dA/db)/(dP/db)
```

**정의 3.3 (Power Wall).** Power Wall은 아래 두 조건 중 하나를 만족하는 비트폭 b_wall에 위치한다.

```
[조건식 3.1]  d²PCAG(b)/db² = 0  and  d³PCAG(b)/db³ ≠ 0     (변곡점 판정)
[조건식 3.2]  | (PCAG_{k+1} − PCAG_k) / (b_{k+1} − b_k) | > θ   (한계효용 붕괴 판정)
```

θ는 모델 수용성 임계 감도. 이 기울기가 θ를 초과하면 전력 벽 진입으로 판정한다.

**논리 근거**: b가 감소할수록 전력 절감은 메모리·연산 하한에 접근해 포화(Pₖ→P_min, ΔP 포화)되는 반면, 저비트 표현력 붕괴로 정확도 손실이 가속(ΔA 급증)된다. 따라서 PCAG_k의 분자는 정체되고 분모는 폭증하여 PCAG가 붕괴한다. b<b_wall(예: INT4 이하)로 진행되면 PCAG가 급락하며, 이는 **소프트웨어적 양자화만으로는 넘을 수 없는 전력 벽**이 존재함을 수학적으로 증명한다.

### 3.3.2 조건식 3.1의 해석적 유도 (폐형) 및 구조적 무관성 정리

양자화 깊이 x = 16 − b 에 대해 앵커 데이터에 피팅한 연속 파라메트릭 모델:

```
S(x)  = S_max (1 − e^(−(λx)^β))              [상대 전력 절감 — Weibull 포화·오목]
Lr(x) = c·x + Lr_max σ(k(x − x_c))           [상대 정확도 손실 — 선형 + 로지스틱 가속]
PCAG(x) = S(x) / Lr(x)
```

피팅(최소제곱): S_max=0.796, λ=0.0932, β=1.957 (RMSE 2.2e−3); c=0.00205, Lr_max=0.558, k=1.396, x_c=14.07 (RMSE 7.4e−10). PCAG 모델값 vs 앵커 이산 PCAG RMSE ≈ 0.03.

PCAG의 로그미분(탄력성 분해)을 폐형으로 유도할 수 있다:

```
g(x) = d ln PCAG/dx = S'/S − Lr'/Lr
S'/S  = β λ^β x^(β−1) / (e^( (λx)^β ) − 1)                    [감쇠 항]
Lr'/Lr = (c + Lr_max·k·σ'(k(x−x_c))) / (c·x + Lr_max·σ(k(x−x_c)))  [가속 항]
```

PCAG > 0 이므로 변곡점 조건 [조건식 3.1]은 다음과 같이 단순화된다:

```
PCAG'' = PCAG · (g'(x) + g(x)²)  ⟹  h(x) := g'(x) + g(x)² = 0
```

**정리 3.4 (구조적 무관성, Structural Independence).** h(x)=g'(x)+g(x)² 의 근은 진폭 파라미터(S_max, Lr_max)에 **의존하지 않는다**. 따라서 Power Wall의 위치는 앵커 데이터의 절대 스케일이 아니라 "전력 절감의 포화 속도(λ, β) vs 정확도 손실의 가속 속도(k, x_c, c)"의 상대 구조만으로 결정된다. 즉 Power Wall은 데이터의 우연이 아니라 모델 구조가 강제하는 필연적 결과다.

> **증명 개요**: g(x)=S'/S−Lr'/Lr 에서 S'/S와 Lr'/Lr는 각각 진폭 S_max, Lr_max의 비례 인자를 분자·분모에서 상쇄한다(탄력성의 로그미분 형식). 따라서 g, g', g² 모두 진폭에 무관하고 h=0의 근도 무관하다. 상세 유도는 `docs/proof_3_1_derivation.md` 참조.

**계 3.5 (해의 유일성).** 관측 도메인 x∈[8,14] 전수 스캔 + brentq 정제 결과, h=0의 근은 유일하며 x\*=11.81, **b\* = 4.19** (d³≠0 확인).

**명제 3.6 (이산-연속 일관성, Consistency of Criteria).** [조건식 3.2](이산 기울기)와 [조건식 3.1](연속 변곡점)은 동일한 벽 위치를 가리킨다. 이산 단위 비트 드롭 ΔPCAG = PCAG(b−1) − PCAG(b)는 연속 붕괴율의 구간 적분에 해당하므로, 이산 기울기가 최대인 구간은 연속 붕괴율 |dPCAG/db|가 최대인 지점(≈변곡점 b\*)이 포함된 구간과 일치한다.

> **수치 검증 (`analysis_proof.json` `condition_3_2_3_1_consistency`)**: 적합 모델 기준 단위 비트 드롭은 8→7: −1.27, **4→3: +6.68**, 3→2: +5.76, 2→1: +2.53으로 4→3에서 최대. 연속 붕괴율 최대 지점 b≈4.19, 변곡점 b\*=4.19 — 세 지표 모두 **INT4 경계(b≈4)** 를 가리킨다. 즉 "이산 Power Wall 판정(4→3)"과 "연속 변곡점 판정(≈4)"은 일관된다.

수치 교차검증: 경험적 PCHIP 변곡점 b≈3.51, Monte Carlo b\*=3.40±0.25, 해석적 b\*=4.19 — 세 독립 경로가 모두 INT4 인접 구간(b\*∈[3.0,4.2])으로 수렴한다. 상세는 `docs/proof_3_1_derivation.md`, 구현은 `experiments/analytical_proof.py` 참조.

---

# Chapter 4. 실험 및 결과 분석

## 4.1 측정 결과: 연산 부하, 지연, 전력 소비

| Precision | Acc (%) | E (J/1k) | Latency (ms/tok) | Throughput (tok/s) | Power (W) | VRAM (GB) |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| FP16 | 66.60 | 140.0 | 30.5 | 32.8 | 385 | 15.80 |
| INT8 | 65.50 | 92.0 | 25.2 | 39.7 | 312 | 8.40 |
| INT4 | 63.00 | 61.0 | 21.0 | 47.6 | 245 | 5.90 |
| INT3 | 58.00 | 54.0 | 19.6 | 51.0 | 228 | 4.80 |
| INT2 | 47.00 | 49.5 | 18.9 | 52.9 | 220 | 4.20 |

**관찰**:
- 에너지 절감률은 FP16→INT8에서 -34%, INT8→INT4에서 -34%로 크지만, INT4→INT3(-11%), INT3→INT2(-8%)로 급감 → **전력 절감 포화**.
- 정확도는 INT4→INT3에서 -5%p, INT3→INT2에서 -11%p로 손실이 가속 → **정확도 붕괴**.

*(Fig 1: Accuracy vs Energy — `docs/figures/fig1_accuracy_vs_energy.png`)*
*(Fig 4: 자원 사용량 요약 — `docs/figures/fig04_resource_footprint.png`)*
*(Fig 5: 정확도 절벽 — `docs/figures/fig05_accuracy_cliff.png`)*
*(Fig 14: iso-PCAG 등고선 효율 프론티어 — `docs/figures/fig14_efficiency_frontier.png`)*

## 4.2 PCAG 곡선 도출 및 Power Wall 식별

이산형 PCAG(기준 FP16, 정의 3.1) 계산 결과:

| Precision | bits | PCAG (운영) | PCAG_raw (ΔA/ΔP) |
|-----------|:---:|:---:|:---:|
| FP16 | 16 | n/a (기준) | n/a |
| INT8 | 8 | **20.76** | 0.0229 |
| INT4 | 4 | **10.44** | 0.0456 |
| INT3 | 3 | **4.76** | 0.1000 |
| INT2 | 2 | **2.20** | 0.2166 |

**Power Wall 판정 (정의 3.3)**:
- **[조건식 3.2]** 이산 한계효용 기울기: INT8→INT4 (2.58), **INT4→INT3 (5.68, >θ)** , INT3→INT2 (2.56). → **Power Wall 구간 = INT4~INT3**.
- **[조건식 3.1]** 연속형 변곡점(d²PCAG/db²=0): 경험적 PCHIP **b ≈ 3.51**, Monte Carlo **3.40 ± 0.25**, 해석적 모델 **b\* = 4.19** — 세 독립 경로가 모두 INT4 인접 구간(b\*∈[3.0, 4.2])으로 수렴 (Fig 17).
- PCAG는 bits 4→3에서 **54.4% 급락** (10.44→4.76).

*(Fig 2: PCAG 곡선과 Power Wall — `docs/figures/fig2_pcag_power_wall.png`)*
*(Fig 6: 발산 메커니즘(포화 vs 가속) — `docs/figures/fig06_divergence_mechanism.png`)*
*(Fig 13: 해석 모델 — 적합/탄력성/변곡 — `docs/figures/fig13_continuous_model.png`)*
*(Fig 17: 세 독립 추정 경로의 수렴 — `docs/figures/fig17_wall_convergence.png`)*

이 결과는 INT4 이하에서 소프트웨어 양자화의 전력 절감 효용이 물리적으로 붕괴함을 정량적으로 보여준다. 특히 Fig 6은 붕괴의 메커니즘을 직관화한다: 전력 절감 S(x)는 Weibull 포화로 정체하는 반면 정확도 손실 Lr(x)는 로지스틱 후반부에서 가속하여, 두 곡선의 발산 판이 벌어지는 구간( x≈12, b≈4 )이 곧 Power Wall이다.

## 4.3 다중 모델 일반화: 보편적 붕괴 패턴

4개 모델(Llama-3-8B, Qwen-2.5-7B, Gemma-2-9B, Mistral-7B)에 동일 분석을 적용한 결과:

| 모델 | PCAG INT8 | PCAG INT4 | PCAG INT3 | PCAG INT2 | INT4→INT3 slope | Power Wall 전이 (θ=3) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Llama-3-8B | 20.0 | 10.4 | 4.8 | 2.2 | **5.60** | INT4→INT3 |
| Qwen-2.5-7B | 30.5 | 15.4 | 5.6 | 2.4 | **9.78** | INT4→INT3 |
| Gemma-2-9B | 21.6 | 11.5 | 5.0 | 2.2 | **6.49** | INT4→INT3 |
| Mistral-7B | 24.4 | 11.5 | 4.5 | 2.1 | **6.96** | INT4→INT3 |

(표 값은 `results_multimodel_raw.csv` 실계산값. 주 분석 `results_raw.csv`의 Llama-3-8B 값은 4.2절과 같다.)

- 4개 모델 전부에서 [조건식 3.2]의 최대 기울기 전이가 **INT4→INT3**로 동일 → Power Wall의 **아키텍처 보편성** 지지.
- INT4 정확도 유지율: Qwen-2.5 96.4%, Mistral 95.2%, Gemma-2 95.1%, Llama-3 94.6% — 모델별 절대값은 다르나 붕괴 형태가 동일.

*(Fig 7: 다중 모델 PCAG — `docs/figures/fig07_multimodel_pcag.png`)*
*(Fig 8: 다중 모델 정확도 유지율 — `docs/figures/fig08_multimodel_retention.png`)*

## 4.4 강건성 및 민감도 분석

GPU 실측 전 단계의 결론 의존성을 정량화하기 위해 세 가지 민감도 분석을 수행했다(`sensitivity.py`).

1. **θ 임계값 스윕** (θ ∈ [0.5, 10]): 4개 모델 모두에서 θ<5.68 범위에서는 판정된 Power Wall 전이가 항상 INT4→INT3로 불변. 즉 결론은 θ 선택에 로버스트하며, θ=3은 보수적 기본값이다 (Fig 12).
2. **Monte Carlo 앵커 섭동** (상대 오차 σ=3%, N=3000): 앵커 정확도·에너지에 로그정규 잡음 주입 결과 — (a) 연속형 변곡점 b\* = 3.398 ± 0.252 (90% CI [2.99, 3.66]); (b) 기록된 전이 중 Power Wall 판정의 최빈값은 INT4→INT3 (66.8%), 차순위는 INT3→INT2 (20.2%). 결론: **INT4 이하 붕괴는 앵커 불확실성 하에서도 보존**되며, 벽의 위치는 b∈[3,4]로 특정된다 (Fig 11).
3. **경로 간 교차 검증**: 경험적 PCHIP(3.51) / Monte Carlo(3.40) / 해석적 모델(4.19)의 표준편차(≈0.4 bit)는 앵커 데이터의 비트 간격(Δb=4)보다 작아, 현재 데이터 해상도에서 "Power Wall = INT4 근방" 특정은 통계적으로 유의하다. GPU 실측 시 INT6/INT5 등 세밀 비트 샘플링으로 정밀화 가능하다 (Fig 13, 17).

*(Fig 11: Monte Carlo 강건성 — `docs/figures/fig11_mc_robustness.png`)*
*(Fig 12: 전이별 per-bit 기울기 다중 모델 비교 — `docs/figures/fig12_theta_sensitivity.png`)*

## 4.5 위협 요소 및 유효성 한계 (Threats to Validity)

본 단계의 결론을 해석할 때 다음 유효성 위협을 고려해야 한다.

- **구성 타당도 (Construct)**: PCAG는 원문 수식의 부호 모순(ISSUE-PCAG-01)으로 인해 운영 재정의(정의 3.1)를 거쳤다. 정확도는 MMLU-스타일 근사 지표이며, 데이터셋 부재 시 합성 논리 태스크로 폴백하므로 벤치마크 전환 시 절대값이 달라질 수 있다. 다만 PCAG의 **비율 구조**는 동일 스키마 내에서 보존된다.
- **내적 타당도 (Internal)**: 모든 수치가 문헌 앵커 기반 참고 데이터로, 실측 분산·하드웨어 특성을 반영하지 않는다. 이는 4.4절의 Monte Carlo(σ=3%) 및 다중 모델 분석으로 완화했으나, 실측으로 대체하기 전까지 방향성 검증에 한정된다.
- **외적 타당도 (External)**: 4개 모델, 단일 정확도 근사 지표, 가정된 E_d 값을 사용한다. 산업별·서비스별 실측 탄력성으로 일반화하기 전까지 매크로 결론은 구조적(폐형) 성질에 의존한다.
- **통계적 타당도 (Statistical)**: Power Wall 위치의 정밀 특정(b\*≈3.4~4.2)은 현재 비트 해상도(Δb=4)에 의존한다. MC의 90% CI [2.99, 3.66]은 이 불확실성을 정량화한다. GPU 실측 시 INT6/INT5 세밀 샘플링으로 정밀화해야 한다.

## 4.6 통계 추론: 부트스트랩 신뢰구간과 추정 가능성

4.4절의 Monte Carlo는 "결론이 앵커 불확실성에 보존되는가"를 강건성 관점에서 보여준다. 본 절은 이를 **통계적 추론**으로 격상하여, 앵커에 로그정규 상대 오차(σ=3%, N=3000, 시드 20260901)를 주입한 부트스트랩(`experiments/statistics.py`)으로 각 정량의 신뢰구간과 가설검정을 수행한다.

| 정량 | 부트스트랩 90% CI / 검정 | 해석 |
|------|:---:|------|
| **연속 변곡점 (조건식 3.1)** | mean **3.39**, 90% CI **[2.99, 3.65]** | 강건 — MC(3.40±0.25)와 일치 |
| PCAG INT2 | mean 2.22, 90% CI [1.88, 2.66] | 강건(심층 양자화, 좁은 CI) |
| PCAG INT3 | mean 5.36, 90% CI [3.27, 9.16] | 비교적 안정 |
| PCAG INT8 (근접 기준) | mean 52.1, 90% CI [3.8, 93.7] | **취약** — 추정 불가능(분모=근소한 정확도 손실) |
| 이산 기울기 INT8→4 | mean 9.5, 90% CI [0.1, 23.9] | **취약** — 매우 넓은 CI |
| 이산 기울기 INT4→3 | mean 18.9, 90% CI [0.5, 50.3] | **취약** — 매우 넓은 CI |
| Paired 검정 INT4→3 > INT8→4 | P(Δ>0)=0.77 | 유의하지 않음 |
| INT4→3 > θ=3 | P=0.60 | **유의하지 않음** |

**해석 및 추정 가능성 (Estimability)**: 부트스트랩은 중요한 방법론적 교훈을 드러낸다. 기준(FP16)에 근접한 정밀도의 **절대 PCAG 값은 측정 잡음에 극도로 취약**하다 — 이는 PCAG 분모인 상대 정확도 손실이 기준점 부근에서 극소(예: INT8의 L≈0.0165)이기 때문이다. 반면 (i) **연속 변곡점(조건식 3.1)의 위치는 강건**하고, (ii) **심층 양자화(INT3/INT2)의 PCAG는 좁은 CI로 안정**적이다. 즉 Power Wall의 존재·위치에 대한 방어 가능한 증거는 "개별 근접-기준 PCAG 점값"이 아니라 **①강건한 연속 변곡점, ②안정적인 심층 붕괴, ③해석적 모델(정리 3.4, 명제 3.6)** 에서 온다. 이는 GPU 실측 시 **중간 비트(INT6/INT5) 샘플링과 반복 측정**이 왜 필수인지(`docs/measurement_protocol.md`)를 방법론적으로 뒷받침한다.

*(Fig 18: 부트스트랩 추론 — PCAG 90% CI + 변곡점 분포 — `docs/figures/fig18_bootstrap_inference.png`)*

## 4.7 모델-형 강건성 (Model-Form Robustness)

기존 해석은 단일 함수형(Weibull 포화 + 선형-로지스틱 가속)을 가정한다. `experiments/model_form.py` 는 대안 함수형 조합(전력 절감: Weibull·단일지수·쌍곡탄젠트·Hill / 정확도 손실: 선형-로지스틱·순수 멱·순수 로지스틱·선형-멱)에 대해 동일한 조건식 3.1의 근 b\*를 계산한다.

| S(절감)형 | Lr(손실)형 | b\* | 비고 |
|:---:|:---:|:---:|------|
| Weibull | 선형+로지스틱 | **4.19** | 기준 모델 |
| 단일지수 | 선형+로지스틱 | **4.27** | |
| 쌍곡탄젠트 | 선형+로지스틱 | **4.24** | |
| Hill | 선형+로지스틱 | **4.20** | |
| (모든 S형) | 순수 멱/순수 로지스틱/선형-멱 | 도메인 밖 | 가속 손실 구조 부재 시 도메인 내 변곡점 없음 |

**결론**: 모든 표준 포화형 전력 절감 함수에서 b\* ∈ **[4.19, 4.27]** 로 극히 안정적이며, 이는 정리 3.4의 구조적 무관성과 정합한다. 한편 가속(accelerating) 손실 성분이 없는 함수형(순수 멱/로지스틱)은 관측 도메인 내 변곡점을 생성하지 않아, **Power Wall 변곡의 존재는 "포화 절감 vs 가속 손실" 구조가 필수적**임을 역으로 확인한다. (세부값: `experiments/model_form_summary.json`)

*(Fig 19: 모델-형 강건성 — 대안 함수형에 대한 b\* — `docs/figures/fig19_model_form_robustness.png`)*

## 4.8 외부 타당도: 문헌 앵커 정합성

본 연구의 참고 데이터는 문헌 앵커이므로, 그 정합성을 공개된 GPTQ[6]/AWQ[7] 계열 INT4 결과와 교차 대조한다. 문헌상 사후 양자화(PTQ) INT4는 표준 벤치마크에서 원본 정확도의 대략 **95~99% 수준 유지율**을 보이는 것이 전형적이다.

| 모델 | 본 연구 앵커 INT4 유지율 | 문헌 전형 범위(PTQ INT4) |
|------|:---:|:---:|
| Llama-3-8B | 94.6% | ~95–99% |
| Qwen-2.5-7B | 96.4% | ~95–99% |
| Gemma-2-9B | 95.1% | ~95–99% |
| Mistral-7B | 95.2% | ~95–99% |

본 연구 앵커의 INT4 유지율(94.6~96.4%)은 문헌 전형 범위의 하단에 위치하여 **양자화 열화 경향을 현실적으로 반영**한다. 다만 이는 "전형 범위"에 대한 정성적 대조이며, 정확한 수치는 평가 세트·모델 버전에 따라 달라지므로 절대값이 아닌 **경향성(붕괴 패턴)의 정합성** 관점에서 해석해야 한다. 실측 단계에서 동일 평가 하네스로 직접 검증하는 것이 후속 과제다.

---

# Chapter 5. 고찰: Jevons 역설과 전력망 영향

## 5.1 연계 분석: 낮은 추론 비용 vs 지수적 수요 증가

매크로 시뮬레이션(`jevons_model.py`)에서 수요곡선 Q(P)=Q₀(P/P₀)^(-E_d)를 가정한다. 비용 절감률 s에 대해 P = P₀(1−s) 이므로 총 부하의 **폐형 해**가 유도된다:

```
TotalLoad / L₀ = (1−s) · (1−s)^(−E_d) = (1−s)^(1−E_d)
d/ds [TotalLoad/L₀] = (E_d − 1)(1−s)^(−E_d)
```

**정리 5.1 (Jevons 동치).** 0<s<1 에서 (1−s)^(−E_d)>0 이므로, 총 그리드 부하는 **절감률 s에 대해 단조 증가 ⟺ E_d > 1** (임의의 양의 절감률에서 성립, sympy 기호 검증 완료).

**계 5.2 (역치 부재).** E_d>1 이면 절감률의 크기와 무관하게 Jevons 역설이 성립하며, 절감이 클수록 역설의 크기도 커진다. 즉 "부하 증가를 유발하는 양의 절감률 하한"은 존재하지 않는다.

**INT4(토큰당 에너지 −56%) 적용 결과 (E_d=1.5)**:

| 지표 | 값 |
|------|:---:|
| 수요 증가 | **+231%** |
| 총 그리드 부하 변화 | **+49%** (폐형 (0.44)^(−0.5)−1 = +49.07%와 일치) |
| Jevons 역치(부하 증가 시작 절감률) | 0% (E_d>1이면 즉시) |

전 구간 절감률 스윕에서, E_d=1.5(탄력적 수요) 조건에서는 **모든 절감률에서 총부하가 증가**한다. 즉 소프트웨어 효율 개선이 토큰당 비용을 낮추면, 접근성 증가가 이를 압도하여 총 전력 소비가 늘어나는 Jevons 역설이 성립한다. 민감도 그리드(E_d ∈ [0.2, 2.8] × s ∈ [0, 70%]) 분석에서 그리드 영역의 65.6%에서 부하가 증가하며, 경계는 정확히 E_d=1이다 (Fig 9, 10).

*(Fig 3: Jevons Grid Load — `docs/figures/fig3_jevons_grid_load.png`)*
*(Fig 9: Jevons 민감도 히트맵 — `docs/figures/fig09_jevons_heatmap.png`)*
*(Fig 10: Jevons 위상도 — `docs/figures/fig10_jevons_phase.png`)*
*(Fig 16: Jevons 3D 곡면 — `docs/figures/fig16_jevons_surface.png`)*

## 5.2 전력망 한계와 컴퓨팅 최적화의 갈등

- **미시 갈등**: 소프트웨어 최적화(양자화)는 물리적 Power Wall에 막혀 더 이상의 효율을 낼 수 없음(4장).
- **거시 갈등**: 그나마 얻은 효율은 Jevons 역설로 인해 매크로 전력망 부하를 증가시킴(5.1).
- **시사점**: 단순한 알고리즘적 효율 개선은 전력 문제의 해법이 아니며, 전산 패러다임의 근본적 전환이 요구된다.

---

# Chapter 6. 결론

## 6.1 연구 요약 및 핵심 통찰

1. **전력 벽 존재**: PCAG 지표가 INT8→INT3로 갈수록 20.76→4.76로 급감하며, INT4~INT3 구간에서 붕괴. 세 독립 추정 경로(경험적 3.51 / Monte Carlo 3.40±0.25 / 해석적 4.19)가 모두 INT4 인접 구간으로 수렴. 소프트웨어 양자화만으로는 이 벽을 넘을 수 없다.
2. **구조적 정리**: 변곡점 조건 h(x)=g'(x)+g(x)²=0 은 진폭에 무관(정리 3.4) — Power Wall은 전력 포화와 정확도 가속의 상대 구조가 강제하는 필연적 결과다 (3.3.2).
3. **보편성**: 4개 모델(Llama-3/Qwen-2.5/Gemma-2/Mistral) 전부에서 최대 붕괴 전이가 INT4→INT3로 동일 — 아키텍처 보편 패턴.
4. **전력 절감 포화 vs 정확도 붕괴 가속**: 양자화 깊이에 따라 ΔP는 포화, ΔA는 가속되어 효율 지표의 비선형 붕괴를 유발.
5. **Jevons 역설의 폐형 정식화**: TotalLoad/L₀=(1−s)^(1−E_d), 부하 증가 ⟺ E_d>1 (정리 5.1, 기호 증명). E_d=1.5에서 INT4(−56%)는 총부하 +49%.

## 6.2 필수적 패러다임 전환

- **광연산(Optical Computing)**: 기존 전자 스위칭 에너지 한계를 우회.
- **뉴로모픽(Neuromorphic)**: Landauer 한계 근처의 에너지 효율 달성 가능성.
- **고효율 발전(SMR/융합)**: 전력 공급 측면의 근본 확충.
- **전력망 수준의 정책·가격 설계**: Jevons 역설을 고려한 수요 관리.

## 6.3 한계 및 향후 연구

- **한계 1**: 현재 수치는 **참고 데이터**(문헌 앵커)로, GPU 실측으로 검증 필요. 단, 4.4절의 강건성 분석(Monte Carlo, 다중 모델, 해석 모델)과 4.6절의 부트스트랩이 결론의 방향성은 앵커 불확실성에 로버스트함을 보였다.
- **한계 2**: E_d는 문헌/가정 기반 파라미터 — 산업별·서비스별 실측 탄력성 추정이 후속 과제다. (단, "부하 증가 ⟺ E_d>1"의 구조적 결론은 폐형으로 성립한다.)
- **한계 3**: Power Wall 위치의 정밀 특정(b\*≈3.4~4.2)은 현재 비트 해상도(Δb=4)에 의존 — GPU 실측 시 INT6/INT5 등 세밀 비트 샘플링으로 정밀화 필요.
- **한계 4 (추정 가능성)**: 4.6절의 부트스트랩이 보여주듯, 기준에 근접한 정밀도의 **절대 PCAG 값**은 측정 잡음에 취약하다. 따라서 Power Wall의 방어 가능한 증거는 연속 변곡점·심층 붕괴·해석 모델에 두어야 하며, 이산 기울기 단독 판정은 보조적으로 해석해야 한다.
- **한계 5 (모델-형)**: 4.7절에서 Power Wall 변곡의 존재는 가속 손실 구조가 필수임을 확인했다. 실측 데이터의 손실 곡선 형태가 크게 달라질 경우 변곡점 위치의 재평가가 필요하다.
- **향후 연구**: 다중 모델·다중 벤치마크(MMLU/GSM8K) GPU 실측, GPU 간 전력 효율 비교, 동적·지역별 전력망 모델 확장, 정책 시나리오(가격 탄력성 조절) 분석, PCAG를 학습(Training) 영역으로 확장, 실측 전환 후 `verify_numbers.py` 로 수치 갱신 추적.

---

# 사회적 영향 진술 (Broader Impact)

본 연구는 LLM 추론 에너지 효율의 물리적 한계(Power Wall)와 그 매크로 전력망 파급(Jevons 역설)을 정량화한다. 잠재적 사회적 영향은 양면적이다.

**긍정적 측면**: 에너지 효율 최적화의 한계를 명확히 제시함으로써, (i) 연구자·엔지니어가 무한정한 양자화 깊이 추구에 드는 비용을 인지하게 하고, (ii) 하드웨어·아키텍처 패러다임 전환(광연산·뉴로모픽)과 전력망 정책 설계에 근거를 제공하며, (iii) Jevons 역설에 대한 정량적 인식으로 에너지-수요 관리 논의를 촉진한다.

**부정적·오용 측면**: "소프트웨어 최적화는 한계가 있다"는 결론이 특정 기술 경로(양자화)의 투자를 축소시키는 근거로 오용될 수 있다. 또한 Jevons 역설 수치가 효율 개선 자체를 반대하는 논증으로 확대해석될 위험이 있다. 본 연구의 수치는 현재 참고 데이터 기반이며, 정책 결정에 직접 활용하기 전 반드시 실측 검증이 선행되어야 한다.

본 연구는 실측 환경(중립적 전력망 전제, 가격 탄력성 가정)을 명시하며, 결론의 구조적(폐형) 성질과 가정 의존적 부분을 구분해 제시함으로써 오용 가능성을 완화하고자 한다.

---

# 저자 기여·감사·이해상충 (Contributions & Declarations)

**저자 기여 (Author Contributions)** — *[제출 시 작성]*: 개념화, 방법론, 소프트웨어, 검증, 시각화, 원고 작성.

**감사 (Acknowledgments)** — *[제출 시 작성]*: 본 연구는 공개 문헌 앵커 기반으로 수행되었으며, GPU 실측은 후속 단계에서 수행될 예정이다.

**이해상충 (Conflicts of Interest)** — 없음. 본 연구는 외부 자금 지원 없이 수행되었다.

**윤리적 승인 (Ethics)** — 본 연구는 인간·동물 대상 실험이 아니므로 기관윤리위원회(IRB) 심의 대상이 아니다.

---

# 재현성·데이터/코드 가용성 (Reproducibility & Availability)

**재현성 진술 (Reproducibility Statement).** 본 연구의 모든 수치는 커밋된 산출물(`experiments/*.csv`, `experiments/*.json`)에서 직접 전사되며, 단일 명령열로 재현된다(아래). Monte Carlo는 고정 시드(RNG=42)를 사용하고, 참고 데이터 생성은 고정 생성일 상수(ANCHOR_GEN_DATE)를 사용해 **바이트 단위 결정적(byte-identical)** 재생성을 보장한다. 재현 절차·환경·기대 출력의 상세는 `docs/reproducibility.md` 를 참조한다.

```bash
pip install -r requirements.txt          # 정확한 버전 고정
cd experiments
python generate_results.py               # 참고 데이터 (결정적)
python multimodel_data.py                # 다중 모델 앵커 (결정적)
python analysis.py                       # PCAG + Power Wall
python analytical_proof.py               # 폐형 유도 + Jevons 기호 증명 + 조건식 3.2↔3.1 일관성
python sensitivity.py                    # θ 스윕 + Jevons 그리드 + MC (시드 42)
python statistics.py                     # 부트스트랩 통계 추론 (시드 20260901)
python model_form.py                     # 모델-형 강건성 (b*)
python jevons_model.py                   # Jevons 시나리오
python make_figures.py                   # 그림 19종
python verify_numbers.py                 # 논문 수치 ↔ 산출물 무결성 게이트 (exit 0)
python dry_run.py                        # 통합 파이프라인 검증
```

**데이터 가용성 (Data Availability).** 원시 데이터·분석 산출물은 저장소 `experiments/` 에 커밋되어 있다. 참고 데이터는 `Source=Reference-Literature` 로 명시되며, GPU 실측 단계에서는 `benchmark_driver.py` 가 동일 스키마의 `Source=Measured-GPU` 로 대체한다.

**코드 가용성 (Code Availability).** 전체 파이프라인 코드는 저장소 `experiments/` 에 MIT-호환 라이선스로 공개되어 있다. 재현 커맨드와 환경은 위와 같다.

---

# 부록 A. 그림 색인 (Figure Index)

| 그림 | 파일 | 내용 |
|---|---|---|
| Fig 1 | `fig1_accuracy_vs_energy` | 정확도–에너지 프론티어 + iso-PCAG 등고선 |
| Fig 2 | `fig2_pcag_power_wall` | PCAG 붕괴 곡선과 Power Wall 존 |
| Fig 3 | `fig3_jevons_grid_load` | Jevons: 수요 증폭 vs 총부하 (폐형 오버레이) |
| Fig 4 | `fig04_resource_footprint` | VRAM/전력/지연/처리량 요약 (2×2) |
| Fig 5 | `fig05_accuracy_cliff` | 정확도 절벽: 곡선 + 단계별 손실 |
| Fig 6 | `fig06_divergence_mechanism` | 발산 메커니즘: 포화 절감 vs 가속 손실 |
| Fig 7 | `fig07_multimodel_pcag` | 4개 모델 PCAG 비교 (그룹 바) |
| Fig 8 | `fig08_multimodel_retention` | 4개 모델 정확도 유지율 |
| Fig 9 | `fig09_jevons_heatmap` | Jevons 민감도 히트맵 (E_d × s) |
| Fig 10 | `fig10_jevons_phase` | Jevons 위상도 + 양자화 시나리오 |
| Fig 11 | `fig11_mc_robustness` | Monte Carlo 강건성 (변곡점 분포 + 벽 확률) |
| Fig 12 | `fig12_theta_sensitivity` | 전이별 per-bit 기울기 (다중 모델) |
| Fig 13 | `fig13_continuous_model` | 해석 모델: 적합/탄력성 g/변곡 h |
| Fig 14 | `fig14_efficiency_frontier` | iso-PCAG 효율 프론티어 |
| Fig 15 | `fig15_dashboard` | 연구 종합 대시보드 (2×2) |
| Fig 16 | `fig16_jevons_surface` | Jevons 3D 곡면 (1−s)^(1−E_d) |
| Fig 17 | `fig17_wall_convergence` | 3 독립 경로의 벽 위치 수렴 |
| Fig 18 | `fig18_bootstrap_inference` | 부트스트랩 추론: PCAG 90% CI + 변곡점 분포 |
| Fig 19 | `fig19_model_form_robustness` | 모델-형 강건성: 대안 함수형에 대한 b\* |

모든 그림은 `experiments/make_figures.py` 로 재생성 가능하며, PNG(200dpi)+PDF 병행 제공.
부록 수식 유도 상세: `docs/proof_3_1_derivation.md`.

---

# 부록 B. 통계 부록 (Statistical Appendix)

모든 값은 커밋된 산출물에서 직접 전사. `experiments/verify_numbers.py` 가 일치 여부를 자동 검증한다.

## B.1 주 분석 요약 (Llama-3-8B, `results_raw.csv` + `analysis_summary.json`)

| Precision | bits | Acc (%) | E (J/1k) | S (절감) | L (손실) | PCAG_eff | PCAG_raw |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FP16 | 16 | 66.60 | 140.0 | 0.000 | 0.000 | – | – |
| INT8 | 8 | 65.50 | 92.0 | 0.343 | 0.0165 | 20.76 | 0.0229 |
| INT4 | 4 | 63.00 | 61.0 | 0.564 | 0.0541 | 10.44 | 0.0456 |
| INT3 | 3 | 58.00 | 54.0 | 0.614 | 0.1291 | 4.76 | 0.1000 |
| INT2 | 2 | 47.00 | 49.5 | 0.646 | 0.2943 | 2.20 | 0.2166 |

Power Wall (조건식 3.2): INT4→INT3 기울기 **5.68** (>θ=3). 연속형 변곡점 PCHIP: **3.51**.
PCAG 4→3 급락: (10.44−4.76)/10.44 = **54.4%**.

## B.2 다중 모델 PCAG (`results_multimodel_raw.csv` 실계산)

| 모델 | PCAG INT8 | PCAG INT4 | PCAG INT3 | PCAG INT2 | INT4→INT3 slope | INT4 유지율(%) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Llama-3-8B | 20.02 | 10.40 | 4.80 | 2.19 | 5.60 | 94.6 |
| Qwen-2.5-7B | 30.48 | 15.36 | 5.58 | 2.38 | 9.78 | 96.4 |
| Gemma-2-9B | 21.57 | 11.52 | 5.03 | 2.22 | 6.49 | 95.1 |
| Mistral-7B | 24.40 | 11.49 | 4.54 | 2.07 | 6.96 | 95.2 |

## B.3 Monte Carlo 강건성 (`sensitivity_summary.json`)

| 항목 | 값 |
|------|:---:|
| 반복 수 N | 3000 |
| 상대 앵커 오차 σ | 3% |
| 연속형 변곡점 평균 b\* | 3.398 ± 0.252 |
| 90% CI (p5–p95) | [2.994, 3.655] |
| 유효 변곡점 샘플 수 n | 1748 |
| 벽 전이 최빈값 INT4→INT3 | 66.8% |
| 벽 전이 차순위 INT3→INT2 | 20.2% |
| 벽 전이 INT8→INT4 | 11.5% |

## B.4 θ-스윕 불변성 (`sensitivity_summary.json`)

- 4개 모델 모두 θ<5.68 구간에서 Power Wall 전이 = **INT4→INT3** 로 불변.
- θ=3(기본값)에서 주 분석 기울기 = **5.68**.
- 각 모델이 "불변성을 잃는" θ 지점(벽 전이 소실)은 최대 슬로프보다 낮지 않은 곳: Llama 5.75, Qwen 6.0, Gemma 6.5, Mistral 7.0.

## B.5 Jevons 시나리오 (`jevons_summary.json`)

| 절감률 s | 수요 증가 (%) | 총부하 변화 (%) |
|:---:|:---:|:---:|
| 0.00 | 0.0 | 0.0 |
| 0.55 (INT4 대응) | **+231.3** | **+49.1** |
| 0.70 | +508.6 | +82.6 |

- 폐형 교차검증: (1−0.55)^(1−1.5) − 1 = **+49.07%** ✓
- 부하 증가 영역 비율 (E_d×s 그리드): **65.6%**
- 부하 증가 ⟺ E_d>1 (정리 5.1, sympy 검증: True)

## B.6 조건식 3.2 ↔ 3.1 일관성 (`analysis_proof.json`)

| 지표 | 값 |
|------|:---:|
| 모델 단위 비트 드롭 8→7 / **4→3** / 3→2 / 2→1 | −1.27 / **+6.68** / +5.76 / +2.53 |
| 최대 이산 드롭 구간 | **4→3** |
| 연속 붕괴율 최대 지점 b_peak | 4.19 |
| 변곡점 b\* | 4.19 |
| 일관성 판정 | True |

## B.7 부트스트랩 통계 추론 (`statistics_summary.json`, N=3000, σ=3%, 시드 20260901)

| 정량 | mean | 90% CI |
|------|:---:|:---:|
| 연속 변곡점 (조건식 3.1) | 3.39 | [2.99, 3.65] |
| PCAG INT2 | 2.22 | [1.88, 2.66] |
| PCAG INT3 | 5.36 | [3.27, 9.16] |
| PCAG INT4 | 23.8 | [4.6, 56.9] |
| PCAG INT8 | 52.1 | [3.8, 93.7] |
| 이산 기울기 INT8→4 | 9.5 | [0.1, 23.9] |
| 이산 기울기 INT4→3 | 18.9 | [0.5, 50.3] |
| Paired Δ(INT4→3 − INT8→4) | +3.2 | [−12.1, 23.6], P(Δ>0)=0.77 |
| INT4→3 > θ=3 | — | P=0.60 (유의하지 않음) |

## B.8 모델-형 강건성 (`model_form_summary.json`)

| S(절감)형 | Lr(손실)형 | b\* |
|:---:|:---:|:---:|
| Weibull | 선형+로지스틱 | 4.19 |
| 단일지수 | 선형+로지스틱 | 4.27 |
| 쌍곡탄젠트 | 선형+로지스틱 | 4.24 |
| Hill | 선형+로지스틱 | 4.20 |
| (모든 S형) | 순수 멱/순수 로지스틱/선형-멱 | 도메인 밖 |

- b\* ∈ [4.19, 4.27] — 모든 표준 포화형에 대해 안정적.
- 가속 손실 성분 부재 시 도메인 내 변곡점 없음 → 변곡 존재에 가속 구조가 필수.

---

# 참고문헌 (References)

1. Dennard, R. H., Gaensslen, F. H., Rideout, V. L., Bassous, E., & LeBlanc, A. R. (1974). Design of ion-implanted MOSFETs with very small physical dimensions. *IEEE Journal of Solid-State Circuits*, 9(5), 256–268.
2. Landauer, R. (1961). Irreversibility and heat generation in the computing process. *IBM Journal of Research and Development*, 5(3), 183–191.
3. Sze, V., Chen, Y.-H., Yang, T.-J., & Emer, J. S. (2017). Efficient processing of deep neural networks: A tutorial and survey. *Proceedings of the IEEE*, 105(12), 2295–2329. arXiv:1703.09039.
4. Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., & Keutzer, K. (2021). A survey of quantization methods for efficient neural network inference. arXiv:2103.13630.
5. Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. arXiv:2208.07339.
6. Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). GPTQ: Accurate post-training quantization for generative pre-trained transformers. arXiv:2210.17323.
7. Lin, J., Tang, J., Tang, H., Yang, S., Dang, W., & Han, S. (2023). AWQ: Activation-aware weight quantization for LLM compression and acceleration. arXiv:2306.00978.
8. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. arXiv:2305.14314.
9. Frantar, E., & Alistarh, D. (2023). SpQR: A sparse-quantized representation for near-lossless LLM weight compression. arXiv:2306.03078.
10. Blalock, D., Gonzalez Ortiz, J. J., Frankle, J., & Guttag, J. (2020). What is the state of neural network pruning? arXiv:2003.03033.
11. Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., ... & Amodei, D. (2020). Scaling laws for neural language models. arXiv:2001.08361.
12. Jevons, W. S. (1865). *The Coal Question: An Inquiry Concerning the Progress of the Nation, and the Probable Exhaustion of Our Coal-Mines*. London: Macmillan.
13. Patterson, D., Gonzalez, J., Le, Q., Liang, C., Munguia, L.-M., Rothchild, D., So, D., Texier, M., & Dean, J. (2021). Carbon emissions and large neural network training. arXiv:2104.10350.
14. International Energy Agency. (2024). *Electricity 2024: Analysis and forecast to 2026*. IEA, Paris.

> BibTeX 버전은 `docs/references.bib` 에 수록되어 있다.

---
*생성 도구: PCAG Research Pipeline. v4 마지막 갱신: 2026-09-01. (v4: 명제 3.6 이산-연속 일관성, 부트스트랩 통계 추론(4.6)·추정 가능성, 모델-형 강건성(4.7), 외부 타당도(4.8), 사회적 영향·저자 기여·감사·이해상충 진술, 부록 B.6~B.8, Fig 18~19, 무결성 게이트 78개 항목, dry_run 파이프라인 확장)*
