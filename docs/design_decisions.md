# 연구 설계 결정 기록 (Design Decision Records, ADR)

본 문서는 연구의 방법론·도구 설계에서 내린 주요 결정과 그 **근거·대안·반대 고려·결정 시점**을
기록한다. (ADR 패턴 — 각 결정은 Why/Alternatives/Trade-offs 포함) 코드 변경 이력은
`logs/change_log.md`, 기술 장애는 `logs/troubleshooting_archive.md`, 일정 기록은
`research_journal.md` 를 각각 참고.

> 상태 표기: [Adopted] 채택 / [Superseded] 대체됨 / [Proposed] 제안

---

## DD-01: PCAG 실측을 "문헌과의 직접 비교"가 아닌 "균일 방법론 미세 비트 곡선"으로 설계

- **상태**: Adopted (2026-08-31, Phase 8)
- **결정**: FP16→INT8→INT6→INT5→INT4→INT3→INT2 의 **모든 비트폭을 한 가지 per-channel
  RTN 대칭 양자화**(lowbit.py)로 측정해, 방법론 혼합이 없는 PCAG 곡선을 실측의 1차
  산출물로 삼는다.
- **근거 (Why)**:
  1. T4 무료 환경에서 GPTQ/AWQ(보정 기반)는 VRAM·시간 부담이 크고, bnb는 INT3/INT2
     미지원(ISSUE-METHOD-01) → "문헌과 같은 방법"으로 측정할 수 없음.
  2. 방법이 섞인 곡선(예: INT8은 bnb, INT3은 RTN)은 기울기/변곡점이 방법 차이로
     왜곡되어 Power Wall 판정이 방법론 아티팩트가 됨.
  3. PCAG/Power Wall 결론은 이미 해석적으로 "진폭 무관 구조적 결과"(analytical_proof)
     로 증명됨 → 실측의 역할은 **동일 조건 하에서 그 붕괴 구조를 확인·정량화**하는 것.
- **대안 (Alternatives)**:
  - auto-gptq/autoawq 로 INT3/INT2 보정 측정 → 방법은 문헌과 일치하나, 곡선의 모든
    지점이 서로 다른 방법일 수 있음 + T4/8B 보정 비용. 기각.
  - bnb INT8/INT4 + RTN INT3/INT2 혼합 → 곡선 내 방법 불연속. 기각.
- **영향 (Trade-offs)**: 실측 절대 정확도가 문헌(GPTQ/AWQ, MMLU)과 직접 비교 불가.
  → Notes 에 `quant=uniform per-channel RTN` 명시 + 표준 방법(bnb) 검증 세트를
  별도 CSV 로 분리(DD-02)해 "방법 정합성 vs 문헌 비교 가능성"을 트레이드오프로 관리.

---

## DD-02: 양자화 방법 분리 — `--quant-method rtn/bnb` + 별도 CSV

- **상태**: Adopted (2026-08-31, Phase 8)
- **결정**: `benchmark_driver.py` 에 `--quant-method` 를 추가.
  - `rtn`(기본): 균일 RTN 곡선 → `results_multimodel_raw.csv` / `results_raw.csv`
  - `bnb`: bitsandbytes 표준 INT8/INT4/FP4 → `results_multimodel_bnb_raw.csv` (별도)
  - `--update-main`(results_raw.csv 갱신)은 **rtn 에서만** 허용.
- **근거 (Why)**: 기준 CSV 를 bnb 검증 세트가 오염시키면(ISSUE-DATA-02) 곡선의
  방법론 일관성이 깨져 Power Wall 위치가 왜곡. bnb 는 "표준 방법과의 질적 정합성 확인"
  용으로 분리 보존.
- **영향**: 곡선 CSV 와 검증 CSV 두 파일을 관리해야 함. 대신 데이터 무결성(방법 동질성)이
  파일 단위로 보장됨.

---

## DD-03: 정확도 평가를 ARC-Easy 표준 0-shot 객관식으로 전환 (합성 태스크 → 표준 벤치마크)

- **상태**: Adopted (2026-08-31, Phase 8)
- **결정**: `eval_harness.py` 기본 평가를 `arc_easy`(ai2_arc/ARC-Easy, test split,
  고정 seed 샘플, A–E 글자 정확도)로. `datasets` 미설치/로드 실패 시 내장 합성 논리
  태스크로 자동 폴백.
- **근거 (Why)**: ① 합성 태스크(산술 규칙 추종)는 문헌과 비교 불가능한 폐쇄 프록시.
  ② ARC-Easy 는 0-shot 객관식으로 T4/8B 에서 수 분 내 실행 가능(MMLU 전체는 과도).
  ③ 문헌 값과 비교 가능한 공개 표준 벤치마크.
- **대안**: MMLU(dev) 전체 → 8B × 7비트폭 × 4모델에서 시간·VRAM 부담 큼. 기각.
  GSM8K(few-shot) → 세대 평가라 시간 부담. 기각.
- **영향**: 정확도 절대치는 ARC-Easy 도메인에서의 수치가 됨 — 문헌의 MMLU-style
  수치와 **직접 비교 금지**(Notes/가이드에 명시). PCAG 의 상대 비율 구조는 그대로 유효.

---

## DD-04: 반복 측정 도입 — `--repeats N` (mean ± std)

- **상태**: Adopted (2026-08-31, Phase 8)
- **결정**: 정밀도당 `--repeats 3` 기본. 지연/처리량/전력/에너지의 mean 을 기록하고
  표준편차는 Notes(`std_*`)에 기록. 정확도는 결정적이라 1회만.
- **근거 (Why)**: 문헌 앵커는 점 추정값뿐이라 실측 분산을 담을 수 없음. 실측 연구의
  차별점은 "동일 조건 반복 → 신뢰 구간"을 제공하는 것. 또한 단일 호출 정합(ISSUE-CODE-09)
  을 반복 단위로 확장.
- **영향**: 소요 시간 ×3 (모델 1개 ≈ 35–50분). `--repeats 1` 로 축소 가능 — 표준편차
  없이 평균만 기록.

---

## DD-05: 그림 생성 데이터 구동화 — 하드코딩 앵커 제거

- **상태**: Adopted (2026-08-31, Phase 8)
- **결정**: `make_figures.py` 가 `analysis_summary.json`(power_wall, continuous_inflection)
  과 `analysis_proof.json`(b*)을 직접 읽어 그림의 벽 구간·기울기·변곡점·축 범위를 계산.
  문헌 데이터일 때는 기존 값으로 폴백.
- **근거 (Why)**: ISSUE-CODE-10 — 하드코딩된 문헌 수치(5.68, 4.19, 3.51, xlim 등)가
  실측 데이터에서 그림과 본문의 불일치를 유발. 그림은 논문의 증거물이므로 입력을
  산출물 JSON/CSV 로부터만.
- **영향**: 그림 함수 시그니처에 `asum` 인자 추가. 문헌 단계에서는 시각적 회귀가
  최소화되도록 폴백 유지.

---

## DD-06: INT6/INT5 세밀 비트폭 지원 (로드맵 "세밀 비트 샘플링" 달성)

- **상태**: Adopted (2026-08-31, Phase 8)
- **결정**: `lowbit.py` 를 2/3/4/5/6/8-bit 로 확장, analysis/sensitivity/
  analytical_proof 의 `PREC_BITS`/비트폭 지도에 INT6(6)/INT5(5)/FP4(4) 추가.
- **근거 (Why)**: README 로드맵의 미완 항목. Power Wall(b≈4 근처)을 1bit 해상도로
  정밀화하면 기존 Δb=4 해상도(FP16→INT8→INT4)의 모델 불확실성(±0.4 bit)을 줄일 수 있음.
- **영향**: 곡선 샘플 5→7개. bnb 는 INT6/INT5 미지원이라 RTN 전용.

---

## DD-07: Colab 실측 인프라 원칙 — 모델별 분할 + 단위 즉시 저장 + 재개

- **상태**: Adopted (2026-08-31, Phase 7)
- **결정**: (1) 세션당 1모델 실행(`--models`). (2) 정밀도별 즉시 CSV upsert +
  체크포인트 JSON. (3) `--resume` 재개. (4) `--drive-dir` 로 Drive 즉시 미러링.
- **근거 (Why)**: 무료 Colab 세션 타임아웃(ISSUE-ENV-04)에서 전체 유실을 방지하고,
  "끊겨도 Drive 가 기록원" 이 되는 구조.
- **영향**: 파일이 많아지지만(CSV + checkpoint) 재현·재개가 안전. 노트북 Run all
  재실행이 곧 복구 절차.

---

## DD-08: 전력 계측 폴백 체인 (PyNVML → nvidia-smi → torch.cuda TDP 추정)

- **상태**: Adopted (2026-08-31, Phase 7)
- **결정**: `telemetry.PowerMeter` 가 ① PyNVML powerDraw ② nvidia-smi CLI ③
  `estimate()`(torch.cuda GPU 이름 → TDP 상수표, E=P×t) 순으로 폴백. 추정 사용 시
  Notes 에 `power=torch.cuda TDP estimate` 명시.
- **근거 (Why)**: 가상화 GPU(Colab 등)는 power.read 가 차단됨(ISSUE-ENV-05). 실측 불가를
  "기록 불가"로 두지 않고, **추정임을 라벨링**해 PCAG 상대 지표로 활용.
- **영향**: 절대 전력·에너지는 추정치일 수 있음 — 논문 방법론 절에 실측/추정 구분을
  명시해야 함. 실전력이 필요한 결론(절대 J)에는 부적합.