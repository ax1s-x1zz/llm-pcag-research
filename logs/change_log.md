# 변경 로그 (Change Log)

연구 원고·파이프라인의 버전 이력. 각 항목은 [날짜, 문서/코드, 변경 내용]을 기록한다.
(연구 과정의 의사결정·장애는 `research_journal.md`, 기술 회고는 `logs/troubleshooting_archive.md` 참조.)

---

## v4 — 2026-09-01 (main, pre-Colab 원래 연구 방향)

**문서**
- `docs/main_paper.md` v4:
  - 명제 3.6 (이산-연속 일관성, 조건식 3.2 ↔ 3.1) 추가 + 수치 검증 인용.
  - 4.6 통계 추론: 부트스트랩 신뢰구간 + 추정 가능성(estimability) 분석.
  - 4.7 모델-형 강건성 (대안 함수형에 대한 b\*).
  - 4.8 외부 타당도: 문헌(GPTQ/AWQ) 대비 앵커 INT4 유지율 정합성.
  - 사회적 영향(Broader Impact), 저자 기여·감사·이해상충·윤리 진술.
  - 한계 4(추정 가능성)·한계 5(모델-형) 추가, 재현 명령열에 신규 스크립트 반영.
  - 부록 B.6~B.8 (일관성·부트스트랩·모델-형 표).
- `docs/proof_3_1_derivation.md`: 5절 "조건식 3.2 ↔ 3.1 일관성" 추가.
- `docs/reproducibility.md`: 재현 명령열·골든 해시·검증 매트릭스 갱신 (78개 항목).

**코드**
- `experiments/statistics.py` 신설 — 부트스트랩 CI + 가설검정 (시드 20260901).
  - 강건 정량: 연속 변곡점 CI [2.99,3.65], INT2 PCAG 안정.
  - 정직한 한계: 근접-기준 PCAG·이산 기울기 유의성은 취약.
- `experiments/model_form.py` 신설 — 대안 함수형(Weibull/Exp/Tanh/Hill × LinLog/Power/Logistic/LinPow)에 대한 b\*.
  - b\* ∈ [4.19, 4.27] (표준 포화형 4종), 가속 손실 구조 부재 시 변곡점 도메인 밖.
- `experiments/analytical_proof.py` — `condition_3_2_3_1_consistency` 추가 (명제 3.6 수치 검증).
- `experiments/verify_numbers.py` — 60 → 78개 항목으로 확장.
- `experiments/make_figures.py` — Fig 18(부트스트랩 추론), Fig 19(모델-형 강건성) 추가 (17→19종).
- `experiments/dry_run.py` — statistics.py·model_form.py·analytical_proof.py·verify_numbers.py 포함
  전체 파이프라인 검증으로 확장 + 신규 산출물(fig18/19) 검증 항목 추가.

**검증**
- `verify_numbers.py` : 78 PASS / 0 FAIL.
- `dry_run.py` : 전 파이프라인(분석→증명→민감도→부트스트랩→모델-형→Jevons→게이트→산출물) OK.

## v3 — 2026-09-01 (main, pre-Colab 원래 연구 방향)

**문서**
- `docs/main_paper.md` v3:
  - 정의(Definition 3.1~3.3)·정리(Theorem 3.4, 5.1)·계(Corollary 3.5, 5.2) 번호 부여.
  - 기호/표기(notation) 표 추가 (3.3.0).
  - 관련 연구 확장 + 본문 인용 [1]~[14] (2장).
  - 위협 요소 및 유효성 한계(Threats to Validity) 절 추가 (4.5).
  - 재현성·데이터/코드 가용성 진술 추가.
  - 통계 부록 B 추가 (주 분석, 다중 모델, Monte Carlo, θ-스윕, Jevons 표).
  - 참고문헌 14종으로 확장·번호 매김.
- `docs/references.bib` 신설 — 참고문헌 BibTeX.
- `docs/measurement_protocol.md` 신설 — GPU 실측 표준 프로토콜.
- `docs/reproducibility.md` 신설 — 환경·명령열·골든 해시·검증 매트릭스.
- `logs/change_log.md` 신설 — 본 문서.

**코드**
- `experiments/verify_numbers.py` 신설 — 논문 수치 ↔ 산출물 무결성 게이트 (60개 항목).
- `experiments/generate_results.py`, `experiments/multimodel_data.py` — 생성일을 고정 상수
  (ANCHOR_GEN_DATE)로 변경 → 재현 시 결정적(byte-identical) 출력.
- `requirements.txt` 신설 — 검증된 버전 고정.

**검증**
- 전체 파이프라인 재실행 확인 (`dry_run.py` OK).
- `verify_numbers.py` : 60 PASS / 0 FAIL.

## v2 — 2026-08-28 (main 이전 이력)

- `docs/main_paper.md` v2: 다중 모델 일반화, 해석적 유도(3.3.4), Monte Carlo 강건성(4.4),
  Jevons 폐형(5.1), 그림 색인 부록 A 추가.
- `experiments/` : `multimodel_data.py`, `sensitivity.py`, `analytical_proof.py`, `make_figures.py`(그림 17종).
- `docs/proof_3_1_derivation.md` 신설.

## v1 — 2026-08-27

- `docs/main_paper.md` v1: PCAG 정의(운영 재정의), Power Wall 판정, Jevons 시뮬레이션, 그림 fig1~3.
- `experiments/` : `schema.py`, `telemetry.py`, `eval_harness.py`, `benchmark_driver.py`,
  `generate_results.py`, `analysis.py`, `jevons_model.py`, `dry_run.py`.
- `docs/figures/` fig1~3, `logs/troubleshooting_archive.md`(ISSUE 7건).

---
*생성 도구: PCAG Research Pipeline. v3 작성일 2026-09-01.*
