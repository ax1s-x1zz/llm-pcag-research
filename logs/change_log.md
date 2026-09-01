# 변경 로그 (Change Log)

연구 원고·파이프라인의 버전 이력. 각 항목은 [날짜, 문서/코드, 변경 내용]을 기록한다.
(연구 과정의 의사결정·장애는 `research_journal.md`, 기술 회고는 `logs/troubleshooting_archive.md` 참조.)

---

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
