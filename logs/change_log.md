# 변경 로그 (Change Log)

본 저장소의 코드/데이터/문서 변경을 커밋 단위로 기록한다. 각 항목은
[커밋 해시 | 날짜 | 변경 | 이유/근거] 를 따른다. 설계 결정의 이유는
`docs/design_decisions.md`, 기술 장애는 `logs/troubleshooting_archive.md` 참고.

> 해시는 로컬 main 브랜치 기준(간략 7자리). 원격 push 전 시점의 순서 기록.

---

## 2026-08-31 — Phase 7~9: Colab 실측 인프라 + 방법론 재설계

### `e6e8dec` — feat(experiments): INT3/INT2 packed 저비트 양자화 엔진 추가
- **변경**: `experiments/lowbit.py` 신규 — per-channel RTN 대칭 양자화를 uint8
  비트패킹(INT2=0.25B, INT3=0.375B/값)으로 저장. 양자화 시 원본 fp16 즉시 해제.
- **근거**: bitsandbytes 는 INT3/INT2 미지원(ISSUE-METHOD-01). T4 16GB 에서 FP16 전체
  로딩 없이 저비트 GPU 상주 필요.

### `5914fd6` — feat(experiments): benchmark_driver Colab T4 최적화
- **변경**: `benchmark_driver.py` 재작성 — `--models` 분할 실행, 정밀도별 즉시 CSV
  upsert + 체크포인트 JSON + `--resume`, `--drive-dir` Drive 미러링, OOM 방지
  (`PYTORCH_CUDA_ALLOC_CONF`, 배치 chunking, peak VRAM 리셋).
- **근거**: 무료 Colab 세션 타임아웃(ISSUE-ENV-04). 기준 CSV(`results_raw.csv`) 자동
  갱신 옵션 `--update-main`.

### `4d07b85` — feat(experiments): telemetry torch.cuda 전력/에너지 추정 폴백
- **변경**: `telemetry.py` 에 `PowerMeter.estimate()` + TDP 상수표(T4=70W 등) 추가.
- **근거**: 가상화 GPU 전력 계측 불가(ISSUE-ENV-05, DD-08).

### `312c0d7` — feat(experiments): make_figures 실측(Measured-GPU) 감지 시 출처 문구 갱신
- **변경**: `make_figures.py` `refresh_footnote()` — 실측 CSV 존재 시 그림 하단을
  `Source=Measured-GPU, Google Colab T4` 로 자동 전환.

### `d8fd076` — docs: 무료 Colab T4 실측 실행 가이드 추가
- **변경**: `docs/colab_execution_guide.md` 신규 + README 링크.

### `5b168ef` — fix(experiments): 학술 무결성 보완 (단일 측정 호출·에너지 정규화·no_eval 빈 값)
- **변경**: 전력 계측과 성능 측정을 단일 `_bench()` 호출로 통합(ISSUE-CODE-09).
  `--energy_norm`(J/1,000 tokens) 추가. `--no_eval` 시 Accuracy 빈 값(ISSUE-CODE-07).
  Notes 에 INT3/INT2 RTN·TDP 추정·평가 방법 명시.

### `591a144` — fix(analysis): 정확도/에너지 누락 행 자동 제외
- **변경**: analysis/sensitivity/make_figures 3개 로더가 float() 파싱 실패 행을
  스킵 + 개수 경고. (ISSUE-CODE-07 의 소비처 대응)

### `c7c891d` — feat(experiments): lowbit 4/5/6/8-bit 확장
- **변경**: 비트패킹을 2/3/4/5/6/8 지원(INT8=1B, INT6=0.75B, INT5=0.625B, INT4=0.5B/값).
- **근거**: DD-06 — 미세 비트 곡선 {16,8,6,5,4,3,2} 달성. pure-Python 라운드트립 검증.

### `922be94` — feat(experiments): ARC-Easy 표준 0-shot 객관식 평가
- **변경**: `eval_harness.py` 에 `run_arc_easy()` — ai2_arc/ARC-Easy test, 고정 seed,
  A–E 글자 정확도. datasets 미설치 시 synthetic 폴백.
- **근거**: DD-03 — 문헌 비교 가능한 표준 벤치마크.

### `e9b061d` — feat(experiments): 방법 분리(rtn/bnb) + 반복 측정 + 7비트폭
- **변경**: `--quant-method rtn/bnb`, bnb 는 `results_multimodel_bnb_raw.csv` 분리.
  `--repeats`(mean±std), `--update-main` rtn 전용(기본 off).
- **근거**: DD-02, DD-04, ISSUE-DATA-02.

### `abac342` — feat(analysis): INT6/INT5 비트폭 지원 + make_figures 데이터 구동화
- **변경**: analysis/sensitivity/analytical_proof PREC_BITS 확장. make_figures 가
  analysis_summary.json 을 읽어 벽 구간·기울기·변곡점·축 범위 계산(하드코딩 제거).
- **근거**: DD-05, ISSUE-CODE-10.

### `fa6668d` — docs: Colab 가이드를 '균일 RTN PCAG 곡선' 학술 프로토콜로 갱신
- **변경**: 가이드를 새 방법론(7비트폭 rtn + ARC-Easy + repeats + energy_norm +
  bnb 별도 세트)으로 재작성.

### `a1d6f1f` — feat(colab): 원클릭 실측 노트북 추가
- **변경**: `notebooks/PCAG_Measured_GPU.ipynb` (24셀) — Run all 로 실측+파이프라인 완료.
- **근거**: 수동 절차를 1회 실행으로 축약.

### `(본 문서) ` — docs: 연구 기록 문서 일괄 보강
- **변경**: `research_journal.md` Phase 7~9, `logs/troubleshooting_archive.md`
  ISSUE-ENV-04/05, METHOD-01, CODE-07~11, DATA-02, `docs/design_decisions.md` (DD-01~08)
  신규, `docs/measurement_protocol.md` 신규, 본 change_log 신규.
- **근거**: 사용자 요청 — 실측 인프라 작업을 연구자 표준 기록 형태로 문서화.

---

## 2026-08-28 — Phase 6: GPU 무관 확장 (기존 기록)

- `ebe5db7` assets(figures): 논문용 그림 17종 (PNG 200dpi + PDF)
- `628d12e` docs(paper): 논문 원고 v2 + 해석적 유도 부록
- `4f902ff` data(experiments): 참고 데이터·분석 산출물
- `a2e7ec0` feat(experiments): 측정·분석·증명·시각화 전체 파이프라인
- `29c43b9` docs: 연구 프로토콜·저널·트러블슈팅 아카이브

> 상세 내용은 `research_journal.md` 2026-08-27/28 항목 참고.