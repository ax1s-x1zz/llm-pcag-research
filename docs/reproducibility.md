# 재현성 보고서 (Reproducibility Report)

> 본 문서는 `docs/main_paper.md` 의 모든 수치·그림·증명이 **단일 명령열로 결정적(byte-identical) 재현**됨을
> 보장하는 재현 절차와 환경을 명세한다. GPU 실측 단계 전환 시에는 `docs/measurement_protocol.md` 를 따른다.

---

## 1. 검증 시점의 환경 (2026-09-01)

| 항목 | 값 |
|---|---|
| OS | Linux |
| CPU | Intel Core Ultra 7 155H (22 threads) |
| RAM | 7.6 GB (Swap 2 GB) — 참고 데이터 파이프라인 충분 |
| GPU | 없음 (`nvidia-smi` 미존재) — 본 문서는 참고 데이터 단계 기준 |
| Python | 3.14.4 (venv 내 pip 부트스트랩: `get-pip.py --break-system-packages`) |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| scipy | 1.18.1 |
| sympy | 1.14.0 |

정확한 버전은 `requirements.txt` 에 고정되어 있다.

## 2. 재현성 원칙

1. **결정적 데이터 생성**: 참고 데이터 스크립트(`generate_results.py`, `multimodel_data.py`)는
   `datetime.date.today()` 대신 **고정 생성일 상수(ANCHOR_GEN_DATE)** 를 사용한다.
   → 재실행 시 출력이 **바이트 단위로 동일**하다.
2. **고정 시드**: Monte Carlo는 `np.random.default_rng(42)` (sensitivity.py) 로 시드 고정.
3. **수치 전사 금지**: 논문의 모든 수치는 커밋된 JSON/CSV에서 직접 전사되며,
   `experiments/verify_numbers.py` 가 60개 항목을 자동 검증한다.
4. **산출물 커밋**: 원시 데이터·분석 결과가 저장소에 커밋되어 있어 네트워크 없이도 검증 가능.

## 3. 재현 명령열 (참고 데이터 단계)

```bash
python -m venv .venv                      # 또는 get-pip.py 부트스트랩 후 venv 재구축
.venv/bin/pip install -r requirements.txt

cd experiments
python generate_results.py                # (1) 참고 데이터 — 결정적
python multimodel_data.py                 # (2) 다중 모델 앵커 — 결정적
python analysis.py                        # (3) PCAG 분석 + Power Wall 판정
python analytical_proof.py                # (4) 폐형 유도 + Jevons 기호 증명 (+ proof_3_1_derivation.md)
python sensitivity.py                     # (5) θ 스윕 + Jevons 그리드 + Monte Carlo (시드 42)
python jevons_model.py                    # (6) Jevons 시나리오
python make_figures.py                    # (7) 그림 17종 (PNG 200dpi + PDF)
python verify_numbers.py                  # (8) 무결성 게이트 — 모든 수치 PASS 기대, exit 0
python dry_run.py                         # (9) 통합 파이프라인 검증
```

> 참고: `make_figures.py` 재실행 시 matplotlib 버전에 따라 PNG/PDF 바이트가 달라질 수 있다
> (렌더링 엔진 버전 의존). **수치 내용은 변하지 않는다.** 데이터/분석 JSON은 결정적이므로
> 바이트 동일하다.

## 4. 골든 해시 (Golden Checksums, 2026-09-01 기준)

재현 후 아래 파일의 SHA-256 이 동일하면 데이터·분석 단계가 정확히 재현된 것이다.

```
75782aa0111c2415975ea132592b7b9dec1102fcf6d16b4c7b1159687849dd5e  experiments/results_raw.csv
410bea16cb8778e5db7f7efe0f22e5deed0393a0cd8e444159486fbff69a9315  experiments/results_multimodel_raw.csv
8162db85f87500fe353b8af439e10cb4843b8c71c1b03d2264f7f15d06f7a448  experiments/analysis_summary.json
5a38e92ee1b96c7c94bb5247ee3357f1c3d9a271f53a401f939e24dc599ee125  experiments/analysis_proof.json
ee005121b6cef35b6ab566876a9c7059eeb9aaf496a7b402496057a947a3f507  experiments/sensitivity_summary.json
7d9ff44dd1182dc572a507e52087cd1598c26f63f0e325c694d103ca6227b61f  experiments/jevons_summary.json
```

```bash
cd <repo-root>
sha256sum experiments/results_raw.csv experiments/results_multimodel_raw.csv \
          experiments/analysis_summary.json experiments/analysis_proof.json \
          experiments/sensitivity_summary.json experiments/jevons_summary.json
```

## 5. 논문 수치 ↔ 산출물 검증 매트릭스

`experiments/verify_numbers.py` 가 다음 항목을 자동 검증한다(총 60개).

| 논문 항목 | 값 | 출처 |
|---|---|---|
| PCAG INT8/INT4/INT3/INT2 | 20.76 / 10.44 / 4.76 / 2.20 | `analysis_summary.json` |
| Power Wall 전이 | INT4→INT3 | `analysis_summary.json` |
| INT4→INT3 기울기 | 5.68 | `analysis_summary.json` |
| 연속형 변곡점 PCHIP | 3.51 | `analysis_summary.json` |
| 해석적 b\* | 4.19 (x\*=11.81, d³≠0) | `analysis_proof.json` |
| MC 평균·SD·90% CI | 3.40±0.25, [2.99, 3.66] | `sensitivity_summary.json` |
| MC 벽 전이 확률 (INT4→3) | 66.8% | `sensitivity_summary.json` |
| 다중 모델 기울기 | Llama 5.60 / Qwen 9.78 / Gemma 6.49 / Mistral 6.96 | `sensitivity_summary.json` |
| θ<5.68 불변성 | True | `sensitivity_summary.json` |
| Jevons 부하 증가 영역 | 65.6% | `sensitivity_summary.json` |
| INT4 수요 증가 / 총부하 | +231% / +49% (폐형 49.07%) | `jevons_summary.json` |
| 해석 모델 파라미터 | S_max 0.796, λ 0.0932, β 1.957, c 0.00205, Lr_max 0.558, k 1.396, x_c 14.07 | `analysis_proof.json` |
| 원시 앵커 (acc/energy) | FP16 66.6/140 → INT2 47.0/49.5 | `results_raw.csv` |

## 6. GPU 실측 단계 전환 시 재현 갱신

- `docs/measurement_protocol.md` 의 Swap Procedure(§8)를 따른다.
- 실측치(`Source=Measured-GPU`)로 교체하면 **골든 해시와 verify_numbers.py 가 실패**하는 항목이 곧
  논문에서 갱신할 수치이다. 갱신 후 이 보고서의 해시·매트릭스도 재기록한다.
- Power Wall 정밀화를 위해 INT6/INT5 세밀 비트 샘플링을 추가한다.

---
*생성 도구: PCAG Research Pipeline. 작성일 2026-09-01.*
