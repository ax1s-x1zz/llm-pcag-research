# 트러블슈팅 아카이브 (Troubleshooting Archive)

본 문서는 연구/개발 과정에서 발생한 실질적 기술 장애를 기록한 엔지니어링 회고록이다.
각 항목은 [이슈 ID, 시각, 증상, 근본 원인 가설, 시도한 해결책, 최종 수정, 교훈] 형식을 따른다.

---

## ISSUE-ENV-01: GPU 부재 — 실측 불가 환경
- **시각**: 2026-08-27 00:50 (Phase 0)
- **증상**: `nvidia-smi` 명령이 없음. `torch.cuda.is_available()==False` 예상. RAM 7.6GB 제한.
- **근본 원인 가설**: 렌탈 GPU 미확보 환경. 본 프로젝트의 핵심 실험(FP16/INT8/INT4 다중 정밀도 추론)은 GPU가 필수.
- **시도한 해결책**: GPU를 직접 빌릴 수 없으므로 실측 대신 **문헌 앵커 기반 참고 데이터**로 전체 파이프라인을 구동. 절대 실측으로 위장하지 않음.
- **최종 수정**: `generate_results.py` 가 `Source=Reference-Literature` 행을 생성. GPU 확보 시 `benchmark_driver.py` 가 `Source=Measured-GPU` 로 같은 CSV 스키마를 덮어쓸 수 있게 구조화. `dry_run.py` 로 통합 검증.
- **교훈**: 실험 환경 부재 시 "도구(파이프라인)를 완성 + 대체 데이터로 검증 가능하게" 만드는 것이 무결성을 유지하는 최선책. 원고와 데이터에 출처 라벨을 남겨 신뢰성 유지.

---

## ISSUE-ENV-02: 시스템 pip/venv 부재
- **시각**: 2026-08-27 00:53 (Phase 0)
- **증상**: `python3 -m pip` → "No module named pip". `ensurepip` 부재. PEP 668 (externally-managed) 제약.
- **근본 원인 가설**: 배포판 Python이 pip 미포함, 시스템 레벨 패키지 격리 정책.
- **시도한 해결책**: `get-pip.py` 부트스트랩 시도 → `--break-system-packages` 없이 실패. `python3 -m venv` 생성 → venv 내 pip도 없음.
- **최종 수정**: `python3 get-pip.py --break-system-packages` 를 `/tmp/opencode/venv` 에 적용해 pip 26.2.1 확보. 이후 `numpy/pandas/matplotlib/scipy/sympy` 설치 성공 (Python 3.14 휠 지원 확인).
- **교훈**: 시스템 Python은 망가뜨리지 말고 격리된 venv에서 부트스트랩하는 것이 안전. 의존성은 venv에만 설치.

---

## ISSUE-CODE-01: torch import로 인한 스키마 의존성 충돌
- **시각**: 2026-08-27 01:00 (Phase 1)
- **증상**: `python generate_results.py` 실행 시 `ModuleNotFoundError: No module named 'torch'`.
- **근본 원인 가설**: `generate_results.py` 가 `benchmark_driver.py`에서 `RESULTS_COLUMNS`를 import하는데, benchmark_driver 모듈 최상단에서 `import torch` (미설치) 때문에 실패.
- **시도한 해결책**: torch를 설치하려 했으나 Python 3.14 GPU 제약으로 부적절. 스키마를 별도 격리 모듈로 분리하는 것이 근본 해결.
- **최종 수정**: 공용 스키마 `schema.py` (RESULTS_COLUMNS, CSV_PATH) 신설. `benchmark_driver.py` 는 torch를 `_lazy_imports()`로 지연 import. `generate_results.py` 는 `schema`만 import.
- **교훈**: 무거운 ML 의존성(torch/transformers)은 "실측 실행 시점에만" 로드되도록 게으른 import 패턴을 써야, 분석/데이터 생성 파이프라인과 결합도를 낮출 수 있음.

---

## ISSUE-PCAG-01: PCAG 원문 수식 부호 모순 (수학적 무결성)
- **시각**: 2026-08-27 01:10 (Phase 3)
- **증상**: 원문 [공식 3.1] `PCAGk = (Ak-A0)/(P0-Pk)`. 여기서 `Ak≤A0` → 분자 `≤0`, `Pk<P0` → 분모 `>0`. 즉 **항상 PCAG ≤ 0**. 그런데 원문 3.3.2는 "상승/최적 구간 (PCAGk > 0)"이라고 정의 → **수식과 해석의 부호 모순** 발견.
- **근본 원인 가설**: 원문 작성자가 "정확도 하락(음) 대비 전력 절감(양)"의 효율성을 양수로 표현하고 싶었으나, ΔA를 그대로 분자에 넣어 부호가 뒤집힘. (즉 의도는 `|ΔA|` 또는 상대 비율이었을 것)
- **시도한 해결책**: ① 부호를 무시하고 원문 그대로 쓸지, ② 효율성(interpretable) 정의로 바꿀지 검토.
- **최종 수정**: **운영(interpretable) PCAG** 를 다음과 같이 정의해 사용:
  - `PCAG_k = ( (P0-Pk)/P0 ) / ( (A0-Ak)/A0 )` — "단위 상대 정확도 손실당 상대 전력 절감" 효율 지표(항상 양수).
  - 원문의 `ΔA/ΔP` 값(`pcag_raw`)도 함께 보고해 투명성 유지.
  - 논문 3.3절에서 이 재정의 근거를 명시적으로 서술.
- **교훈**: 기존 정의 문서의 수식이 해석 구간 정의와 모순되면, 조용히 "고치지 말고" 근거와 함께 재정의해야 학술적 무결성이 지켜짐. 이는 실제 리뷰에서 지적받을 소지가 높은 지점이므로 원고에 명시.

---

## ISSUE-CODE-02: None값 포맷 오류
- **시각**: 2026-08-27 01:12 (Phase 3)
- **증상**: `analysis.py` 실행 시 `TypeError: unsupported format string passed to NoneType.__format__`. 기준 모델(FP16)은 `dA=0`이라 `pcag_eff=None`.
- **근본 원인 가설**: 기준점은 분모/분자가 0이 되어 PCAG가 정의되지 않음(0/0). None을 그대로 f-string 포맷에 사용.
- **시도한 해결책**: 기준점을 제외하거나 None 처리.
- **최종 수정**: `_fmt/_fmt2` 헬퍼로 None을 `"n/a"` 표시. `discrete_slopes`에서도 `pcag_eff`가 None인 점은 제외.
- **교훈**: 기준(reference) 데이터 포인트는 지표 계산에서 특수 처리(제외/미정의)가 필요. edge case를 놓치지 않도록 None 안전 코딩.

---

## ISSUE-CODE-03: PCHIP x 감소순 오류로 연속형 변곡점 미검출
- **시각**: 2026-08-27 01:15 (Phase 3)
- **증상**: `analysis_summary.json` 의 `continuous_inflection=None`. 디버깅 결과 `ValueError: x must be strictly increasing sequence`.
- **근본 원인 가설**: `PchipInterpolator`는 x를 오름차순으로 요구하는데, 데이터가 bits 8→4→3→2 (감소순)으로 전달됨. 내부 `argsort` 없이 전달한 버그.
- **시도한 해결책**: 초기 검증 코드에서 원인 직접 확인. `argsort`로 정렬.
- **최종 수정**: `fit_continuous_pcag`에서 `np.argsort(b)` 로 정렬 후 보간. 연속형 변곡점 b≈3.51 검출 성공 (이산 Power Wall INT4~3 구간과 일치).
- **교훈**: 보간/회귀 라이브러리의 입력 순서 가정을 항상 확인. "안 되면 조용히 넘어가고" not, 디버깅 콘솔에서 단위 검증.

---

## ISSUE-DATA-01: 참고 데이터의 과학적 앵커링 한계
- **시각**: 2026-08-27 01:05 (Phase 2)
- **증상**: (의도된 한계) 참고 데이터는 문헌 기반으로 합성되어 실측 분산/하드웨어 특성 미반영.
- **근본 원인 가설**: GPU 미실측이라는 환경 제약.
- **시도한 해결책**: (없음 — 원천적 제약)
- **최종 수정**: `results_raw.csv` 에 `Source=Reference-Literature` 명시, 논문에서 "참고 데이터 시나리오"로 정직하게 한정. GPU 실측 시 교체 경로 명확화.
- **교훈**: 데이터의 한계를 숨기지 않고 문서화하는 것이 리뷰 대응에 중요. 제출 전 반드시 GPU 실측 교체 필요 명시.

---

## ISSUE-ENV-03: /tmp venv 소멸 — 세션 재시작 후 의존성 유실
- **시각**: 2026-08-28 (Phase 6, GPU 무관 확장 작업)
- **증상**: `/tmp/opencode/venv/bin/python` 부재. 기존에 구축한 venv와 설치 패키지(numpy/pandas/matplotlib/scipy/sympy)가 모두 삭제됨.
- **근본 원인 가설**: `/tmp` 휘발성 파일시스템 — 세션/재부팅 간 내용이 보존되지 않음.
- **시도한 해결책**: `python3 -m venv` 재시도 → ensurepip 부재로 재실패 (ISSUE-ENV-02 재발).
- **최종 수정**: 동일 부트스트랩 경로 재적용: `get-pip.py --break-system-packages` → pip 26.2.1 → `virtualenv` 설치 → `/tmp/opencode/venv` 재구축 → 5개 패키지 재설치.
- **교훈**: /tmp 기반 환경은 매 세션마다 재구축 각오 필요. 재현 절차를 저널에 문서화해 두어 복구 시간 최소화. 장기적으로는 워크스페이스 내 persistent 경로 사용 검토.

---

## ISSUE-CODE-04: 복소수 스텝 미분에서 np.expm1·np.real 오용
- **시각**: 2026-08-28 (analytical_proof.py)
- **증상**: 변곡점 방정식 h(x)의 근 탐색 결과가 비물리적(h' ≈ −6.3e18). 첫 구현에서는 h'에 허수·실수 혼용 의심.
- **근본 원인 가설**: 두 중첩 버그. (1) `np.expm1`은 복소수 인자를 지원하지 않아 내부에서 허수부가 버려짐(ComplexWarning). (2) 복소수 스텝 미분은 `g'(x) = Im(g(x+iε))/ε` 인데 `np.real`로 실수부를 나눠버림 → g/ε ≈ 1e18 발산.
- **시도한 해결책**: sympy nsolve 기반 폐형 전수 해석 시도 → 비-정수 지수(Weibull β)로 simplify가 수 분간 미종료(별도 이슈 ISSUE-CODE-05).
- **최종 수정**: `exp(u)−1`로 교체(np.exp는 복소수 지원) + `np.imag` 사용. 복소수 스텝 ε=1e-20로 기계정밀도 미분 확보. 중앙차분 대조 검증으로 폐형 g의 상대오차 <6e-8 확인.
- **교훈**: 복소수 스텝 미분은 함수 전체가 복소수 해석적이어야 한다 — numpy 함수별 복소수 지원 여부를 확인할 것. 수치 기교 적용 시 반드시 독립적 대조 검증(중앙차분)을 병행.

---

## ISSUE-CODE-05: sympy simplify 폐형 유도 시 타임아웃 (비-정수 지수)
- **시각**: 2026-08-28 (analytical_proof.py)
- **증상**: `sp.simplify`가 Weibull 항 (λx)^β (β: 기호, 비-정수)을 포함한 PCAG/g/h 표현식에서 5분 이상 미종료 → 쉘 타임아웃.
- **근본 원인 가설**: 기호 지수를 포함한 exp-다항 혼합식의 simplify는 조합폭발. 논문 제출용 "예쁜 폐형"을 억지로 얻으려는 시도가 본말전도.
- **시도한 해결책**: ① simplify 단순화 옵션 조정 — 여전히 느림. ② 폐형식을 손으로 유도하고 코드는 그 폐형을 직접 구현 + 수치 대조 검증으로 방향 전환.
- **최종 수정**: g(x)의 각 항(S'/S, Lr'/Lr)은 폐형을 직접 기술(논문/부록에 수식 명시), 심볼릭 검증이 필요한 Jevons 다항-지수 항등식만 sympy 사용(즉시 완료). h의 근은 복소수 스텝 미분 + 전수 스캔 + brentq로 고신뢰 계산.
- **교훈**: sympy는 "증명이 필요한 항등식"에 국한하고, 미분방정식 해석은 폐형 직접 유도 + 고정밀 수치 검증 조합이 실용적. 기호 지수가 들어간 식의 simplify는 비용을 먼저 측정.

---

## ISSUE-CODE-06: Monte Carlo 변곡점 검출 무응답 (예외 무시 버그)
- **시각**: 2026-08-28 (sensitivity.py)
- **증상**: MC 3000회 섭동에서 변곡점 샘플이 0개(mean=None). 단일 실행(analysis.py)에서는 b≈3.51이 잘 검출되던 로직.
- **근본 원인 가설**: `f = lambda x: float(interp(x))` 에 400점 배열을 통째로 전달 → `float()` TypeError → 바깥 `except: pass`가 조용히 삼켜서 모든 반복이 스킵됨.
- **시도한 해결책**: 샘플 1개에 대해 디버그 실행으로 예외 확인.
- **최종 수정**: 스칼라 float() 래퍼 제거, PCHIP을 배열에 직접 벡터 호출. 근 위치는 영교차 선형 보간으로 정제. 재실행 결과 b*=3.398±0.252 (90% CI [2.99,3.66]) 정상 검출.
- **교훈**: `except: pass`는 수치 파이프라인에서 치명적 — 최소한 로깅하거나 카운터로 실패율을 보고할 것. "0개 결과"는 곧 전수 실패 신호로 해석하고 원인을 추적.