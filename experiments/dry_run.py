# experiments/dry_run.py
# Phase 1 검증: 전체 파이프라인(데이터 생성 → PCAG 분석 → 해석 증명 → 민감도 →
#               통계 추론 → 모델-형 강건성 → Jevons → 시각화 → 무결성 게이트) 통합 dry-run.
# - GPU가 없어도 실행 가능 (참고 데이터 기반).
# - GPU 실측 시: benchmark_driver.py 가 results_raw.csv 를 덮어쓴 뒤 이 스크립트로 전 과정 재검증.
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def step(name, script):
    print(f"\n{'='*60}\n[{name}] python {script}\n{'='*60}")
    r = subprocess.run([PY, os.path.join(ROOT, script)], cwd=ROOT,
                       capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        raise SystemExit(f"[{name}] 실패 (exit={r.returncode})")
    print(f"[{name}] OK")


def main():
    # 1) 참고 데이터 (또는 GPU 실측 결과) CSV 준비 확인
    csv_path = os.path.join(ROOT, "results_raw.csv")
    if not os.path.exists(csv_path):
        step("데이터 생성", "generate_results.py")
    else:
        print(f"CSV 존재: {csv_path} (재사용)")

    # 2) 다중 모델 앵커
    if not os.path.exists(os.path.join(ROOT, "results_multimodel_raw.csv")):
        step("다중 모델 앵커", "multimodel_data.py")
    else:
        print("다중 모델 CSV 존재 (재사용)")

    # 3) PCAG 분석 + Fig1/Fig2
    step("PCAG 분석 + Fig1/Fig2", "analysis.py")

    # 4) 해석적 증명 (조건식 3.1 + Jevons 폐형 + 조건식 3.2↔3.1 일관성) + 부록 문서
    step("해석적 증명", "analytical_proof.py")

    # 5) 민감도 (θ 스윕 + Jevons 그리드 + Monte Carlo)
    step("민감도 분석", "sensitivity.py")

    # 6) 부트스트랩 통계 추론
    step("부트스트랩 통계 추론", "statistics.py")

    # 7) 모델-형 강건성
    step("모델-형 강건성", "model_form.py")

    # 8) Jevons 시뮬레이션 + Fig3
    step("Jevons 모델 + Fig3", "jevons_model.py")

    # 9) 무결성 게이트 (논문 수치 ↔ 산출물)
    step("무결성 게이트", "verify_numbers.py")

    # 10) 산출물 검증
    figures = os.path.normpath(os.path.join(ROOT, "..", "docs", "figures"))
    checks = [
        ("results_raw.csv", os.path.exists(csv_path)),
        ("results_multimodel_raw.csv", os.path.exists(os.path.join(ROOT, "results_multimodel_raw.csv"))),
        ("analysis_summary.json", os.path.exists(os.path.join(ROOT, "analysis_summary.json"))),
        ("analysis_proof.json", os.path.exists(os.path.join(ROOT, "analysis_proof.json"))),
        ("sensitivity_summary.json", os.path.exists(os.path.join(ROOT, "sensitivity_summary.json"))),
        ("statistics_summary.json", os.path.exists(os.path.join(ROOT, "statistics_summary.json"))),
        ("model_form_summary.json", os.path.exists(os.path.join(ROOT, "model_form_summary.json"))),
        ("jevons_summary.json", os.path.exists(os.path.join(ROOT, "jevons_summary.json"))),
        ("fig1", os.path.exists(os.path.join(figures, "fig1_accuracy_vs_energy.png"))),
        ("fig2", os.path.exists(os.path.join(figures, "fig2_pcag_power_wall.png"))),
        ("fig3", os.path.exists(os.path.join(figures, "fig3_jevons_grid_load.png"))),
        ("fig18", os.path.exists(os.path.join(figures, "fig18_bootstrap_inference.png"))),
        ("fig19", os.path.exists(os.path.join(figures, "fig19_model_form_robustness.png"))),
    ]
    print("\n=== 산출물 검증 ===")
    all_ok = True
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
        all_ok &= ok
    if not all_ok:
        raise SystemExit("일부 산출물 누락")
    print("\nDry-run 통합 검증 성공.")


if __name__ == "__main__":
    main()
