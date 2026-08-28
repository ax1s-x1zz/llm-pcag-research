# experiments/dry_run.py
# Phase 1 검증: 파이프라인(데이터 생성 -> PCAG 분석 -> Jevons) 통합 dry-run.
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

    # 2) PCAG 분석
    step("PCAG 분석 + Fig1/Fig2", "analysis.py")

    # 3) Jevons 시뮬레이션 + Fig3
    step("Jevons 모델 + Fig3", "jevons_model.py")

    # 4) 산출물 검증
    figures = os.path.normpath(os.path.join(ROOT, "..", "docs", "figures"))
    checks = [
        ("results_raw.csv", os.path.exists(csv_path)),
        ("analysis_summary.json", os.path.exists(os.path.join(ROOT, "analysis_summary.json"))),
        ("jevons_summary.json", os.path.exists(os.path.join(ROOT, "jevons_summary.json"))),
        ("fig1", os.path.exists(os.path.join(figures, "fig1_accuracy_vs_energy.png"))),
        ("fig2", os.path.exists(os.path.join(figures, "fig2_pcag_power_wall.png"))),
        ("fig3", os.path.exists(os.path.join(figures, "fig3_jevons_grid_load.png"))),
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