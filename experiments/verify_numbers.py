# experiments/verify_numbers.py
# 학술 무결성 자동 검증 게이트 (Academic Integrity Gate)
#
# docs/main_paper.md (및 README/README_kr)에 인용된 모든 대표 수치가
# 커밋된 산출물(CSV/JSON)과 일치하는지 검증한다.
#
# 검증 대상:
#   - 원시 참고 데이터 (results_raw.csv / results_multimodel_raw.csv)
#   - PCAG 분석 (analysis_summary.json)
#   - 해석적 유도 (analysis_proof.json)
#   - 민감도·Monte Carlo·θ 스윕·Jevons 그리드 (sensitivity_summary.json)
#   - Jevons 시나리오 (jevons_summary.json)
#
# 사용법:
#   python verify_numbers.py            # 전체 검증, 실패 시 exit 1
#   python verify_numbers.py --brief     # 통과 요약만 출력
#
# 중요: GPU 실측(benchmark_driver.py)으로 앵커가 교체되면,
# 이 게이트를 실행해 "어느 논문 수치가 이제 낡았는지" 즉시 파악한다.
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-2          # 상대 허용 오차 (그림/표에서 반올림된 인용치 대비)
ABS_TOL = 1e-2      # 절대 허용 오차 (소수 2자리 인용치 대비)


def load_json(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def load_csv(name):
    with open(os.path.join(HERE, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- 검증 항목 레지스트리 --------------------------------------------------
# 각 항목: (설명, actual, expected, [abs_tol])
CHECKS = []


def check(label, actual, expected, abs_tol=ABS_TOL, rel_tol=TOL):
    CHECKS.append((label, actual, expected, abs_tol, rel_tol))


# =====================================================================
# 1. 원시 참고 데이터 (results_raw.csv — Llama-3-8B 주 분석)
# =====================================================================
main_rows = {r["Precision"]: r for r in load_csv("results_raw.csv")}

EXP_MAIN_ACC = {"FP16": 66.6, "INT8": 65.5, "INT4": 63.0, "INT3": 58.0, "INT2": 47.0}
EXP_MAIN_ENG = {"FP16": 140.0, "INT8": 92.0, "INT4": 61.0, "INT3": 54.0, "INT2": 49.5}
for prec, exp in EXP_MAIN_ACC.items():
    check(f"results_raw[{prec}].accuracy", float(main_rows[prec]["Accuracy_Score"]), exp)
for prec, exp in EXP_MAIN_ENG.items():
    check(f"results_raw[{prec}].energy", float(main_rows[prec]["Total_Energy_J"]), exp)
for prec in main_rows:
    check(f"results_raw[{prec}].source",
          main_rows[prec]["Source"], "Reference-Literature")
check("results_raw[INT4].energy_savings",
      (140.0 - float(main_rows["INT4"]["Total_Energy_J"])) / 140.0, 0.5643)

# =====================================================================
# 2. PCAG 분석 (analysis_summary.json)
# =====================================================================
an = load_json("analysis_summary.json")
pcag = {p["precision"]: p for p in an["points"]}
EXP_PCAG = {"INT8": 20.76, "INT4": 10.44, "INT3": 4.76, "INT2": 2.20}
for prec, exp in EXP_PCAG.items():
    check(f"analysis.pcag_eff[{prec}]", pcag[prec]["pcag_eff"], exp, abs_tol=0.02)
check("analysis.power_wall.from", an["power_wall"]["from"], "INT4")
check("analysis.power_wall.to", an["power_wall"]["to"], "INT3")
check("analysis.power_wall.slope_per_bit", an["power_wall"]["slope_per_bit"], 5.68, abs_tol=0.01)
check("analysis.continuous_inflection", an["continuous_inflection"]["second_derivative_zero_candidates_bits"][0], 3.51, abs_tol=0.02)

# =====================================================================
# 3. 해석적 유도 (analysis_proof.json)
# =====================================================================
pr = load_json("analysis_proof.json")
fp = pr["model"]["fitted_params"]
check("proof.S_max", fp["S_max"], 0.7960, abs_tol=0.002)
check("proof.lambda", fp["lambda"], 0.0932, abs_tol=0.002)
check("proof.beta", fp["beta"], 1.957, abs_tol=0.02)
check("proof.c_lin", fp["c_lin"], 0.00205, abs_tol=0.0002)
check("proof.Lr_max", fp["Lr_max"], 0.558, abs_tol=0.01)
check("proof.k", fp["k"], 1.396, abs_tol=0.02)
check("proof.xc", fp["xc"], 14.07, abs_tol=0.1)
check("proof.rmse_S", fp["rmse_S"], 2.2e-3, abs_tol=0.5e-3)
check("proof.rmse_L", fp["rmse_L"], 7.4e-10, abs_tol=1e-10)
check("proof.pcag_rmse", pr["model"]["pcag_rmse_discrete_vs_model"], 0.028, abs_tol=0.01)
root = pr["inflection_condition_3_1"]["roots"][0]
check("proof.b_star", root["b_star"], 4.19, abs_tol=0.02)
check("proof.x_star", root["x_star"], 11.81, abs_tol=0.05)
check("proof.h_prime_nonzero", root["h_prime_nonzero"], True)
check("proof.empirical_pchip_b", pr["inflection_condition_3_1"]["empirical_pchip_b"], 3.51, abs_tol=0.02)
check("proof.monte_carlo_mean_b", pr["inflection_condition_3_1"]["monte_carlo_mean_b"], 3.40, abs_tol=0.02)
check("proof.jevons_factorization_verified", pr["jevons_closed_form"]["factorization_verified"], True)

# =====================================================================
# 4. 민감도·Monte Carlo·θ 스윕 (sensitivity_summary.json)
# =====================================================================
se = load_json("sensitivity_summary.json")
mc = se["monte_carlo"]
check("mc.n_iter", mc["n_iter"], 3000)
check("mc.rel_sigma", mc["rel_sigma"], 0.03)
ib = mc["inflection_bits"]
check("mc.mean", ib["mean"], 3.40, abs_tol=0.02)
check("mc.std", ib["std"], 0.25, abs_tol=0.02)
check("mc.p5", ib["p5"], 2.99, abs_tol=0.02)     # 90% CI 하한
check("mc.p95", ib["p95"], 3.66, abs_tol=0.02)   # 90% CI 상한
check("mc.wall_prob.INT4to3", mc["wall_transition_prob"]["INT4->INT3"], 0.668, abs_tol=0.01)
check("mc.wall_prob.INT3to2", mc["wall_transition_prob"]["INT3->INT2"], 0.202, abs_tol=0.01)
check("mc.wall_prob.INT8to4", mc["wall_transition_prob"]["INT8->INT4"], 0.115, abs_tol=0.01)

# θ 스윕 불변성: 주 분석(Llama)에서 θ < 5.68 구간 wall 전이는 INT4->INT3
tsm = se["theta_sweep_main"]["sweep"]
invariant = all(e["wall_transition"] == "INT4->INT3"
                for e in tsm if e["theta"] < 5.68)
check("theta_sweep.invariant_below_5.68", invariant, True)
check("theta_sweep.wall_slope_at_theta3",
      next(e["wall_slope"] for e in tsm if e["theta"] == 3.0), 5.68, abs_tol=0.01)

# 다중 모델 INT4->INT3 기울기 (4.3절 표)
EXP_MULTI_SLOPE = {
    "meta-llama/Llama-3-8B": 5.60,
    "Qwen/Qwen2.5-7B": 9.78,
    "google/gemma-2-9b": 6.49,
    "mistralai/Mistral-7B": 6.96,
}
tm = se["theta_sweep_multimodel"]
for model, exp in EXP_MULTI_SLOPE.items():
    slope = next(s for s in tm[model]["slopes_per_bit"]
                 if s["from"] == "INT4" and s["to"] == "INT3")["slope_per_bit"]
    check(f"multimodel.slope[{model}]", slope, exp, abs_tol=0.02)

# Jevons 그리드 — 부하 증가 영역 비율
check("jevons_grid.region_fraction", se["jevons_grid"]["jevons_region_fraction"], 0.656, abs_tol=0.002)

# =====================================================================
# 5. Jevons 시나리오 (jevons_summary.json)
# =====================================================================
jv = load_json("jevons_summary.json")
check("jevons.E_d", jv["summary"]["E_d"], 1.5)
# INT4 대응 절감률(s=0.55 그리드)에서의 수요/부하
s55 = next(r for r in jv["summary"]["records"] if abs(r["savings"] - 0.55) < 0.01)
check("jevons.demand_growth_pct", s55["demand_growth_pct"], 231.0, abs_tol=2.0)
check("jevons.load_change_pct", s55["load_change_pct"], 49.0, abs_tol=1.0)

# 폐형 교차검증: (1-s)^(1-E_d) - 1
import math
closed = (1 - 0.55) ** (1 - 1.5) - 1
check("jevons.closed_form_load_change", closed * 100, 49.0, abs_tol=1.0)

# =====================================================================
# 6. 조건식 3.2 ↔ 3.1 일관성 (analysis_proof.json, 해석적 다리)
# =====================================================================
cs = pr["condition_3_2_3_1_consistency"]
check("consistency.max_discrete_drop_interval", cs["model_max_discrete_drop_interval"], "4->3")
check("consistency.continuous_peak_b", cs["continuous_peak_collapse"]["b_peak"], 4.19, abs_tol=0.02)
check("consistency.discrete_matches_continuous", cs["discrete_matches_continuous"], True)
check("consistency.model_drop_4to3", cs["model_discrete_per_bit_drop"]["4->3"], 6.68, abs_tol=0.1)

# =====================================================================
# 7. 부트스트랩 통계 추론 (statistics_summary.json)
# =====================================================================
st = load_json("statistics_summary.json")
check("boot.n_boot", st["n_boot"], 3000)
check("boot.seed", st["seed"], 20260901)
# 강건한 정량: 연속 변곡점 (조건식 3.1) CI — MC(3.40±0.25)와 일치
ib = st["inflection_bits"]
check("boot.inflection.mean", ib["mean"], 3.39, abs_tol=0.03)
check("boot.inflection.ci90_low", ib["ci90_low"], 2.99, abs_tol=0.03)
check("boot.inflection.ci90_high", ib["ci90_high"], 3.66, abs_tol=0.03)
# 강건한 정량: 심층 양자화 PCAG (INT2) — 안정
check("boot.pcag.INT2.mean", st["pcag"]["INT2"]["mean"], 2.22, abs_tol=0.05)
check("boot.pcag.INT2.ci90_high", st["pcag"]["INT2"]["ci90_high"], 2.66, abs_tol=0.05)
# 정직한 한계: 이산 기울기 유의성은 불충분 (estimability)
check("boot.test.conclusion", st["test_INT4to3_gt_theta3"]["conclusion"], "not-significant")
check("boot.test.p", st["test_INT4to3_gt_theta3"]["p"], 0.60, abs_tol=0.05)
check("boot.diff.p_gt_0", st["slope_diff_INT4to3_minus_INT8to4"]["p_gt_0"], 0.77, abs_tol=0.03)

# =====================================================================
# 8. 모델-형 강건성 (model_form_summary.json)
# =====================================================================
mf = load_json("model_form_summary.json")
check("modelform.n_inflection_in_domain", mf["n_inflection_in_domain"], 4)
check("modelform.b_star_min", mf["b_star_min"], 4.19, abs_tol=0.02)
check("modelform.b_star_max", mf["b_star_max"], 4.27, abs_tol=0.02)
check("modelform.all_in_INT4_adjacent", mf["b_star_all_in_INT4_adjacent"], True)

# =====================================================================
# 실행
# =====================================================================
def main():
    brief = "--brief" in sys.argv
    n_fail = 0
    n_pass = 0
    for label, actual, expected, abs_tol, rel_tol in CHECKS:
        ok = _close(actual, expected, abs_tol, rel_tol)
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        if not brief or not ok:
            mark = "PASS" if ok else "FAIL"
            print(f"[{mark}] {label}: got={_f(actual)} expect={_f(expected)}")
    print("-" * 60)
    print(f"검증 결과: {n_pass} PASS / {n_fail} FAIL / 총 {len(CHECKS)}")
    if n_fail:
        print("주의: GPU 실측으로 앵커가 교체된 경우, 실패 항목이 곧 갱신 필요한 논문 수치입니다.")
        sys.exit(1)
    return 0


def _close(actual, expected, abs_tol, rel_tol):
    if isinstance(expected, bool):
        return actual == expected
    try:
        a, e = float(actual), float(expected)
    except (TypeError, ValueError):
        return str(actual) == str(expected)
    if a == e:
        return True
    return abs(a - e) <= abs_tol + rel_tol * abs(e)


def _f(x):
    try:
        return f"{float(x):.6g}"
    except (TypeError, ValueError):
        return str(x)


if __name__ == "__main__":
    main()
