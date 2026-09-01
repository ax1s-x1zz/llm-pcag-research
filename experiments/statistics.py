# experiments/statistics.py
# 부트스트랩 기반 통계 추론 — PCAG·Power Wall 판정의 불확실성 정량화.
#
# 기존 sensitivity.py의 Monte Carlo(σ=3%, N=3000)는 "결론이 앵커 불확실성에 보존되는가"를
# 강건성 관점에서 보여준다. 본 모듈은 이를 **통계적 추론(statistical inference)**으로 격상한다:
#   (1) 각 정밀도의 PCAG에 대한 부트스트랩 신뢰구간(백분위수 90% CI).
#   (2) Power Wall 판정에 쓰이는 이산 기울기(INT8→INT4 vs INT4→INT3)에 대한 CI.
#   (3) **가설 검정**: "INT4→INT3 기울기가 INT8→INT4보다 유의하게 크다" (paired bootstrap)
#       및 "INT4→INT3 기울기가 θ=3을 유의하게 초과한다".
#   (4) 연속형 변곡점(조건식 3.1)의 부트스트랩 CI.
#
# 앵커 불확실성은 기존 MC와 동일하게 로그정규 상대 오차(σ=3%)로 모델링한다.
# 재현성을 위해 전용 시드(부트스트랩 RNG)를 사용한다.
#
# 출력: experiments/statistics_summary.json
import os
import csv
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_MAIN = os.path.join(HERE, "results_raw.csv")
OUT_JSON = os.path.join(HERE, "statistics_summary.json")

PREC_BITS = {"FP16": 16, "INT8": 8, "INT4": 4, "INT3": 3, "INT2": 2}
N_BOOT = 3000          # 부트스트랩 반복
REL_SIGMA = 0.03       # 앵커 상대 오차 (MC와 동일)
THETA = 3.0            # 조건식 3.2 임계
BOOT_SEED = 20260901   # 부트스트랩 전용 고정 시드


def load_main():
    rows = []
    with open(CSV_MAIN, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["Precision"]:
                continue
            rows.append({
                "precision": r["Precision"], "bits": PREC_BITS[r["Precision"]],
                "accuracy": float(r["Accuracy_Score"]),
                "energy": float(r["Total_Energy_J"]),
            })
    rows.sort(key=lambda x: x["bits"], reverse=True)
    return rows


def perturb(points, rng):
    """앵커에 로그정규 상대 오차 주입. 원본 변형 없이 새 리스트 반환."""
    out = []
    for p in points:
        out.append({
            **p,
            "accuracy": p["accuracy"] * float(rng.lognormal(-0.5 * REL_SIGMA ** 2, REL_SIGMA)),
            "energy": p["energy"] * float(rng.lognormal(-0.5 * REL_SIGMA ** 2, REL_SIGMA)),
        })
    return out


def compute_metrics(points):
    """점들로부터 PCAG(정밀도별), 이산 기울기, 연속 변곡점(조건식 3.1) 추출."""
    P0, A0 = points[0]["energy"], points[0]["accuracy"]
    effs = {}
    for p in points:
        dP, dA = P0 - p["energy"], A0 - p["accuracy"]
        effs[p["precision"]] = (dP / P0) / (dA / A0) if dA > 0 and dP > 0 else None

    # 이산 기울기 (정의 3.2 분자 |ΔPCAG/Δb|)
    prec_order = ["FP16", "INT8", "INT4", "INT3", "INT2"]
    slopes = {}
    for a, b in zip(prec_order, prec_order[1:]):
        if effs[a] is not None and effs[b] is not None:
            db = PREC_BITS[a] - PREC_BITS[b]
            slopes[f"{a}->{b}"] = abs((effs[b] - effs[a]) / db)

    # 연속 변곡점 (PCHIP 이계도함수 영교차)
    inf = None
    pts = [(p["bits"], effs[p["precision"]]) for p in points if effs[p["precision"]] is not None]
    if len(pts) >= 3:
        b = np.array([x[0] for x in pts], float)
        y = np.array([x[1] for x in pts], float)
        order = np.argsort(b)
        try:
            from scipy.interpolate import PchipInterpolator
            interp = PchipInterpolator(b[order], y[order])
            grid = np.linspace(b.min(), b.max(), 400)
            h = 1e-4
            d2 = (interp(grid + h) - 2 * interp(grid) + interp(grid - h)) / (h * h)
            sgn = np.sign(d2)
            for i in range(len(sgn) - 1):
                if sgn[i] != sgn[i + 1] and sgn[i] != 0:
                    inf = float(grid[i] + (grid[i + 1] - grid[i]) * abs(d2[i])
                                / (abs(d2[i]) + abs(d2[i + 1]) + 1e-30))
                    break
        except Exception:
            inf = None
    return effs, slopes, inf


def bootstrap(points, n=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    pcag_boot = {k: [] for k in ["INT8", "INT4", "INT3", "INT2"]}
    slope_boot = {"INT8->INT4": [], "INT4->INT3": []}
    inf_boot = []
    # paired 검정용: 두 기울기가 모두 존재하는 반복을 저장
    pair_slopes_84 = []
    pair_slopes_43 = []
    for _ in range(n):
        effs, slopes, inf = compute_metrics(perturb(points, rng))
        for k in pcag_boot:
            if effs.get(k) is not None:
                pcag_boot[k].append(effs[k])
        for k in slope_boot:
            if k in slopes:
                slope_boot[k].append(slopes[k])
        if "INT8->INT4" in slopes and "INT4->INT3" in slopes:
            pair_slopes_84.append(slopes["INT8->INT4"])
            pair_slopes_43.append(slopes["INT4->INT3"])
        if inf is not None:
            inf_boot.append(inf)

    def summ(a):
        a = np.asarray(a, float)
        return {
            "n": int(len(a)),
            "mean": round(float(a.mean()), 3),
            "std": round(float(a.std(ddof=1)), 3),
            "ci90_low": round(float(np.percentile(a, 5)), 3),
            "ci90_high": round(float(np.percentile(a, 95)), 3),
        }

    # Paired hypothesis tests (같은 반복의 두 기울기를 대응시킴)
    d_slopes = np.array(pair_slopes_43) - np.array(pair_slopes_84)
    # one-sided: P(INT4→3 > INT8→4)
    p_diff = float((d_slopes > 0).mean())
    # one-sided: P(INT4→3 > θ)
    p_theta = float((np.array(pair_slopes_43) > THETA).mean())

    return {
        "n_boot": n,
        "rel_sigma": REL_SIGMA,
        "theta": THETA,
        "seed": seed,
        "pcag": {k: summ(v) for k, v in pcag_boot.items() if v},
        "slopes": {k: summ(v) for k, v in slope_boot.items() if v},
        "slope_diff_INT4to3_minus_INT8to4": {
            "n_paired": int(len(d_slopes)),
            "mean": round(float(d_slopes.mean()), 3),
            "ci90_low": round(float(np.percentile(d_slopes, 5)), 3),
            "ci90_high": round(float(np.percentile(d_slopes, 95)), 3),
            "p_gt_0": round(p_diff, 4),
        },
        "test_INT4to3_gt_theta3": {
            "p": round(p_theta, 4),
            "conclusion": ("significant" if p_theta >= 0.95 else "not-significant"),
        },
        "inflection_bits": summ(inf_boot) if inf_boot else {"n": 0},
    }


def main():
    points = load_main()
    print("=== 부트스트랩 통계 추론 (앵커 σ=3%, N=3000, 시드 20260901) ===")
    res = bootstrap(points)
    print("\n-- PCAG 신뢰구간 (90% CI) --")
    for k, v in res["pcag"].items():
        print(f"  {k:<5} mean={v['mean']} 90%CI=[{v['ci90_low']}, {v['ci90_high']}] (n={v['n']})")
    print("\n-- 이산 기울기 신뢰구간 --")
    for k, v in res["slopes"].items():
        print(f"  {k:<12} mean={v['mean']} 90%CI=[{v['ci90_low']}, {v['ci90_high']}]")
    print("\n-- Paired 검정: INT4→INT3 vs INT8→INT4 --")
    dd = res["slope_diff_INT4to3_minus_INT8to4"]
    print(f"  Δslope mean={dd['mean']} 90%CI=[{dd['ci90_low']}, {dd['ci90_high']}] "
          f"P(Δ>0)={dd['p_gt_0']}")
    print(f"  [INT4→3 > θ=3] P={res['test_INT4to3_gt_theta3']['p']} "
          f"→ {res['test_INT4to3_gt_theta3']['conclusion']}")
    inf = res["inflection_bits"]
    if inf["n"]:
        print(f"\n-- 연속 변곡점 (조건식 3.1) --")
        print(f"  mean={inf['mean']} 90%CI=[{inf['ci90_low']}, {inf['ci90_high']}] (n={inf['n']})")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {OUT_JSON}")


if __name__ == "__main__":
    main()
