# experiments/sensitivity.py
# GPU 없이 수행하는 민감도·강건성 분석 3종.
#
# (A) θ 임계값 스윕 — 조건식 3.2의 |ΔPCAG/Δb| > θ 판정이 θ 선택에 얼마나 의존하는가?
# (B) Jevons 탄력성 그리드 — E_d × 절감율 → 총 그리드 부하 변화(%)
#     해석적 폐형: TotalLoad/L0 = (1-s)^(1-E_d)  ⇒  E_d>1 이면 임의 s>0에서 부하 증가.
#     (jevons_model.py 의 Q(P)=Q0(P/P0)^(-E_d) 모델에서 직접 유도됨)
# (C) Monte Carlo 강건성 — 앵커 데이터 불확실성(σ=3% 상대 오차) 하에서
#     Power Wall 위치(INT4→INT3)와 연속형 변곡점(조건식 3.1)이 보존되는가?
#
# 출력: experiments/sensitivity_summary.json
import os
import csv
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_MAIN = os.path.join(HERE, "results_raw.csv")
CSV_MULTI = os.path.join(HERE, "results_multimodel_raw.csv")
OUT_JSON = os.path.join(HERE, "sensitivity_summary.json")

PREC_BITS = {"FP16": 16, "INT8": 8, "INT6": 6, "INT5": 5, "INT4": 4,
             "INT3": 3, "INT2": 2, "FP4": 4}
RNG = np.random.default_rng(42)  # 재현 가능성


def load_csv(path):
    models = {}
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["Precision"]:
                continue
            try:
                acc = float(r["Accuracy_Score"])
                energy = float(r["Total_Energy_J"])
            except (TypeError, ValueError):
                # 정확도/에너지 누락 행 (예: --no_eval 실측) → 민감도 분석에서 제외
                skipped += 1
                continue
            models.setdefault(r["Model_Name"], []).append({
                "precision": r["Precision"],
                "bits": PREC_BITS[r["Precision"]],
                "accuracy": acc,
                "energy": energy,
            })
    if skipped:
        print(f"[sensitivity.load_csv] 데이터 누락 행 {skipped}개 스킵")
    for m in models:
        models[m].sort(key=lambda x: x["bits"], reverse=True)
    return models


def pcag_effs(points):
    """기준=FP16. 운영 PCAG = (상대 전력 절감)/(상대 정확도 손실)."""
    P0, A0 = points[0]["energy"], points[0]["accuracy"]
    out = []
    for p in points:
        dP, dA = P0 - p["energy"], A0 - p["accuracy"]
        out.append((dP / P0) / (dA / A0) if dA > 0 and dP > 0 else None)
    return out


def discrete_slopes(points, effs):
    pts = [(p, e) for p, e in zip(points, effs) if e is not None]
    res = []
    for i in range(len(pts) - 1):
        (a, ea), (b, eb) = pts[i], pts[i + 1]
        res.append({
            "from": a["precision"], "to": b["precision"],
            "from_bits": a["bits"], "to_bits": b["bits"],
            "slope_per_bit": abs((eb - ea) / (b["bits"] - a["bits"])),
        })
    return res


# ---------------------------------------------------------------- (A) θ 스윕
def theta_sweep(points, thetas):
    effs = pcag_effs(points)
    slopes = discrete_slopes(points, effs)
    records = []
    for th in thetas:
        viol = [s for s in slopes if s["slope_per_bit"] > th]
        wall = max(viol, key=lambda s: s["slope_per_bit"]) if viol else None
        records.append({
            "theta": round(float(th), 3),
            "wall_transition": (f"{wall['from']}->{wall['to']}" if wall else None),
            "wall_slope": (round(wall["slope_per_bit"], 3) if wall else None),
        })
    return {"slopes_per_bit": slopes, "sweep": records}


# ---------------------------------------------------- (B) Jevons 탄력성 그리드
def jevons_grid(e_d_grid, s_grid):
    """폐형: load_ratio = (1-s)^(1-E_d). 반환 matrix[i][j] = load_change_pct."""
    mat = np.empty((len(e_d_grid), len(s_grid)))
    for i, ed in enumerate(e_d_grid):
        for j, s in enumerate(s_grid):
            mat[i, j] = ((1.0 - s) ** (1.0 - ed) - 1.0) * 100.0
    # Jevons 역설 성립 영역(부하 증가) 비율
    jevons_frac = float((mat > 0).mean())
    return mat, jevons_frac


# ------------------------------------------- (C) Monte Carlo 앵커 강건성
def mc_robustness(points, n_iter=3000, rel_sigma=0.03):
    """앵커 acc/energy에 로그정규 상대 오차 주입 → Power Wall 전이/변곡점 분포."""
    wall_counts = {}
    inflections = []
    n_valid = 0
    for _ in range(n_iter):
        pert = []
        for p in points:
            pert.append({
                **p,
                "accuracy": p["accuracy"] * float(RNG.lognormal(-0.5 * rel_sigma ** 2, rel_sigma)),
                "energy": p["energy"] * float(RNG.lognormal(-0.5 * rel_sigma ** 2, rel_sigma)),
            })
        pert.sort(key=lambda x: x["bits"], reverse=True)
        effs = pcag_effs(pert)
        slopes = discrete_slopes(pert, effs)
        viol = [s for s in slopes if s["slope_per_bit"] > 3.0]
        if viol:
            wall = max(viol, key=lambda s: s["slope_per_bit"])
            key = f"{wall['from']}->{wall['to']}"
            wall_counts[key] = wall_counts.get(key, 0) + 1
            n_valid += 1
        # 연속형 변곡점(조건식 3.1) — PCHIP 이계도함수 영교차
        pts = [(p["bits"], e) for p, e in zip(pert, effs) if e is not None]
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
                        # 선형 보간으로 근 위치 정제
                        b0 = grid[i] + (grid[i + 1] - grid[i]) * abs(d2[i]) / (
                            abs(d2[i]) + abs(d2[i + 1]) + 1e-30)
                        inflections.append(float(b0))
                        break
            except Exception:
                pass
    return {
        "n_iter": n_iter,
        "rel_sigma": rel_sigma,
        "wall_transition_counts": wall_counts,
        "wall_transition_prob": {k: round(v / max(n_valid, 1), 4)
                                 for k, v in wall_counts.items()},
        "inflection_bits": {
            "n": len(inflections),
            "mean": round(float(np.mean(inflections)), 3) if inflections else None,
            "std": round(float(np.std(inflections)), 3) if inflections else None,
            "p5": round(float(np.percentile(inflections, 5)), 3) if inflections else None,
            "p95": round(float(np.percentile(inflections, 95)), 3) if inflections else None,
        },
        "_inflection_samples": [round(x, 3) for x in inflections],
    }


def main():
    models = load_csv(CSV_MAIN)
    print("=== (A) θ 임계값 스윕 (Llama-3-8B) ===")
    thetas = np.arange(0.5, 10.01, 0.25)
    sweepA = theta_sweep(models["meta-llama/Llama-3-8B"], thetas)
    for s in sweepA["slopes_per_bit"]:
        print(f"  slope {s['from']}->{s['to']}: {s['slope_per_bit']:.3f} /bit")
    changes = []
    for rec in sweepA["sweep"]:
        if not changes or changes[-1]["wall_transition"] != rec["wall_transition"]:
            changes.append(rec)
    for c in changes:
        print(f"  θ={c['theta']:.2f} → wall={c['wall_transition']} "
              f"(slope {c['wall_slope']})")

    print("\n=== (A-2) 다중 모델 θ 스윕 ===")
    multi = load_csv(CSV_MULTI)
    sweepA_multi = {}
    for name, pts in multi.items():
        sw = theta_sweep(pts, thetas)
        sweepA_multi[name] = sw
        walls = {}
        for rec in sw["sweep"]:
            walls.setdefault(rec["wall_transition"], 0)
            walls[rec["wall_transition"]] += 1
        print(f"  {name}: {walls}")

    print("\n=== (B) Jevons 탄력성 그리드 (폐형 (1-s)^(1-E_d)) ===")
    e_d_grid = np.round(np.linspace(0.2, 2.8, 40), 3)
    s_grid = np.round(np.linspace(0.0, 0.70, 36), 4)
    mat, jevons_frac = jevons_grid(e_d_grid, s_grid)
    print(f"  그리드 {mat.shape}, 부하 증가 영역 비율: {jevons_frac*100:.1f}%")
    # 검증: E_d=1.5, s=0.55 → (0.45)^(-0.5)-1 ≈ +49.07%
    chk = ((1 - 0.55) ** (1 - 1.5) - 1) * 100
    print(f"  검증 E_d=1.5, s=55%: 폐형 +{chk:.2f}% (jevons_summary.json의 +49.07%와 일치해야)")

    print("\n=== (C) Monte Carlo 강건성 (σ=3%, N=3000) ===")
    mc = mc_robustness(models["meta-llama/Llama-3-8B"], n_iter=3000, rel_sigma=0.03)
    print(f"  Wall 전이 분포: {mc['wall_transition_counts']}")
    print(f"  변곡점 bits: mean={mc['inflection_bits']['mean']} "
          f"std={mc['inflection_bits']['std']} "
          f"90%CI=[{mc['inflection_bits']['p5']}, {mc['inflection_bits']['p95']}]")

    out = {
        "theta_sweep_main": sweepA,
        "theta_sweep_multimodel": {k: {"slopes_per_bit": v["slopes_per_bit"],
                                       "sweep": v["sweep"]}
                                   for k, v in sweepA_multi.items()},
        "jevons_grid": {
            "e_d_grid": e_d_grid.tolist(),
            "savings_grid": s_grid.tolist(),
            "load_change_pct_matrix": np.round(mat, 3).tolist(),
            "jevons_region_fraction": round(jevons_frac, 4),
            "closed_form": "load_ratio = (1-s)^(1-E_d)",
        },
        "monte_carlo": {k: v for k, v in mc.items() if k != "_inflection_samples"},
        "_mc_inflection_samples": mc["_inflection_samples"],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n요약 저장: {OUT_JSON}")


if __name__ == "__main__":
    main()
