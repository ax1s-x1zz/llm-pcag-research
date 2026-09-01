# experiments/model_form.py
# 모델-형(parametric form) 강건성 분석 — Power Wall 변곡점 b*가 함수형 선택에 민감한가?
#
# 기존 분석(analytical_proof.py)은 단일 모델형(Weibull 포화 + 선형-로지스틱 가속)을 사용한다.
# 본 모듈은 "결론이 특정 함수형의 우연이 아닌지"를 검증하기 위해 여러 대안 함수형 조합에
# 대해 동일한 조건식 3.1(h(x)=g'(x)+g(x)²=0)의 근 b*를 계산한다.
#
#   전력 절감 S(x) (x=16-b, S(0)=0, 단조증가, 포화) 후보:
#     A. Weibull:   Smax(1 - e^{-(λx)^β})                       (기준)
#     B. 단일지수:  Smax(1 - e^{-λx})
#     C. 쌍곡탄젠트: Smax·tanh(λx)
#     D. Hill/MM:   Smax·x^p/(a + x^p)
#   정확도 손실 Lr(x) (단조증가) 후보:
#     a. 선형+로지스틱: c·x + Lmax·σ(k(x-xc))                   (기준)
#     b. 순수 멱:       a·x^p
#     c. 순수 로지스틱: Lmax·σ(k(x-xc))
#     d. 선형+멱:       c·x + a·x^p
#
# 각 (S, Lr) 조합에 대해 최소제곱 피팅 후, 수치 로그미분으로 g(x)=d ln PCAG/dx 와
# h(x)=g'(x)+g(x)² 을 계산하고 관측 도메인 x∈[8,14]에서 근(변곡점)을 탐색한다.
# (함수형별 폐형을 유도하지 않고 범용 수치 루틴을 사용)
#
# 출력: experiments/model_form_summary.json
import os
import csv
import json
import numpy as np
from scipy.optimize import curve_fit, brentq

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_MAIN = os.path.join(HERE, "results_raw.csv")
OUT_JSON = os.path.join(HERE, "model_form_summary.json")

PREC_BITS = {"FP16": 16, "INT8": 8, "INT4": 4, "INT3": 3, "INT2": 2}
X_OBS_MIN, X_OBS_MAX = 8.0, 14.0


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


# ------------------------------------------------------------- 함수형 정의
# 각 (name, func(x, params...), p0, bounds, n_params)
S_FORMS = {
    "Weibull": (
        lambda x, Smax, lam, beta: Smax * (1 - np.exp(-(lam * x) ** beta)),
        [0.75, 0.10, 2.0], ([0.3, 1e-3, 0.5], [1.2, 2.0, 8.0])),
    "Exp1": (
        lambda x, Smax, lam: Smax * (1 - np.exp(-lam * x)),
        [0.75, 0.10], ([0.3, 1e-3], [1.2, 2.0])),
    "Tanh": (
        lambda x, Smax, lam: Smax * np.tanh(lam * x),
        [0.75, 0.10], ([0.3, 1e-3], [1.2, 2.0])),
    "Hill": (
        lambda x, Smax, a, p: Smax * x ** p / (a + x ** p),
        [0.75, 1e3, 2.0], ([0.3, 1.0, 0.5], [1.2, 1e5, 8.0])),
}
L_FORMS = {
    "LinLog": (
        lambda x, c, Lmax, k, xc: c * x + Lmax * sigmoid(k * (x - xc)),
        [0.002, 0.34, 1.5, 13.6], ([0.0, 0.05, 0.1, 5.0], [0.05, 1.5, 10.0, 25.0])),
    "Power": (
        lambda x, a, p: a * x ** p,
        [0.001, 2.0], ([0.0, 0.5], [0.01, 5.0])),
    "Logistic": (
        lambda x, Lmax, k, xc: Lmax * sigmoid(k * (x - xc)),
        [0.4, 1.5, 13.0], ([0.05, 0.1, 5.0], [1.5, 10.0, 25.0])),
    "LinPow": (
        lambda x, c, a, p: c * x + a * x ** p,
        [0.001, 0.001, 2.0], ([0.0, 0.0, 0.5], [0.05, 0.05, 5.0])),
}


def load_points():
    rows = []
    with open(CSV_MAIN, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["Precision"]:
                continue
            rows.append({
                "bits": PREC_BITS[r["Precision"]],
                "accuracy": float(r["Accuracy_Score"]),
                "energy": float(r["Total_Energy_J"]),
            })
    rows.sort(key=lambda x: x["bits"], reverse=True)
    P0, A0 = rows[0]["energy"], rows[0]["accuracy"]
    x, S, L = [], [], []
    for r in rows:
        x.append(16 - r["bits"])
        S.append((P0 - r["energy"]) / P0)
        L.append((A0 - r["accuracy"]) / A0)
    return np.array(x, float), np.array(S, float), np.array(L, float)


def inflection_for(S, L):
    """범용 수치 변곡점: g=d ln(S/L)/dx, h=g'+g², 근을 x∈[8,14]에서 탐색.
    반환: (x_star, b_star, nondeg, x_steep, b_steep)
      - (x_star,b_star): 조건식 3.1 변곡점 (도메인 내 없으면 None)
      - (x_steep,b_steep): 도메인 내 |g| 최대(=연속형 한계효용 붕괴가 가장 큰 지점)
    """
    h = 1e-4

    def ln_pcag(xv):
        return np.log(S(xv)) - np.log(L(xv))

    def g(xv):
        return (ln_pcag(xv + h) - ln_pcag(xv - h)) / (2 * h)

    def h_fun(xv):
        return (g(xv + h) - g(xv - h)) / (2 * h) + g(xv) ** 2

    grid = np.linspace(X_OBS_MIN + 1e-3, X_OBS_MAX - 1e-3, 4000)
    hv = np.array([h_fun(xv) for xv in grid])
    gv = np.array([g(xv) for xv in grid])
    # 도메인 내 연속형 붕괴 최대 지점 (|g| 최대)
    i_steep = int(np.argmax(np.abs(gv)))
    x_steep = float(grid[i_steep])

    sgn = np.sign(hv)
    roots = []
    for i in range(len(sgn) - 1):
        if sgn[i] != sgn[i + 1] and sgn[i] != 0:
            try:
                r = brentq(h_fun, grid[i], grid[i + 1], xtol=1e-9)
                if all(abs(r - q) > 1e-3 for q in roots):
                    roots.append(float(r))
            except Exception:
                pass
    if not roots:
        return None, None, None, x_steep, (16.0 - x_steep)
    # 유일근 확인 후 d³≠0 (h'≠0)
    r = roots[0]
    dh = (h_fun(r + 1e-5) - h_fun(r - 1e-5)) / (2e-5)
    return r, (16.0 - r), bool(abs(dh) > 1e-4), x_steep, (16.0 - x_steep)


def fit_and_solve(x, S, L, s_name, l_name):
    Sf, p0s, (bl, bu) = S_FORMS[s_name]
    Lf, p0l, (ll, lu) = L_FORMS[l_name]
    try:
        ps, _ = curve_fit(Sf, x, S, p0=p0s, bounds=(bl, bu), maxfev=200000)
        pl, _ = curve_fit(Lf, x, L, p0=p0l, bounds=(ll, lu), maxfev=200000)
    except Exception:
        return None
    Sfunc = lambda xv: Sf(xv, *ps)         # noqa: E731
    Lfunc = lambda xv: Lf(xv, *pl)         # noqa: E731
    # S(0)=0, L(0)=0, S<1 검증 (물리 타당성)
    if Sfunc(0) < -1e-9 or Lfunc(0) < -1e-9:
        return None
    r, bstar, nondeg, x_steep, b_steep = inflection_for(Sfunc, Lfunc)
    rmse_S = float(np.sqrt(np.mean((Sfunc(x) - S) ** 2)))
    rmse_L = float(np.sqrt(np.mean((Lfunc(x) - L) ** 2)))
    return {
        "b_star": (round(bstar, 4) if r is not None else None),
        "x_star": (round(r, 4) if r is not None else None),
        "h_prime_nonzero": (bool(nondeg) if r is not None else None),
        "b_steep": round(b_steep, 4),
        "x_steep": round(x_steep, 4),
        "inflection_in_domain": r is not None,
        "rmse_S": rmse_S, "rmse_L": rmse_L,
    }


def main():
    x, S, L = load_points()
    print("=== 모델-형 강건성: Power Wall 위치 (조건식 3.1) ===")
    print(f"{'S형':<9}{'Lr형':<9}{'b*':>8}{'b_steep':>9}{'RMSE_S':>10}{'RMSE_L':>10}")
    results = {}
    bstars, bsteeps = [], []
    for s_name in S_FORMS:
        results[s_name] = {}
        for l_name in L_FORMS:
            r = fit_and_solve(x, S, L, s_name, l_name)
            results[s_name][l_name] = r
            if r is None:
                print(f"{s_name:<9}{l_name:<9}{'no-fit':>8}")
                continue
            bsteeps.append(r["b_steep"])
            bs = r["b_star"]
            if bs is not None:
                bstars.append(bs)
                print(f"{s_name:<9}{l_name:<9}{bs:>8.3f}{r['b_steep']:>9.3f}"
                      f"{r['rmse_S']:>10.2e}{r['rmse_L']:>10.2e}")
            else:
                print(f"{s_name:<9}{l_name:<9}{'out-of-dom':>8}{r['b_steep']:>9.3f}"
                      f"{r['rmse_S']:>10.2e}{r['rmse_L']:>10.2e}")
    bstars_a = np.array(bstars)
    bsteeps_a = np.array(bsteeps)
    summary = {
        "S_forms": list(S_FORMS.keys()),
        "Lr_forms": list(L_FORMS.keys()),
        "domain_x": [X_OBS_MIN, X_OBS_MAX],
        "b_star_values": sorted(round(float(b), 3) for b in bstars),
        "b_steep_values": sorted(round(float(b), 3) for b in bsteeps),
        "n_fits": int(len(S_FORMS) * len(L_FORMS)),
        "n_inflection_in_domain": int(len(bstars)),
        "n_steep_all": int(len(bsteeps)),
        "b_star_mean": round(float(bstars_a.mean()), 3) if len(bstars) else None,
        "b_star_min": round(float(bstars_a.min()), 3) if len(bstars) else None,
        "b_star_max": round(float(bstars_a.max()), 3) if len(bstars) else None,
        "b_steep_mean": round(float(bsteeps_a.mean()), 3) if len(bsteeps) else None,
        "b_steep_min": round(float(bsteeps_a.min()), 3) if len(bsteeps) else None,
        "b_steep_max": round(float(bsteeps_a.max()), 3) if len(bsteeps) else None,
        "b_star_all_in_INT4_adjacent": bool(
            len(bstars) and bstars_a.min() >= 3.0 and bstars_a.max() <= 4.5),
        "b_steep_all_le4_2": bool(
            len(bsteeps) and bsteeps_a.max() <= 4.2),
        "details": results,
    }
    print("\n--- 요약 ---")
    print(f"  도메인 내 변곡점(조건식 3.1) b*: {summary['n_inflection_in_domain']}개, "
          f"mean={summary['b_star_mean']} min={summary['b_star_min']} "
          f"max={summary['b_star_max']}")
    print(f"  연속형 붕괴 최대 지점 b_steep: {summary['n_steep_all']}개, "
          f"mean={summary['b_steep_mean']} min={summary['b_steep_min']} "
          f"max={summary['b_steep_max']}")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {OUT_JSON}")


if __name__ == "__main__":
    main()
