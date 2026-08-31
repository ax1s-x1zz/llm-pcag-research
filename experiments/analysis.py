# experiments/analysis.py
# PCAG(Power Cost per Accuracy Gain) 수학적 분석 + Power Wall 식별 + 그림 생성.
#
# 참조 [논문 3.3절 V2]:
#   [공식 3.1] PCAGk = (Ak - A0)/(P0 - Pk)                       (이산형, 원문)
#   [공식 3.2] PCAG(b) = -dA/dP = -(dA/db)/(dP/db)               (연속형, 원문)
#   [조건식 3.1] d²PCAG/db² = 0  and  d³PCAG/db³ ≠ 0            (변곡점)
#   [조건식 3.2] |(PCAGk+1 - PCAGk)/(bk+1 - bk)| > θ            (한계효용 붕괴)
#
# ⚠️ 부호 관례 정합(재조정) — 자세한 근거는 logs/troubleshooting_archive.md ISSUE-PCAG-01 참고:
#   원문 [공식 3.1]은 양자화에 대해 항상 PCAG≤0 이 되어 3.3.2의 "상승구간 PCAG>0"과 모순된다.
#   따라서 운영(interpretable) PCAG를 "상대 전력 절감 / 상대 정확도 손실" 로 정의한다:
#     PCAG_k = ( (P0-Pk)/P0 ) / ( (A0-Ak)/A0 )
#   이 값은 양수이며, "단위 정확도 손실당 얼마나 전력을 절감했는가"의 효율성 지표이다.
#   양자화가 깊어질수록 전력 절감은 포화(분자 정체)되고 정확도 손실은 가속(분모 급증)하여
#   PCAG가 붕괴(collapse) → 이것이 물리적 Power Wall.
#   원문 [공식 3.1]의 값(ΔA/ΔP)도 transparenct하게 함께 보고한다.
import os
import json
import csv
import numpy as np

from schema import CSV_PATH

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
OUT_DIR = os.path.normpath(OUT_DIR)


def bit_width(precision):
    return {"FP16": 16, "INT8": 8, "INT4": 4, "INT3": 3, "INT2": 2}[precision]


def load_results(path=CSV_PATH):
    rows = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["Precision"]:
                continue
            try:
                acc = float(r["Accuracy_Score"])
                energy = float(r["Total_Energy_J"])
                power_w = float(r["Avg_Power_W"])
                vram = float(r["VRAM_GB"])
            except (TypeError, ValueError):
                # 정확도/에너지 누락 행 (예: --no_eval 실측) → 분석에서 제외
                skipped += 1
                continue
            rows.append({
                "precision": r["Precision"],
                "bits": bit_width(r["Precision"]),
                "accuracy": acc,
                "energy": energy,                 # J/1000 tokens (또는 배치 전체 J)
                "power_w": power_w,
                "vram": vram,
            })
    if skipped:
        print(f"[load_results] 정확도/에너지 누락 행 {skipped}개 스킵 "
              "(미완료 측정은 분석에서 제외)")
    # bits 기준 정렬
    rows.sort(key=lambda x: x["bits"], reverse=True)
    return rows


def compute_pcag(rows):
    """이산형 PCAG(원문 [3.1] 및 운영 정규화 버전) 계산. 기준=최대 bits(FP16)."""
    base = rows[0]  # FP16
    P0, A0 = base["energy"], base["accuracy"]
    out = []
    for r in rows:
        Pk, Ak = r["energy"], r["accuracy"]
        dP = P0 - Pk
        dA = A0 - Ak
        pcag_raw = dA / dP if dP != 0 else None                       # 원문 [3.1] (≤0)
        pcag_eff = (dP / P0) / (dA / A0) if dA > 0 else None          # 운영(효율) 정의
        out.append({
            **r,
            "P0": P0, "A0": A0,
            "dP": dP, "dA": dA,
            "pcag_raw": pcag_raw,
            "pcag_eff": pcag_eff,
        })
    return out


def fit_continuous_pcag(points_b, points_pcag):
    """연속형 PCAG(b) 곡선. PCHIP 보간(단조 보존).
    반환: (interp_fun, None). PCHIP는 x가 strictly increasing해야 하므로 정렬.
    실패 시 (None, None)."""
    try:
        from scipy.interpolate import PchipInterpolator
        b = np.array(points_b, float)
        y = np.array(points_pcag, float)
        order = np.argsort(b)
        interp = PchipInterpolator(b[order], y[order])
        return interp, None
    except Exception:
        return None, None


def discrete_slopes(data):
    """조건식 3.2용 이산 기울기: |(PCAGk+1 - PCAGk)/(bk+1 - bk)|"""
    pts = [d for d in data if d["pcag_eff"] is not None]
    res = []
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        slope = abs((b["pcag_eff"] - a["pcag_eff"]) / (b["bits"] - a["bits"]))
        res.append({"from": a["precision"], "to": b["precision"],
                    "from_bits": a["bits"], "to_bits": b["bits"],
                    "slope_per_bit": slope})
    return res


def find_power_wall(data, theta=3.0):
    """조건식 3.2 기반 Power Wall 후보 + 연속형 변곡점(3.1) 후보."""
    slopes = discrete_slopes(data)
    # 조건식 3.2: |slope| > θ 인 첫 번째(가장 얕은? -> 가장 급격한) 천이
    violations = [s for s in slopes if s["slope_per_bit"] > theta]
    wall = None
    if violations:
        # 가장 가파른 천이를 Power Wall 구간으로
        wall = max(violations, key=lambda s: s["slope_per_bit"])
    return slopes, wall


def make_fig1(data, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    xs = [d["energy"] for d in data]
    ys = [d["accuracy"] for d in data]
    labels = [d["precision"] for d in data]
    colors = {"FP16": "#1f77b4", "INT8": "#2ca02c", "INT4": "#d62728",
              "INT3": "#ff7f0e", "INT2": "#8c564b"}
    ax.plot(xs, ys, "-o", color="#333", lw=2, ms=8, zorder=2)
    for (x, y, lab) in zip(xs, ys, labels):
        ax.scatter(x, y, color=colors.get(lab, "#333"), s=110, zorder=3, edgecolor="white")
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(8, -2),
                    fontsize=11, fontweight="bold")
    ax.set_xlabel("Energy Consumption (J / 1,000 tokens)", fontsize=12)
    ax.set_ylabel("Inference Accuracy (%)", fontsize=12)
    ax.set_title("Fig 1. Accuracy vs Energy across Quantization (FP16→INT8→INT4)", fontsize=12)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()  # 에너지 큰 쪽(FP16)이 오른쪽? -> 양자화 방향 읽기 편하도록 좌->우 감소
    # 주석: FP16이 우상단 고정밀·고에너지, 아래로 갈수록 저에너지
    ax.set_xlim(xs[-1] - 8, xs[0] + 8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_accuracy_vs_energy.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig1_accuracy_vs_energy.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig1 저장: fig1_accuracy_vs_energy.{png,pdf}")


def make_fig2(data, out_dir, wall, interp=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    bits = [d["bits"] for d in data if d["pcag_eff"] is not None]
    pcag = [d["pcag_eff"] for d in data if d["pcag_eff"] is not None]
    prec = [d["precision"] for d in data if d["pcag_eff"] is not None]
    # 연속 곡선 오버레이
    if interp is not None:
        bgrid = np.linspace(interp.x.min(), interp.x.max(), 300)
        ax.plot(bgrid, [interp(b) for b in bgrid], color="#d62728", lw=1.2,
                ls=":", zorder=1)
    ax.plot(bits, pcag, "-o", color="#d62728", lw=2.5, ms=9, zorder=2)
    for (b, p, lab) in zip(bits, pcag, prec):
        ax.annotate(lab, (b, p), textcoords="offset points", xytext=(0, 8),
                    fontsize=11, fontweight="bold", ha="center")
    # Power Wall 위치 강조
    if wall:
        wall_bits = (wall["from_bits"] + wall["to_bits"]) / 2.0
        ax.axvline(wall_bits, color="#111", ls="--", lw=2)
        ax.annotate("POWER WALL", (wall_bits, max(pcag) * 0.2),
                    rotation=90, textcoords="data", fontsize=12,
                    fontweight="bold", color="#111", ha="right", va="bottom")
    ax.set_xlabel("Weight Bit Precision, b (bits)", fontsize=12)
    ax.set_ylabel("PCAG (relative power saving / relative accuracy loss)", fontsize=11)
    ax.set_title("Fig 2. PCAG Curve — Efficiency Collapse (Power Wall)", fontsize=12)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()  # b 큰(FP16) 쪽이 왼쪽? -> 읽기 편한 방향 조정
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_pcag_power_wall.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig2_pcag_power_wall.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig2 저장: fig2_pcag_power_wall.{png,pdf}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_results()
    print("=== 로드된 데이터 ===")
    for r in rows:
        print(f"  {r['precision']:<5} bits={r['bits']:>2} acc={r['accuracy']:>6.2f} "
              f"E={r['energy']:>6.1f}J/1k")

    data = compute_pcag(rows)
    print("\n=== 이산형 PCAG (기준 FP16) ===")
    print(f"  P0={data[0]['P0']} J/1k, A0={data[0]['A0']}%")
    def _fmt(v):
        return "n/a" if v is None else f"{v:.4f}"
    def _fmt2(v):
        return "n/a" if v is None else f"{v:.3f}"
    for d in data:
        print(f"  {d['precision']:<5} dP={d['dP']:>6.1f} dA={d['dA']:>6.2f} "
              f"PCAG_raw(ΔA/ΔP)={_fmt(d['pcag_raw'])}  "
              f"PCAG_eff={_fmt2(d['pcag_eff'])}")

    slopes, wall = find_power_wall(data)
    print("\n=== 조건식 3.2: 이산 한계효용 기울기 |ΔPCAG/Δb| ===")
    for s in slopes:
        flag = "  <-- > θ(POWER WALL 구간)" if (wall and s == wall) else ""
        print(f"  {s['from']}->{s['to']}: {s['slope_per_bit']:>7.3f} per bit{flag}")
    print(f"  -> Power Wall 후보 구간: {wall['from']} ~ {wall['to']} (bits "
          f"{wall['from_bits']}->{wall['to_bits']})" if wall else "  (미검출)")

    # 연속형 변곡점(조건식 3.1) 시도
    cont = None
    interp = None
    pcag_pts = [(d["bits"], d["pcag_eff"]) for d in data if d["pcag_eff"] is not None]
    if len(pcag_pts) >= 3:
        interp, _ = fit_continuous_pcag(
            [p[0] for p in pcag_pts], [p[1] for p in pcag_pts])
        if interp is not None:
            cont = detect_inflection_continuous(interp)
            print(f"  연속형 변곡점(조건식 3.1) 후보(bits): "
                  f"{cont.get('second_derivative_zero_candidates_bits', '미검출')}")

    summary = {
        "P0_energy": data[0]["P0"], "A0_acc": data[0]["A0"],
        "points": [
            {"precision": d["precision"], "bits": d["bits"],
             "accuracy": d["accuracy"], "energy": d["energy"],
             "pcag_raw": d["pcag_raw"], "pcag_eff": d["pcag_eff"]}
            for d in data],
        "power_wall": ({"from": wall["from"], "to": wall["to"],
                        "from_bits": wall["from_bits"], "to_bits": wall["to_bits"],
                        "slope_per_bit": wall["slope_per_bit"]} if wall else None),
        "continuous_inflection": cont,
    }
    out_json = os.path.join(os.path.dirname(__file__), "analysis_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n분석 요약 저장: {out_json}")

    print("\n=== 그림 생성 ===")
    make_fig1(data, OUT_DIR)
    make_fig2(data, OUT_DIR, wall, interp)


def detect_inflection_continuous(interp):
    """조건식 3.1: d²PCAG/db² = 0 (and d³≠0) 근사 검출.
    interp의 정의역(가장 작은~큰 bits)을 고밀도 샘플링해 이계도함수 부호변화(영교차) 탐색."""
    import numpy as np
    bmin, bmax = interp.x.min(), interp.x.max()
    b_grid = np.linspace(bmin, bmax, 600)
    try:
        f = lambda x: float(interp(x))              # noqa: E731
        h = 1e-4
        d2 = np.array([(f(b + h) - 2 * f(b) + f(b - h)) / (h * h) for b in b_grid])
        d3 = np.array([(f(b + 2 * h) - 2 * f(b + h) + 2 * f(b - h) - f(b - 2 * h)) / (2 * h * h * h) for b in b_grid])
        signs = np.sign(d2)
        crosses = []
        for i in range(len(signs) - 1):
            if signs[i] != signs[i + 1] and signs[i] != 0:
                if abs(float(d3[i])) > 1e-9:  # 조건식 3.1의 d³≠0
                    crosses.append(round(float(b_grid[i]), 2))
        return {"second_derivative_zero_candidates_bits": crosses,
                "bmin": float(bmin), "bmax": float(bmax)}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    main()