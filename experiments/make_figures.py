# experiments/make_figures.py
# 전체 시각 자료 생성 (총 16종, PNG 200dpi + PDF). GPU 무관, 앵커/분석 JSON 기반.
#
# 입력:
#   results_raw.csv             (Llama-3-8B 앵커)
#   results_multimodel_raw.csv  (4개 모델 앵커)
#   analysis_summary.json       (이산 PCAG, Power Wall, PCHIP 변곡점)
#   sensitivity_summary.json    (θ 스윕, Jevons 그리드, Monte Carlo 샘플)
#   analysis_proof.json         (해석 모델 파라미터, 해석적 변곡점)
#   jevons_summary.json         (Jevons 시나리오)
#
# 데이터 원칙: 모든 수치는 Reference-Literature 앵커. 각 그림에 footnote 명시.
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "docs", "figures"))
CSV_MAIN = os.path.join(HERE, "results_raw.csv")
CSV_MULTI = os.path.join(HERE, "results_multimodel_raw.csv")

PREC_BITS = {"FP16": 16, "INT8": 8, "INT4": 4, "INT3": 3, "INT2": 2}
PCOLOR = {"FP16": "#1f77b4", "INT8": "#2ca02c", "INT4": "#d62728",
          "INT3": "#ff7f0e", "INT2": "#8c564b"}
MCOLOR = {"meta-llama/Llama-3-8B": "#1f77b4", "Qwen/Qwen2.5-7B": "#2ca02c",
          "google/gemma-2-9b": "#9467bd", "mistralai/Mistral-7B": "#ff7f0e"}
MSHORT = {"meta-llama/Llama-3-8B": "Llama-3-8B", "Qwen/Qwen2.5-7B": "Qwen-2.5-7B",
          "google/gemma-2-9b": "Gemma-2-9B", "mistralai/Mistral-7B": "Mistral-7B"}

plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
    "savefig.bbox": "tight", "axes.axisbelow": True,
})

FOOT = "Data: literature-anchored reference (Source=Reference-Literature, GPU measurement pending)"


def detect_measured():
    """results_raw.csv / results_multimodel_raw.csv 에 Measured-GPU 행이 있으면 True."""
    for path in (CSV_MAIN, CSV_MULTI):
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("Source") == "Measured-GPU":
                    return True
    return False


def refresh_footnote():
    """실측(Source=Measured-GPU) 데이터 존재 시 모든 그림 하단 출처 문구 갱신."""
    global FOOT
    if detect_measured():
        FOOT = "Data: measured on GPU (Source=Measured-GPU, Google Colab T4)"
    else:
        FOOT = "Data: literature-anchored reference (Source=Reference-Literature, GPU measurement pending)"
    return FOOT


def load_main():
    rows = []
    skipped = 0
    with open(CSV_MAIN, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Precision"]:
                try:
                    rows.append({
                        "precision": r["Precision"], "bits": PREC_BITS[r["Precision"]],
                        "acc": float(r["Accuracy_Score"]),
                        "energy": float(r["Total_Energy_J"]),
                        "power": float(r["Avg_Power_W"]),
                        "latency": float(r["Latency_ms"]),
                        "tps": float(r["Throughput_tps"]),
                        "vram": float(r["VRAM_GB"]),
                    })
                except (TypeError, ValueError):
                    # 정확도/에너지 누락 행 (예: --no_eval 실측) → 그림에서 제외
                    skipped += 1
    if skipped:
        print(f"[make_figures.load_main] 데이터 누락 행 {skipped}개 스킵")
    rows.sort(key=lambda x: -x["bits"])
    P0, A0 = rows[0]["energy"], rows[0]["acc"]
    for r in rows:
        r["S"] = (P0 - r["energy"]) / P0
        r["L"] = (A0 - r["acc"]) / A0
        r["x"] = 16 - r["bits"]
        r["pcag"] = (r["S"] / r["L"]) if r["L"] > 0 else None
    return rows, P0, A0


def load_multi():
    models = {}
    skipped = 0
    with open(CSV_MULTI, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Precision"]:
                try:
                    models.setdefault(r["Model_Name"], []).append({
                        "precision": r["Precision"], "bits": PREC_BITS[r["Precision"]],
                        "acc": float(r["Accuracy_Score"]),
                        "energy": float(r["Total_Energy_J"]),
                    })
                except (TypeError, ValueError):
                    skipped += 1
    if skipped:
        print(f"[make_figures.load_multi] 데이터 누락 행 {skipped}개 스킵")
    out = {}
    for m, pts in models.items():
        pts.sort(key=lambda x: -x["bits"])
        P0, A0 = pts[0]["energy"], pts[0]["acc"]
        for p in pts:
            p["S"] = (P0 - p["energy"]) / P0
            p["L"] = (P0 - p["acc"]) / P0  # placeholder, fixed below
            p["L"] = (pts[0]["acc"] - p["acc"]) / pts[0]["acc"]
            p["pcag"] = (p["S"] / p["L"]) if p["L"] > 0 else None
            p["retention"] = p["acc"] / pts[0]["acc"] * 100
        out[m] = pts
    return out


def load_json(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"))
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  [ok] {name}.png/.pdf")


def footnote(fig, text=None):
    fig.text(0.99, 0.005, text or FOOT, ha="right", va="bottom", fontsize=7.5,
             color="#888", style="italic")


# --------------------------------------------------------------- 모델 곡선들
def s_curve(x, p):
    return p["S_max"] * (1 - np.exp(-(p["lambda"] * x) ** p["beta"]))


def l_curve(x, p):
    return (p["c_lin"] * x
            + p["Lr_max"] / (1 + np.exp(-p["k"] * (x - p["xc"]))))


def pcag_curve(x, p):
    return s_curve(x, p) / l_curve(x, p)


# ============================================================== Fig 1
def fig1(rows, P0, A0):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    # iso-PCAG 라인 (직선): A(E) = A0 (1 - (P0-E)/(P0*c))
    E = np.linspace(40, 150, 100)
    for c, lab in [(40, "PCAG=40"), (20, "20"), (10, "10"), (5, "5"), (2.2, "2.2")]:
        A = A0 * (1 - (P0 - E) / (P0 * c))
        ax.plot(E, A, ls="--", lw=0.9, color="#bbb", zorder=1)
        # 라벨은 왼쪽 끝(E=45)에 정렬해 FP16 포인트와 겹침 방지
        y_lab = A0 * (1 - (P0 - 45) / (P0 * c))
        if 43 < y_lab < 70:
            ax.annotate(lab, xy=(45, y_lab), fontsize=8, color="#999",
                        va="bottom", ha="left")
    # Power Wall zone (b 3~4: 에너지 61→54 J)
    ax.axvspan(54, 61, color="#d62728", alpha=0.07, zorder=0)
    ax.annotate("Power Wall\nzone", xy=(57.5, 49.5), ha="center", fontsize=9,
                color="#d62728", fontweight="bold")
    xs = [r["energy"] for r in rows]
    ys = [r["acc"] for r in rows]
    ax.plot(xs, ys, "-", color="#555", lw=1.5, zorder=2)
    for r in rows:
        ax.scatter(r["energy"], r["acc"], color=PCOLOR[r["precision"]], s=130,
                   zorder=3, edgecolor="white", linewidth=1.2)
        ax.annotate(f"{r['precision']}\n(E {r['S']*100:+.0f}%, acc {r['L']*100:+.1f}%)",
                    (r["energy"], r["acc"]), textcoords="offset points",
                    xytext=(10, -14), fontsize=8.5,
                    color=PCOLOR[r["precision"]], fontweight="bold")
    ax.annotate("", xy=(58, 47.6), xytext=(135, 66.2),
                arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.6,
                                connectionstyle="arc3,rad=0.15"))
    ax.text(103, 56.5, "quantization\ndirection", fontsize=9, color="#666",
            ha="center")
    ax.set_xlabel("Energy (J / 1,000 tokens)")
    ax.set_ylabel("Inference accuracy (%)")
    ax.set_title("Fig 1. Accuracy–Energy Frontier across Quantization (Llama-3-8B)")
    ax.set_xlim(42, 152)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "fig1_accuracy_vs_energy")


# ============================================================== Fig 2
def fig2(rows, proof):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    pts = [(r["bits"], r["pcag"]) for r in rows if r["pcag"]]
    pts = sorted(pts)  # PCHIP: strictly increasing x 필요
    bs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    from scipy.interpolate import PchipInterpolator
    interp = PchipInterpolator(np.array(bs, float), np.array(ys, float))
    grid = np.linspace(2, 8, 300)
    ax.fill_between([3, 4], 0, 24, color="#d62728", alpha=0.08, zorder=0)
    ax.plot(grid, interp(grid), color="#d62728", lw=1.4, ls=":", zorder=1)
    ax.plot(bs, ys, "-o", color="#d62728", lw=2.5, ms=9, zorder=2,
            markeredgecolor="white")
    for b, y, r in zip(bs, ys, [r for r in rows if r["pcag"]]):
        ax.annotate(r["precision"], (b, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=10, fontweight="bold")
    # cliff 화살표/주석
    ax.annotate("", xy=(3.15, 4.9), xytext=(3.9, 10.2),
                arrowprops=dict(arrowstyle="-|>", color="#111", lw=1.8))
    ax.text(3.52, 8.2, "−54.4%\n(PCAG cliff)", fontsize=9.5, ha="center",
            fontweight="bold")
    ax.axvline(4.19, color="#111", ls="--", lw=1.5)
    ax.text(4.19, 22.3, "analytic b*=4.19", rotation=90, va="top", ha="right",
            fontsize=8.5, color="#111")
    ax.text(3.5, 1.2, "POWER WALL ZONE\n(b ∈ [3, 4])", ha="center", fontsize=9.5,
            color="#d62728", fontweight="bold")
    ax.set_xlabel("Weight bit precision, b (bits)")
    ax.set_ylabel("PCAG  =  rel. power saving / rel. accuracy loss")
    ax.set_title("Fig 2. PCAG Collapse — the Power Wall (θ=3, slope 5.68/bit at INT4→INT3)")
    ax.set_ylim(0, 24)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "fig2_pcag_power_wall")


# ============================================================== Fig 3
def fig3(jev, rows):
    cfg = jev["config"]
    Ed = cfg["E_d"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    s = np.array([r["savings"] for r in jev["summary"]["records"]]) * 100
    dem = [r["demand_growth_pct"] for r in jev["summary"]["records"]]
    load = [r["load_change_pct"] for r in jev["summary"]["records"]]
    s_fine = np.linspace(0, 70, 300)
    ax.plot(s_fine, ((1 - s_fine / 100) ** (1 - Ed) - 1) * 100, color="#d62728",
            lw=1.2, ls=":", zorder=1, label=f"closed form $(1-s)^{{1-E_d}}$, $E_d$={Ed}")
    ax.plot(s, load, "-o", color="#d62728", lw=2.5, ms=7, zorder=2,
            label="Total grid load change (%)")
    ax.plot(s, dem, "-s", color="#1f77b4", lw=2, ms=6, zorder=2,
            label="Token demand growth (%)")
    ax.axhline(0, color="#999", ls="--", lw=1)
    # INT4 시점 마커
    s4 = next(r for r in rows if r["precision"] == "INT4")["S"] * 100
    load4 = ((1 - s4 / 100) ** (1 - Ed) - 1) * 100
    ax.scatter([s4], [load4], s=140, marker="*", color="#111", zorder=4)
    ax.annotate(f"INT4 quantization\n(s={s4:.0f}% → load {load4:+.0f}%)",
                (s4, load4), textcoords="offset points", xytext=(12, -28),
                fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#111", lw=1))
    ax.text(36.5, -8, "any s > 0 with $E_d$>1 ⇒ load increases\n(Jevons threshold: $E_d$=1)",
            fontsize=9, color="#555")
    ax.set_xlabel("Energy cost reduction per token via quantization (%)")
    ax.set_ylabel("Change vs FP16 baseline (%)")
    ax.set_title(f"Fig 3. Jevons Paradox on the Macro Grid ($E_d$={Ed}: elastic demand)")
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "fig3_jevons_grid_load")


# ============================================================== Fig 4
def fig4(rows):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))
    prec = [r["precision"] for r in rows]
    panels = [
        ("VRAM (GB)", [r["vram"] for r in rows], "#1f77b4", "−73% @INT2"),
        ("Average power draw (W)", [r["power"] for r in rows], "#2ca02c", "−43% @INT2"),
        ("Latency (ms/token)", [r["latency"] for r in rows], "#ff7f0e", "−38% @INT2"),
        ("Throughput (tokens/s)", [r["tps"] for r in rows], "#9467bd", "+64% @INT2"),
    ]
    for ax, (title, vals, col, note) in zip(axes.flat, panels):
        bars = ax.bar(prec, vals, color=[PCOLOR[p] for p in prec], alpha=0.88,
                      edgecolor="white")
        for b, v, r in zip(bars, vals, rows):
            d = (v / vals[0] - 1) * 100
            ax.annotate(f"{v:.1f}" + (f"\n({d:+.0f}%)" if abs(d) > 0.5 else ""),
                        (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=9)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.28)
        ax.grid(axis="x", alpha=0)
    fig.suptitle("Fig 4. Resource Footprint across Quantization (Llama-3-8B)",
                 fontsize=13, fontweight="bold")
    fig.text(0.99, 0.005, FOOT, ha="right", fontsize=7.5, color="#888")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    save(fig, "fig04_resource_footprint")


# ============================================================== Fig 5
def fig5(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8))
    bs = [r["bits"] for r in rows]
    acc = [r["acc"] for r in rows]
    ax1.plot(bs, acc, "-o", color="#1f77b4", lw=2.5, ms=9, markeredgecolor="white")
    for b, a, r in zip(bs, acc, rows):
        ax1.annotate(f"{r['precision']}\n{a:.1f}%", (b, a),
                     textcoords="offset points", xytext=(0, 10), ha="center",
                     fontsize=9, fontweight="bold", color=PCOLOR[r["precision"]])
    ax1.axvspan(3, 4, color="#d62728", alpha=0.08)
    ax1.text(3.5, 48, "cliff zone", ha="center", fontsize=9, color="#d62728")
    ax1.set_xlabel("Weight bits, b")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("(a) Accuracy vs bit precision")
    ax1.invert_xaxis()
    steps = [(f"{rows[i]['precision']}→{rows[i+1]['precision']}",
              rows[i + 1]["acc"] - rows[i]["acc"]) for i in range(len(rows) - 1)]
    names = [s[0] for s in steps]
    vals = [s[1] for s in steps]
    cols = ["#d62728" if v <= -5 else "#2ca02c" for v in vals]
    bars = ax2.bar(names, vals, color=cols, alpha=0.88, edgecolor="white")
    for b, v in zip(bars, vals):
        ax2.annotate(f"{v:+.1f}p", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="top" if v < 0 else "bottom",
                     fontsize=9.5, fontweight="bold")
    ax2.axhline(0, color="#333", lw=0.8)
    ax2.set_ylabel("Δ accuracy per transition (p)")
    ax2.set_title("(b) Accuracy loss per quantization step")
    ax2.tick_params(axis="x", labelrotation=12)
    fig.suptitle("Fig 5. The Accuracy Cliff: loss accelerates below 4 bits",
                 fontsize=13, fontweight="bold")
    fig.text(0.99, 0.005, FOOT, ha="right", fontsize=7.5, color="#888")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    save(fig, "fig05_accuracy_cliff")


# ============================================================== Fig 6
def fig6(rows, proof):
    p = proof["model"]["fitted_params"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = np.linspace(0, 14.5, 300)
    ax.fill_between(x, s_curve(x, p), l_curve(x, p), color="#d62728", alpha=0.08)
    ax.plot(x, s_curve(x, p), color="#2ca02c", lw=2.2,
            label=f"rel. power saving S(x) — saturating (Weibull fit)")
    ax.plot(x, l_curve(x, p), color="#d62728", lw=2.2,
            label=f"rel. accuracy loss Lr(x) — accelerating (logistic fit)")
    ax.scatter([r["x"] for r in rows if r["precision"] != "FP16"],
               [r["S"] for r in rows if r["precision"] != "FP16"],
               color="#2ca02c", s=55, zorder=3, edgecolor="white")
    ax.scatter([r["x"] for r in rows if r["precision"] != "FP16"],
               [r["L"] for r in rows if r["precision"] != "FP16"],
               color="#d62728", s=55, marker="s", zorder=3, edgecolor="white")
    for r in rows[1:]:
        ax.annotate(r["precision"], (r["x"], r["S"]),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=8.5, color="#2ca02c", fontweight="bold")
        ax.annotate(r["precision"], (r["x"], r["L"]),
                    textcoords="offset points", xytext=(0, -14), ha="center",
                    fontsize=8.5, color="#d62728", fontweight="bold")
    xw = 16 - 4.19
    ax.axvline(xw, color="#111", ls="--", lw=1.4)
    ax.annotate("Power Wall:\nloss acceleration\noutpaces saving saturation",
                xy=(xw, 0.32), xytext=(6.2, 0.46), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="#111", lw=1.2))
    ax.text(7.2, 0.6, "divergence wedge =\nefficiency collapse region",
            fontsize=9, color="#d62728", ha="center")
    ax.set_xlabel("Quantization depth, x = 16 − b")
    ax.set_ylabel("Relative change vs FP16 (fraction)")
    ax.set_title("Fig 6. The Divergence Mechanism: saturating savings vs accelerating loss")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 0.72)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "fig06_divergence_mechanism")


# ============================================================== Fig 7
def fig7(multi):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    precs = ["INT8", "INT4", "INT3", "INT2"]
    models = list(multi.keys())
    w = 0.19
    x = np.arange(len(precs)) if False else np.arange(len(precs))
    for i, m in enumerate(models):
        vals = []
        for prec in precs:
            v = next((p["pcag"] for p in multi[m] if p["precision"] == prec), np.nan)
            vals.append(v)
        ax.bar(x + (i - 1.5) * w, vals, w, label=MSHORT[m], color=MCOLOR[m],
               alpha=0.9, edgecolor="white")
        for xi, v in zip(x + (i - 1.5) * w, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.1f}", (xi, v), ha="center", va="bottom",
                            fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(precs)
    ax.set_ylabel("PCAG (rel. power saving / rel. accuracy loss)")
    ax.set_title("Fig 7. PCAG across Models — universal collapse pattern")
    ax.legend(ncol=4, fontsize=9.5, loc="upper right")
    ax.set_ylim(0, 32)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "fig07_multimodel_pcag")


# ============================================================== Fig 8
def fig8(multi):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for m, pts in multi.items():
        bs = [p["bits"] for p in pts]
        ret = [p["retention"] for p in pts]
        ax.plot(bs, ret, "-o", color=MCOLOR[m], lw=2.2, ms=7, label=MSHORT[m],
                markeredgecolor="white")
        ax.annotate(MSHORT[m].split("-")[0], (bs[-1], ret[-1]),
                    textcoords="offset points", xytext=(6, -3), fontsize=8.5,
                    color=MCOLOR[m], fontweight="bold")
    ax.axvspan(3, 4, color="#d62728", alpha=0.08)
    ax.text(3.5, ax.get_ylim()[0] + 3, "Power Wall\nzone", ha="center",
            fontsize=9, color="#d62728", fontweight="bold")
    ax.set_xlabel("Weight bits, b")
    ax.set_ylabel("Accuracy retention vs FP16 (%)")
    ax.set_title("Fig 8. Accuracy Retention across Models")
    ax.invert_xaxis()
    ax.legend(fontsize=9.5)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "fig08_multimodel_retention")


# ============================================================== Fig 9
def fig9(sens):
    g = sens["jevons_grid"]
    mat = np.array(g["load_change_pct_matrix"])
    e_d = np.array(g["e_d_grid"])
    s = np.array(g["savings_grid"]) * 100
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    im = ax.pcolormesh(s, e_d, mat, cmap="RdYlGn_r", shading="auto",
                       vmin=-25, vmax=90)
    cs = ax.contour(s, e_d, mat, levels=8, colors="k", linewidths=0.6, alpha=0.5)
    ax.clabel(cs, fmt="%.0f%%", fontsize=8)
    ax.contour(s, e_d, mat, levels=[0], colors="#111", linewidths=2.2)
    ax.annotate("$E_d$ = 1 boundary\n(Jevons threshold)", xy=(38, 1.0),
                xytext=(24, 1.22), fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#111"))
    ax.scatter([56.4], [1.5], s=130, marker="*", color="#111", zorder=5)
    ax.annotate("INT4 (Llama-3-8B)\n+49%", (56.4, 1.5), xytext=(44, 1.95),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=1))
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("Total grid load change (%)")
    ax.set_xlabel("Energy cost reduction per token, s (%)")
    ax.set_ylabel("Price elasticity of demand, $E_d$")
    ax.set_title("Fig 9. Jevons Sensitivity Map: grid load change (%)")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "fig09_jevons_heatmap")


# ============================================================== Fig 10
def fig10(sens, rows):
    g = sens["jevons_grid"]
    mat = np.array(g["load_change_pct_matrix"])
    e_d = np.array(g["e_d_grid"])
    s = np.array(g["savings_grid"]) * 100
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    cf = ax.contourf(s, e_d, mat, levels=np.linspace(-30, 90, 13), cmap="RdYlGn_r",
                     alpha=0.85)
    ax.contour(s, e_d, mat, levels=[0], colors="#111", linewidths=2.4)
    ax.text(56, 1.03, "$E_d=1$ : Jevons boundary", fontsize=10, fontweight="bold")
    scen = scen_points(rows)  # [(prec, savings%)]
    for prec, sv in scen:
        for ed, mk in [(1.0, "o"), (1.5, "^"), (2.0, "s")]:
            ax.scatter([sv], [ed], marker=mk, s=70, color=PCOLOR[prec],
                       edgecolor="white", zorder=5,
                       label=f"{prec} (E_d 시나리오)" if ed == 1.5 else None)
    ax.annotate("INT8", xy=(scen[0][1], 1.5), xytext=(scen[0][1] + 2, 1.72),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("INT2", xy=(scen[-1][1], 1.5), xytext=(scen[-1][1] - 1, 1.28),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.9))
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label("Total grid load change (%)")
    ax.set_xlabel("Energy cost reduction, s (%)")
    ax.set_ylabel("Price elasticity, $E_d$")
    ax.set_title("Fig 10. Jevons Phase Diagram — quantization scenarios")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "fig10_jevons_phase")


def scen_points(rows):
    order = ["INT8", "INT4", "INT3", "INT2"]
    return [(p, next(r["S"] * 100 for r in rows if r["precision"] == p))
            for p in order]


# ============================================================== Fig 11
def fig11(sens, proof):
    samples = sens["_mc_inflection_samples"]
    mc = sens["monte_carlo"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    ax1.hist(samples, bins=36, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax1.axvline(3.51, color="#d62728", lw=2, ls="--",
                label="empirical PCHIP: 3.51")
    ax1.axvline(proof["inflection_condition_3_1"]["b_star_primary"], color="#111",
                lw=2, ls="-.", label="analytic model: 4.19")
    ax1.axvline(mc["inflection_bits"]["mean"], color="#2ca02c", lw=2,
                label=f"MC mean: {mc['inflection_bits']['mean']}")
    ax1.set_xlabel("Inflection point b* (bits) — condition [3.1]")
    ax1.set_ylabel("Monte Carlo count")
    ax1.set_title("(a) Inflection point under anchor noise (σ=3%, N=3000)")
    ax1.legend(fontsize=9)
    trans = list(mc["wall_transition_prob"].keys())
    probs = [mc["wall_transition_prob"][t] for t in trans]
    cols = ["#d62728" if t == "INT4->INT3" else "#999" for t in trans]
    bars = ax2.barh(trans, probs, color=cols, alpha=0.88, edgecolor="white")
    for b, v in zip(bars, probs):
        ax2.annotate(f"{v*100:.1f}%", (v, b.get_y() + b.get_height() / 2),
                     va="center", fontsize=9.5, fontweight="bold")
    ax2.set_xlabel("P(transition flagged as Power Wall)")
    ax2.set_title("(b) Wall location probability")
    ax2.set_xlim(0, 1.0)
    fig.suptitle("Fig 11. Robustness of the Power Wall conclusion (Monte Carlo)",
                 fontsize=13, fontweight="bold")
    fig.text(0.99, 0.005, "Mode: INT4→INT3 (66.8% of flagged runs)", ha="right",
             fontsize=8, color="#888")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    save(fig, "fig11_mc_robustness")


# ============================================================== Fig 12
def fig12(sens):
    multi = sens["theta_sweep_multimodel"]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    trans = ["FP16->INT8", "INT8->INT4", "INT4->INT3", "INT3->INT2"]
    ypos = np.arange(len(trans))
    models = list(multi.keys())
    w = 0.2
    for i, m in enumerate(models):
        slopes = {t["from"] + "->" + t["to"]: t["slope_per_bit"]
                  for t in multi[m]["slopes_per_bit"]}
        vals = [slopes.get(t, 0) for t in trans]
        ax.barh(ypos + (1.5 - i) * w, vals, w, label=MSHORT[m], color=MCOLOR[m],
                alpha=0.9, edgecolor="white")
        for yi, v in zip(ypos + (1.5 - i) * w, vals):
            if v > 0:
                ax.annotate(f"{v:.2f}", (v, yi), va="center", fontsize=7.5)
    ax.axvline(3.0, color="#111", ls="--", lw=1.8)
    ax.text(3.05, len(trans) - 0.42, "θ = 3 (base case)", fontsize=9.5,
            fontweight="bold")
    ax.set_yticks(ypos)
    ax.set_yticklabels([t.replace("->", " → ") for t in trans])
    ax.invert_yaxis()
    ax.set_xlabel("|ΔPCAG / Δb|  (per-bit slope, condition [3.2])")
    ax.set_title("Fig 12. Per-bit PCAG slopes across models — INT4→INT3 dominates")
    ax.legend(fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "fig12_theta_sensitivity")


# ============================================================== Fig 13
def fig13(rows, proof):
    p = proof["model"]["fitted_params"]
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 9.6), sharex=True)
    b = np.linspace(2, 8, 400)
    x = 16 - b
    pc = pcag_curve(x, p)
    axes[0].plot(b, pc, color="#d62728", lw=2.2, label="analytic model PCAG(b)")
    axes[0].plot([r["bits"] for r in rows if r["pcag"]],
                 [r["pcag"] for r in rows if r["pcag"]], "o", ms=9,
                 color="#333", label="anchor discrete PCAG", zorder=3,
                 markeredgecolor="white")
    for r in rows[1:]:
        axes[0].annotate(r["precision"], (r["bits"], r["pcag"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=9)
    axes[0].set_ylabel("PCAG")
    axes[0].legend(fontsize=9.5)
    axes[0].set_title("Fig 13. Analytic PCAG model: fit, elasticity, inflection")
    axes[0].invert_xaxis()

    # g(b) = d ln PCAG / db = -g(x)  (x=16-b, |db/dx|=1)
    from analytical_proof import make_g
    g = make_g(p)
    gx = -np.array([float(np.real(g(xx))) for xx in x])
    axes[1].plot(b, gx, color="#2ca02c", lw=2)
    axes[1].axhline(0, color="#333", lw=0.9)
    axes[1].fill_between(b, gx, 0, where=gx > 0, color="#2ca02c", alpha=0.15)
    axes[1].fill_between(b, gx, 0, where=gx < 0, color="#d62728", alpha=0.15)
    axes[1].text(3.0, float(np.max(gx)) * 0.6,
                 "g > 0: PCAG rises with b\n(efficiency gain regime)",
                 fontsize=9, color="#2ca02c", ha="center")
    axes[1].text(7.55, -0.045, "g < 0:\nmodel peak", fontsize=8.5,
                 color="#d62728", ha="center")
    axes[1].set_ylabel("g(b) = d ln PCAG / db")

    bstar = proof["inflection_condition_3_1"]["b_star_primary"]
    h_vals = []
    for bb in b:
        xx = 16 - bb
        # PCAG''(b) = d²PCAG/dx² ∝ h(x)  (|db/dx|=1, 이중 연쇄 부호 +)
        h = float(np.imag(g(xx + 1e-20j)) / 1e-20 + g(xx) ** 2)
        h_vals.append(h)
    axes[2].plot(b, h_vals, color="#1f77b4", lw=2)
    axes[2].axhline(0, color="#333", lw=0.9)
    axes[2].axvline(bstar, color="#111", ls="--", lw=1.6)
    axes[2].annotate(f"condition [3.1] root\nb* = {bstar:.2f}\n(d³ ≠ 0)",
                     xy=(bstar, 0), xytext=(bstar + 0.9, max(h_vals) * 0.5),
                     fontsize=10, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", lw=1.2))
    axes[2].fill_between(b, h_vals, 0, where=np.array(h_vals) > 0,
                         color="#1f77b4", alpha=0.12)
    axes[2].set_xlabel("Weight bits, b")
    axes[2].set_ylabel("PCAG''(b)  ∝  h")
    axes[2].invert_xaxis()
    fig.tight_layout(rect=[0, 0.01, 1, 1])
    save(fig, "fig13_continuous_model")


# ============================================================== Fig 14
def fig14(rows, P0, A0):
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    P0, A0 = rows[0]["energy"], rows[0]["acc"]
    E = np.linspace(45, 145, 200)
    for c, alpha in [(40, 0.5), (20, 0.5), (10, 0.5), (5, 0.5), (2.2, 0.5)]:
        A = A0 * (1 - (P0 - E) / (P0 * c))
        ax.plot(E, A, ls="--", lw=0.9, color="#aaa", zorder=1)
        if E[-1] > 40:
            ax.annotate(f"PCAG={c}", (E[np.argmin(np.abs(E - 132))],
                        A0 * (1 - (P0 - 132) / (P0 * c))),
                        fontsize=8, color="#888", rotation=-18)
    ax.plot([r["energy"] for r in rows], [r["acc"] for r in rows], "-o",
            color="#1f77b4", lw=2.5, ms=9, zorder=3, markeredgecolor="white",
            label="quantization path (Llama-3-8B)")
    for r in rows:
        ax.annotate(r["precision"], (r["energy"], r["acc"]),
                    textcoords="offset points", xytext=(7, -3), fontsize=9.5,
                    fontweight="bold", color=PCOLOR[r["precision"]])
    ax.axvspan(54, 61, color="#d62728", alpha=0.07)
    ax.annotate("Power Wall:\nmax efficiency loss\nper additional bit removed",
                xy=(57.5, 60.5), xytext=(78, 56.5), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", lw=1.1, color="#d62728"),
                color="#d62728")
    ax.set_xlabel("Energy (J / 1,000 tokens)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Fig 14. Efficiency Frontier with iso-PCAG contours")
    ax.legend(fontsize=9.5, loc="lower right")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "fig14_efficiency_frontier")


# ============================================================== Fig 15
def fig15(rows, multi, jev):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    # (a) PCAG bars
    ax = axes[0, 0]
    pts = [r for r in rows if r["pcag"]]
    bars = ax.bar([r["precision"] for r in pts], [r["pcag"] for r in pts],
                  color=[PCOLOR[r["precision"]] for r in pts], alpha=0.9,
                  edgecolor="white")
    for b, r in zip(bars, pts):
        ax.annotate(f"{r['pcag']:.1f}", (b.get_x() + b.get_width() / 2, r["pcag"]),
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_title("(a) PCAG by precision (Llama-3-8B)", fontsize=11)
    ax.set_ylabel("PCAG")
    ax.set_ylim(0, 25)
    # (b) Jevons
    ax = axes[0, 1]
    s = [r["savings"] * 100 for r in jev["summary"]["records"]]
    load = [r["load_change_pct"] for r in jev["summary"]["records"]]
    ax.plot(s, load, "-o", color="#d62728", lw=2.2, ms=5)
    ax.axhline(0, color="#999", ls="--", lw=1)
    ax.fill_between(s, load, 0, where=np.array(load) > 0, color="#d62728",
                    alpha=0.12)
    ax.set_title("(b) Jevons grid load vs savings ($E_d$=1.5)", fontsize=11)
    ax.set_xlabel("Energy saving s (%)")
    ax.set_ylabel("Grid load change (%)")
    # (c) retention at INT4
    ax = axes[1, 0]
    models = list(multi.keys())
    ret4 = [next(p["retention"] for p in multi[m] if p["precision"] == "INT4")
            for m in models]
    bars = ax.bar([MSHORT[m].replace("-8B", "").replace("-7B", "").replace("-9B", "")
                   for m in models], ret4,
                  color=[MCOLOR[m] for m in models], alpha=0.9, edgecolor="white")
    for b, v in zip(bars, ret4):
        ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_title("(c) Accuracy retention at INT4", fontsize=11)
    ax.set_ylim(85, 100)
    # (d) wall slopes
    ax = axes[1, 1]
    trans = ["INT8→INT4", "INT4→INT3", "INT3→INT2"]
    slopes = [2.580, 5.682, 2.561]
    cols = ["#2ca02c", "#d62728", "#2ca02c"]
    bars = ax.bar(trans, slopes, color=cols, alpha=0.9, edgecolor="white")
    for b, v in zip(bars, slopes):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", fontsize=9.5, fontweight="bold")
    ax.axhline(3.0, color="#111", ls="--", lw=1.6)
    ax.text(2.45, 3.12, "θ=3", fontsize=9.5, fontweight="bold")
    ax.set_title("(d) Per-bit PCAG slope vs θ (condition [3.2])", fontsize=11)
    ax.set_ylabel("|ΔPCAG/Δb|")
    fig.suptitle("Fig 15. PCAG Research Overview — Power Wall & Jevons Paradox",
                 fontsize=13.5, fontweight="bold")
    fig.text(0.99, 0.005, FOOT, ha="right", fontsize=7.5, color="#888")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save(fig, "fig15_dashboard")


# ============================================================== Fig 16
def fig16(sens):
    g = sens["jevons_grid"]
    mat = np.array(g["load_change_pct_matrix"])
    e_d = np.array(g["e_d_grid"])
    s = np.array(g["savings_grid"]) * 100
    S, ED = np.meshgrid(s, e_d)
    fig = plt.figure(figsize=(9.0, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(S, ED, mat, cmap="RdYlGn_r", alpha=0.92,
                           rstride=1, cstride=1, linewidth=0.2,
                           edgecolor="#666")
    ax.contour(S, ED, mat, levels=8, zdir="z", offset=mat.min() - 12,
               cmap="RdYlGn_r", linewidths=0.8)
    ax.scatter([56.4], [1.5], [((1 - 0.564) ** (1 - 1.5) - 1) * 100],
               color="#111", s=90, marker="*", zorder=5)
    ax.text2D(0.72, 0.18, "★ INT4 (E_d=1.5): +49%", fontsize=10,
              transform=ax.transAxes, fontweight="bold")
    ax.set_xlabel("Energy saving s (%)")
    ax.set_ylabel("Elasticity E_d")
    ax.set_zlabel("Grid load change (%)")
    ax.set_title("Fig 16. Jevons Surface: load change = (1−s)^(1−E_d) − 1")
    ax.view_init(elev=24, azim=-58)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label="Load change (%)")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "fig16_jevons_surface")


# ============================================================== Fig 17 (legacy upg: wall zone summary)
def fig17(rows, sens, proof):
    """Fig 17: Power Wall 판정 요약 타임라인 (3 독립 경로 비교)."""
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    mc = sens["monte_carlo"]["inflection_bits"]
    paths = [
        ("Empirical PCHIP\n(analysis.py)", 3.51, 0, "#1f77b4"),
        ("Monte Carlo σ=3%\n(N=3000)", mc["mean"], mc["std"], "#2ca02c"),
        ("Analytic model\n(Weibull+logistic)", proof["inflection_condition_3_1"]["b_star_primary"],
         0, "#d62728"),
    ]
    for i, (name, b, sd, col) in enumerate(paths):
        y = len(paths) - 1 - i
        if sd:
            ax.errorbar([b], [y], xerr=[[sd], [sd]], fmt="o", color=col,
                        ms=11, capsize=5, lw=2)
        else:
            ax.scatter([b], [y], s=130, color=col, zorder=3, edgecolor="white")
        ha = "right" if b > 4.0 else "left"
        dx = -10 if b > 4.0 else 10
        ax.annotate(f"{name}: b*={b:.2f}" + (f" ± {sd:.2f}" if sd else ""),
                    (b, y), xytext=(dx, 0), textcoords="offset points",
                    ha=ha, va="center", fontsize=9.5)
    ax.axvspan(3, 4, color="#d62728", alpha=0.08)
    ax.text(3.5, 2.45, "INT4→INT3 transition", ha="center", fontsize=9.5,
            color="#d62728", fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(-0.6, 2.7)
    ax.set_xlim(2.4, 4.8)
    ax.set_xlabel("Power Wall location b* (bits)")
    ax.set_title("Fig 17. Three Independent Estimation Paths Converge near INT4")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "fig17_wall_convergence")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    refresh_footnote()
    print(f"=== 그림 생성 시작 (16+1종) | source={FOOT} ===")
    rows, P0, A0 = load_main()
    multi = load_multi()
    sens = load_json("sensitivity_summary.json")
    proof = load_json("analysis_proof.json")
    jev = load_json("jevons_summary.json")
    fig1(rows, P0, A0)
    fig2(rows, proof)
    fig3(jev, rows)
    fig4(rows)
    fig5(rows)
    fig6(rows, proof)
    fig7(multi)
    fig8(multi)
    fig9(sens)
    fig10(sens, rows)
    fig11(sens, proof)
    fig12(sens)
    fig13(rows, proof)
    fig14(rows, P0, A0)
    fig15(rows, multi, jev)
    fig16(sens)
    fig17(rows, sens, proof)
    print("=== 완료: docs/figures/ 확인 ===")


if __name__ == "__main__":
    main()
