# experiments/jevons_model.py
# 매크로 전력망(Jevons Paradox) 시뮬레이션.
#
# 핵심 방정식 (INSTRUCTIONS.md Phase 4):
#   Total Grid Load = (Energy Consumption per Token) x (Elastic Token Demand Volume)
#
# Jevons Paradox 시나리오:
#   소프트웨어 양자화로 토큰당 에너지 비용이 감소(예: -50%)
#   -> 가격 탄력성 E_d > 1 에 따라 요청량이 지수적으로 급증(예: +300%)
#   -> 총 그리드 부하가 증가 (절대 에너지 감소가 아니라 역설적 증가)
#
# 수요 모델(상수 탄력성 수요곡선):
#   Q(P) = Q0 * (P/P0)^(-E_d)
#   여기서 P=토큰당 에너지 비용(에너지 단위), Q=토큰 수요량.
#   비용이 x% 줄면(P -> P*(1-s)), Q는 (1-s)^(-E_d) 배 증가.
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "docs", "figures"))


def jevons_demand(Q0, P0, E_d, savings_fraction):
    """수요 곡선 Q(P)=Q0*(P/P0)^(-E_d). savings_fraction: 비용 절감 비율(0~1).
    반환 (P_after, Q_after)."""
    P_after = P0 * (1 - savings_fraction)
    Q_after = Q0 * (P_after / P0) ** (-E_d)
    return P_after, Q_after


def simulate(config):
    """config: dict with keys:
        P0_tok (baseline energy per token), Q0 (baseline token demand),
        E_d (price elasticity), savings (list of savings fractions 0~1)
    반환: records list, summary dict."""
    P0 = config["P0_tok"]
    Q0 = config["Q0_tok"]
    E_d = config["E_d"]
    records = []
    for s in config["savings"]:
        P, Q = jevons_demand(Q0, P0, E_d, s)
        load = P * Q  # Total grid load (J)
        records.append({
            "savings": s,
            "cost_after": P,
            "demand_after": Q,
            "demand_growth_pct": (Q - Q0) / Q0 * 100,
            "total_load": load,
            "load_change_pct": (load - P0 * Q0) / (P0 * Q0) * 100,
        })
    # Jevons 역치: 비용 절감에도 총 부하가 증가하는 지점
    baseline_load = P0 * Q0
    summary = {
        "baseline_load": baseline_load,
        "E_d": E_d,
        "records": records,
    }
    return records, summary


def make_fig3(records, config, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    s = [r["savings"] * 100 for r in records]
    demand = [r["demand_growth_pct"] for r in records]
    load = [r["load_change_pct"] for r in records]
    ax.plot(s, load, "-o", color="#d62728", lw=2.5, ms=7, label="Total Grid Load change (%)")
    ax.plot(s, demand, "-s", color="#1f77b4", lw=2, ms=7, label="Token Demand growth (%)")
    ax.axhline(0, color="#999", ls="--", lw=1)
    ax.set_xlabel("Energy Cost Reduction per Token via Quantization (%)", fontsize=11)
    ax.set_ylabel("Change (%) relative to FP16 baseline", fontsize=11)
    ax.set_title(f"Fig 3. Jevons Paradox: Lower Cost → Higher Total Grid Load (E$_d$={config['E_d']})",
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_jevons_grid_load.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig3_jevons_grid_load.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig3 저장: fig3_jevons_grid_load.{png,pdf}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # 시나리오: FP16 기준 토큰당 에너지 비용 단위화, 절감율 0~70% 스윕
    # 실제 에너지 절감: FP16(140J/1k=0.14J/tok) -> INT4(61J/1k=0.061J/tok) = -56%
    config = {
        "P0_tok": 0.14,          # J/token (FP16, 참고데이터 기준)
        "Q0_tok": 1.0,           # 정규화 기준 수요
        "E_d": 1.5,              # 가격 탄력성 >1 (INSTRUCTIONS 요구)
        "savings": np.linspace(0.0, 0.70, 15).tolist(),
    }
    records, summary = simulate(config)

    print("=== Jevons Paradox 시뮬레이션 (가격 탄력성 E_d>1) ===")
    print(f"  Baseline cost P0={config['P0_tok']} J/tok, Q0={config['Q0_tok']}")
    print(f"  탄력성 E_d = {config['E_d']}")
    print(f"  {'절감%':>6} | {'수요증가%':>10} | {'총부하변화%':>10}")
    for r in records:
        print(f"  {r['savings']*100:>6.1f} | {r['demand_growth_pct']:>10.1f} | "
              f"{r['load_change_pct']:>10.1f}")

    # INT4(-56%) 시점의 역설 정량화
    int4_rec = min(records, key=lambda r: abs(r["savings"] - 0.56))
    print("\n=== INT4(에너지 -56%) 적용 시 ===")
    print(f"  수요 증가: +{int4_rec['demand_growth_pct']:.0f}%  "
          f"총 그리드 부하: {int4_rec['load_change_pct']:+.1f}%")
    # 절감에도 불구 부하가 증가하는 임계 절감률(부하 불변 지점)
    inv = next((r for r in records if r["load_change_pct"] >= 0), None)
    if inv:
        print(f"  Jevons 역치(부하가 증가하기 시작하는 절감률): {inv['savings']*100:.1f}%")

    out_json = os.path.join(os.path.dirname(__file__), "jevons_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"config": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                              for k, v in config.items()},
                   "summary": summary,
                   "int4_point": int4_rec}, f, indent=2, ensure_ascii=False)
    print(f"요약 저장: {out_json}")

    make_fig3(records, config, OUT_DIR)


if __name__ == "__main__":
    main()