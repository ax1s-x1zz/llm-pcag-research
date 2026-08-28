# experiments/multimodel_data.py
# 다중 모델 문헌 앵커 참고 데이터 생성 스크립트.
#
# ⚠️ 데이터 원칙(research_journal.md 2026-08-27 준수):
#   GPU 미실측 환경이므로 모든 수치는 **문헌 전형 경향 앵커**이며,
#   Source='Reference-Literature' 로 명시한다. 절대 실측으로 위장하지 않는다.
#   GPU 확보 시 benchmark_driver.py 가 동일 스키마로 Source='Measured-GPU' 덮어쓰기.
#
# 물리적 정합성:
#   E[J/1k tok] ≈ BATCH_AMORT × P[W] × latency[ms/token] / 1000
#   BATCH_AMORT ≈ 0.0119  (Llama-3-8B 앵커에서 역산된 유효 배치 스케일, ~80 배치)
#   -> 모델 간에도 전력×지연×배치 스케일로 에너지가 일관되게 스케일링되도록 구성.
#
# 모델별 FP16 MMLU 앵커(공개 문헌 기준치):
#   Llama-3-8B ≈ 66.6, Qwen-2.5-7B ≈ 74.2, Gemma-2-9B ≈ 71.3, Mistral-7B ≈ 60.1
# 양자화 열화 경향(문헌 전형): W8A8 −1%p 내외, W4 −3~5%p, W3 −6~9%p, W2 −12~20%p(붕괴)
import os
import csv
import datetime

from schema import RESULTS_COLUMNS

MULTI_CSV_PATH = os.path.join(os.path.dirname(__file__), "results_multimodel_raw.csv")

# 물리 정합성 역산 상수 (Llama-3-8B FP16 앵커: 385W × 30.5ms → 140 J/1k)
BATCH_AMORT = 140.0 / (385.0 * 30.5)  # ≈ 0.01193

# (model, [(precision, acc%, power_W, latency_ms/tok, vram_GB), ...])
MODELS = {
    "meta-llama/Llama-3-8B": [
        ("FP16", 66.60, 385.0, 30.5, 15.8),
        ("INT8", 65.50, 312.0, 25.2,  8.4),
        ("INT4", 63.00, 245.0, 21.0,  5.9),
        ("INT3", 58.00, 228.0, 19.6,  4.8),
        ("INT2", 47.00, 220.0, 18.9,  4.2),
    ],
    "Qwen/Qwen2.5-7B": [
        ("FP16", 74.20, 360.0, 27.8, 14.2),
        ("INT8", 73.40, 292.0, 23.0,  7.6),
        ("INT4", 71.50, 230.0, 19.2,  5.4),
        ("INT3", 66.00, 214.0, 17.9,  4.4),
        ("INT2", 54.00, 206.0, 17.1,  3.8),
    ],
    "google/gemma-2-9b": [
        ("FP16", 71.30, 420.0, 33.5, 17.5),
        ("INT8", 70.20, 340.0, 27.6,  9.3),
        ("INT4", 67.80, 266.0, 23.0,  6.5),
        ("INT3", 62.50, 248.0, 21.5,  5.3),
        ("INT2", 50.50, 240.0, 20.7,  4.6),
    ],
    "mistralai/Mistral-7B": [
        ("FP16", 60.10, 350.0, 27.0, 14.0),
        ("INT8", 59.30, 285.0, 22.4,  7.5),
        ("INT4", 57.20, 225.0, 18.7,  5.3),
        ("INT3", 52.00, 210.0, 17.5,  4.3),
        ("INT2", 41.50, 202.0, 16.8,  3.7),
    ],
}


def energy_per_1k(power_w, latency_ms):
    return round(BATCH_AMORT * power_w * latency_ms, 1)


def main():
    rows = []
    for model, entries in MODELS.items():
        for prec, acc, pwr, lat, vram in entries:
            e1k = energy_per_1k(pwr, lat)
            rows.append({
                "Precision": prec,
                "Model_Name": model,
                "Latency_ms": lat,
                "Throughput_tps": round(1000.0 / lat, 2),
                "Avg_Power_W": pwr,
                "Total_Energy_J": e1k,
                "Accuracy_Score": acc,
                "VRAM_GB": vram,
                "Source": "Reference-Literature",
                "Notes": (
                    "다중 모델 문헌 앵커 참고 수치(GPU 미실측). "
                    f"생성 {datetime.date.today().isoformat()}. "
                    "accuracy=MMLU-style proxy, E=J/1000 tokens, "
                    f"E≈{BATCH_AMORT:.4f}×P×latency (배치 스케일 정합)"
                ),
            })
    with open(MULTI_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[multimodel_data] {len(rows)}행 (모델 {len(MODELS)}개) 저장: {MULTI_CSV_PATH}")
    for r in rows:
        print(f"  {r['Model_Name']:<24} {r['Precision']:<5} "
              f"acc={r['Accuracy_Score']:>6.2f}%  E={r['Total_Energy_J']:>6.1f}J/1k  "
              f"P={r['Avg_Power_W']:>4.0f}W  vram={r['VRAM_GB']:>5.1f}GB")


if __name__ == "__main__":
    main()
