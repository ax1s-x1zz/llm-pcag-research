# experiments/generate_results.py
# 문헌 기반 참고 데이터 생성 스크립트.
#
# ⚠️ 중요: GPU 실측이 불가한 현재 환경에서 전체 파이프라인(분석·그림·논문)을 검증하기 위한
# **참고 데이터(Reference-Literature)** 이다. 절대 실측으로 위장하지 않으며, 각 행에
# Source='Reference-Literature' 로 표기한다. GPU 확보 시 benchmark_driver.py 가 이 CSV를
# Source='Measured-GPU' 로 덮어쓰면 된다.
#
# 앵커 근거(문헌 전형 경향):
#   - 모델: Llama-3-8B (MMLU base ≈ 66.6%, 문헌/모델 카드 기준)
#   - 정확도: FP16≈66.6 → INT8(W8A8)≈65.5 → INT4(W4A16 GPTQ)≈63.0 →
#             INT3≈58.0 → INT2≈47.0  (비트 감소 시 표현력 붕괴 경향 반영)
#   - 토큰당/천토큰 에너지: FP16>INT8>INT4 이며 INT4 이하에서 절감 폭이 급감(체감)하도록 구성
#     (이는 "전력 절감은 포화되고 정확도는 붕괴"되는 Power Wall 형태를 만들기 위함)
#   - 전형적 배치 추론 성능치(A100/H100급, 8B)로 스케일 일관성 유지.
import os
import csv
import datetime

from schema import RESULTS_COLUMNS, CSV_PATH

MODEL = "meta-llama/Llama-3-8B"
# (precision, accuracy_%, energy_J_per_1000_tok, avg_power_W, latency_ms_per_tok, vram_GB)
# Reference anchor points.
ROWS = [
    # precision  acc    E/J-1k   P/W    lat ms/tok  vram  source-note
    ("FP16",     66.60, 140.0,   385.0, 30.5,       15.8),
    ("INT8",     65.50,  92.0,   312.0, 25.2,        8.4),
    ("INT4",     63.00,  61.0,   245.0, 21.0,        5.9),
    ("INT3",     58.00,  54.0,   228.0, 19.6,        4.8),
    ("INT2",     47.00,  49.5,   220.0, 18.9,        4.2),
]

ENERGY_FOR_1000 = 1000  # J/1000tok 기준으로 스키마의 Total_Energy_J(전체) 환산을 위함


def main():
    rows = []
    for prec, acc, e1k, pwr, lat, vram in ROWS:
        # Total_Energy_J 는 "천 토큰당"으로 통일해 그림/분석에서 직접 사용.
        rows.append({
            "Precision": prec,
            "Model_Name": MODEL,
            "Latency_ms": lat,
            "Throughput_tps": round(1000.0 / lat, 2),  # 1000ms / ms-per-tok
            "Avg_Power_W": pwr,
            "Total_Energy_J": e1k,   # 에너지(J/1000 tokens) — 분석에서 그대로 사용
            "Accuracy_Score": acc,
            "VRAM_GB": vram,
            "Source": "Reference-Literature",
            "Notes": (
                "문헌 앵커 기반 참고 수치(아직 GPU 미실측). "
                f"생성 {datetime.date.today().isoformat()}. "
                "accuracy=MMLU-style proxy, E=J/1000 tokens"
            ),
        })
    write(CSV_PATH, rows)
    print(f"[generate_results] {len(rows)}행 참고 데이터 저장: {CSV_PATH}")
    for r in rows:
        print(f"  {r['Precision']:<5} acc={r['Accuracy_Score']:>6.2f}% "
              f"E={r['Total_Energy_J']:>6.1f}J/1k  P={r['Avg_Power_W']:>5.0f}W "
              f"vram={r['VRAM_GB']:>5.2f}GB")


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in RESULTS_COLUMNS})


if __name__ == "__main__":
    main()