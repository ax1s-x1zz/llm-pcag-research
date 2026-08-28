# experiments/schema.py
# 실험 CSV 공용 스키마 정의. torch 등 무거운 의존성 없이 import 가능해야 함.
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "results_raw.csv")

RESULTS_COLUMNS = [
    "Precision", "Model_Name", "Latency_ms", "Throughput_tps",
    "Avg_Power_W", "Total_Energy_J", "Accuracy_Score", "VRAM_GB",
    "Source", "Notes",
]