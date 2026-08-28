# experiments/benchmark_driver.py
# GPU 추론 벤치마크 드라이버.
# - 다중 정밀도(FP16 / INT8 / INT4) 로 LLM 로드 및 추론.
# - 전원 계측(PyNVML/nvidia-smi) + 지연/처리량/VRAM 측정.
# - 정확도 평가(eval_harness) 후 experiments/results_raw.csv 에 기록.
#
# 사용법(실측 시):
#   python benchmark_driver.py --model meta-llama/Llama-3-8B --prompts 40 --max_new_tokens 128
#
# 현재 GPU가 없는 환경에서 --reference 데이터 생성은 generate_results.py 를 사용.
import argparse
import os
import time
import csv

# torch는 지연 import (이 모듈은 분석/참고데이터 생성 시에도 import되므로 무거운 의존성 격리)
from schema import RESULTS_COLUMNS, CSV_PATH


def _lazy_imports():
    """GPU 실측 시에만 필요한 무거운 의존성들을 로드."""
    import torch
    from telemetry import PowerMeter
    import eval_harness as _eh
    return torch, PowerMeter, _eh


def has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def load_model(model_name, precision, device="cuda"):
    """precision: 'FP16'|'INT8'|'INT4' (참고: FP32 fallback은 FP16으로 취급)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # bitsandbytes 로딩 여부
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
        HAS_BNB = True
    except Exception:
        HAS_BNB = False
    load_kwargs = {"device_map": "auto", "low_cpu_mem_usage": True}
    if precision == "FP16":
        load_kwargs["torch_dtype"] = torch.float16
    elif precision == "INT8":
        if not HAS_BNB:
            raise RuntimeError("INT8 필요하나 bitsandbytes 미설치")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "INT4":
        if not HAS_BNB:
            raise RuntimeError("INT4 필요하나 bitsandbytes 미설치")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4")
    else:
        raise ValueError(f"지원하지 않는 정밀도: {precision}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def measure_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return None


def run_inference(model, tokenizer, prompts, max_new_tokens, device):
    """배치 추론. (avg_latency_ms_per_token, throughput_tps, total_new_tokens) 반환."""
    encodings = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    n_input = encodings.input_ids.shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **encodings, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    dt = time.time() - t0
    total_new = (out.shape[1] - n_input) * out.shape[0]
    latency_ms_per_token = (dt * 1000.0) / max(1, total_new)
    throughput_tps = total_new / max(dt, 1e-9)
    return latency_ms_per_token, throughput_tps, total_new


def warmup(model, tokenizer, device, max_new_tokens=16):
    prompt = "안녕하세요. 테스트 문장입니다."
    run_inference(model, tokenizer, [prompt], max_new_tokens, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3-8B")
    ap.add_argument("--prompts", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--precisions", nargs="+", default=["FP16", "INT8", "INT4"])
    ap.add_argument("--eval_subset", default="synthetic")
    ap.add_argument("--output", default=CSV_PATH)
    args = ap.parse_args()

    torch, PowerMeter, eval_harness = _lazy_imports()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU가 없습니다. 실측은 GPU 필요. 참고 데이터는 generate_results.py 사용.")

    device = "cuda"
    meter = PowerMeter()
    print(f"PowerMeter available={meter.available}")
    if not meter.available:
        print("경고: 전원 계측 불가 → Total_Energy_J 는 추정값으로 기록")

    prompts = [f"지시: 다음 질문에 답하세요. 질문 {i}: 대한민국의 수도는?"
               for i in range(args.prompts)]

    rows = []
    for prec in args.precisions:
        print(f"\n=== Loading {args.model} as {prec} ===")
        model, tokenizer = load_model(args.model, prec, device)
        warmup(model, tokenizer, device)

        # 전원 계측 하에서 성능 측정
        def _bench():
            return run_inference(model, tokenizer, prompts, args.max_new_tokens, device)

        t0 = time.time()
        avg_power, energy = meter.measure(_bench)
        latency, tps, total_new = _bench()  # 별도 호출로 성능치 확보 (계측 오버헤드 분리)
        dt = time.time() - t0
        if energy is None:
            energy = avg_power * dt if avg_power else None

        accuracy = eval_harness.run_eval(
            model, tokenizer, device, subset=args.eval_subset)
        vram = measure_vram_gb()

        rows.append({
            "Precision": prec,
            "Model_Name": args.model,
            "Latency_ms": round(latency, 4),
            "Throughput_tps": round(tps, 4),
            "Avg_Power_W": round(avg_power, 2) if avg_power else None,
            "Total_Energy_J": round(energy, 2) if energy else None,
            "Accuracy_Score": round(accuracy, 3),
            "VRAM_GB": round(vram, 3) if vram else None,
            "Source": "Measured-GPU",
            "Notes": f"max_new_tokens={args.max_new_tokens}",
        })
        print(f"{prec}: acc={accuracy:.2f}% latency={latency:.2f}ms/tok "
              f"tps={tps:.1f} power={avg_power}W energy={energy}J vram={vram}GB")

        del model
        torch.cuda.empty_cache()

    write_csv(args.output, rows)
    print(f"\n결과 저장: {args.output}")


def write_csv(path, rows, append=False):
    exists = os.path.exists(path)
    with open(path, "a" if append else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if not exists or not append:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in RESULTS_COLUMNS})


if __name__ == "__main__":
    main()