# experiments/benchmark_driver.py
# GPU 추론 벤치마크 드라이버 (무료 Colab T4 16GB 최적화).
#
# 지원 Precision:
#   FP16 : torch_dtype=float16 + device_map="auto"
#   INT8 : bitsandbytes load_in_8bit + device_map="auto"
#   INT4 : bitsandbytes NF4 (compute fp16, double quant) + device_map="auto"
#   INT3 / INT2 : bitsandbytes 미지원 → lowbit.py 로 packed 저비트 양자화
#                 (FP16 로 스트리밍 로드 → Linear 만 packed 변환 → GPU 상주)
#
# Colab 세션 대응:
#   - 모델별 분할 실행: --models 에 실행할 모델만 지정 (세션당 1모델 권장)
#   - 정밀도별 즉시 저장: 각 precision 측정이 끝나는 즉시 CSV/체크포인트 기록
#     → 세션이 끊겨도 지금까지 측정분은 보존되고 --resume 으로 이어서 가능
#   - Google Drive 연동: --drive-dir <마운트 경로> 로 매 측정마다 결과 미러링
#   - OOM 방지: PYTORCH_CUDA_ALLOC_CONF, 배치 chunking, empty_cache, gc
#
# 사용법 예시 (Colab):
#   python benchmark_driver.py --models meta-llama/Llama-3-8B --precisions FP16 INT8 INT4 INT3 INT2 \
#       --drive-dir /content/drive/MyDrive/pcag_results --resume
#   python benchmark_driver.py --models Qwen/Qwen2.5-7B --precisions FP16 INT8 INT4 INT3 INT2
import argparse
import gc
import os
import time
import csv
import json

# torch는 지연 import (이 모듈은 분석/참고데이터 생성 시에도 import되므로 무거운 의존성 격리)
from schema import RESULTS_COLUMNS, CSV_PATH

PREC_ORDER = {"FP16": 0, "INT8": 1, "INT4": 2, "INT3": 3, "INT2": 4}
DEFAULT_MODELS = ["meta-llama/Llama-3-8B"]
DEFAULT_PRECISIONS = ["FP16", "INT8", "INT4", "INT3", "INT2"]
MULTI_CSV_PATH = os.path.join(os.path.dirname(__file__), "results_multimodel_raw.csv")


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


def _prep_cuda_env():
    """T4 16GB 에서 CUDA 메모리 단편화/OOM 을 줄이는 환경 설정."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _safe_max_memory():
    """INT3/INT2 변환 중 여유를 두기 위한 GPU 메모리 상한 (전체의 ~80%)."""
    import torch
    try:
        _, total = torch.cuda.mem_get_info()
        limit = max(1, int(total * 0.80 // (1024 ** 3)))
        return {0: f"{limit}GiB"}
    except Exception:
        return None


def load_model(model_name, precision, device="cuda"):
    """precision: 'FP16'|'INT8'|'INT4'|'INT3'|'INT2'.

    FP16/INT8/INT4 는 bitsandbytes+accelerate(device_map="auto").
    INT3/INT2 는 lowbit.py packed 양자화 (bitsandbytes 미지원 비트).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
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
            raise RuntimeError("INT8 필요하나 bitsandbytes 미설치 (pip install bitsandbytes)")
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "INT4":
        if not HAS_BNB:
            raise RuntimeError("INT4 필요하나 bitsandbytes 미설치 (pip install bitsandbytes)")
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True)
    elif precision in ("INT3", "INT2"):
        # FP16 으로 스트리밍 로드(전부 GPU에 못 올리는 경우를 대비해 상한 설정) 후
        # Linear 를 packed 저비트로 교체 → 원본 fp16 해제 → GPU 상주.
        load_kwargs["torch_dtype"] = torch.float16
        mm = _safe_max_memory()
        if mm is not None:
            load_kwargs["max_memory"] = mm
    else:
        raise ValueError(f"지원하지 않는 정밀도: {precision}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if precision in ("INT3", "INT2"):
        from lowbit import quantize_model
        bits = {"INT3": 3, "INT2": 2}[precision]
        model = quantize_model(model, bits)
        # packed 저비트 가중치 + 임베딩만 GPU로 이동 (fp16 8B 대비 1/4~1/8 크기)
        model.to(device)
        torch.cuda.empty_cache()
        gc.collect()

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def measure_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return None


def run_inference(model, tokenizer, prompts, max_new_tokens, device, batch_size=4):
    """배치 chunking 추론. OOM 방지용 --batch_size 로 나눠 실행.

    (avg_latency_ms_per_token, throughput_tps, total_new_tokens) 반환.
    """
    import torch
    total_new = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            encodings = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
            out = model.generate(
                **encodings, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id)
            total_new += (out.shape[1] - encodings.input_ids.shape[1]) * out.shape[0]
            del encodings, out
    dt = time.time() - t0
    latency_ms_per_token = (dt * 1000.0) / max(1, total_new)
    throughput_tps = total_new / max(dt, 1e-9)
    return latency_ms_per_token, throughput_tps, total_new


def warmup(model, tokenizer, device, max_new_tokens=16, batch_size=4):
    prompt = "안녕하세요. 테스트 문장입니다."
    run_inference(model, tokenizer, [prompt], max_new_tokens, device, batch_size)


# ---------------------------------------------------------------- CSV 유틸
def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("Precision")]


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in RESULTS_COLUMNS})


def upsert_row(path, row):
    """(Model_Name, Precision) 기준으로 기존 행을 교체하고 새 행 추가 (즉시 저장).

    같은 모델·정밀도의 문헌 앵커(Reference-Literature)나 이전 실측 행을 덮어써
    재실행(idempotent)과 부분 진행(체크포인트)이 안전하게 동작한다.
    """
    key = (row["Model_Name"], row["Precision"])
    rows = [r for r in read_rows(path) if (r.get("Model_Name"), r.get("Precision")) != key]
    rows.append(row)
    rows.sort(key=lambda r: (r.get("Model_Name", ""), PREC_ORDER.get(r.get("Precision"), 9)))
    write_rows(path, rows)
    return rows


def copy_to_drive(src, drive_dir):
    """측정 결과를 Google Drive 경로로 즉시 미러링."""
    if not drive_dir:
        return
    os.makedirs(drive_dir, exist_ok=True)
    import shutil
    dst = os.path.join(drive_dir, os.path.basename(src))
    try:
        shutil.copy2(src, dst)
        print(f"  [drive] {src} → {dst}")
    except Exception as e:
        print(f"  [drive] 미러링 실패(무시): {e}")


# ---------------------------------------------------------------- 체크포인트
def load_checkpoint(path):
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return set(json.load(f).get("completed", []))
        except Exception:
            return set()
    return set()


def save_checkpoint(path, completed):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"completed": sorted(completed)}, f, ensure_ascii=False, indent=2)


def main():
    _prep_cuda_env()
    ap = argparse.ArgumentParser(
        description="GPU 실측 벤치마크 (Colab T4 최적화, 모델별 분할 실행 지원)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="실행할 모델 목록 (세션당 1개 권장: Colab 타임아웃 대비)")
    ap.add_argument("--model", default=None, help="단일 모델 (--models 의 별칭)")
    ap.add_argument("--precisions", nargs="+", default=DEFAULT_PRECISIONS,
                    help="실행할 정밀도 (기본: FP16 INT8 INT4 INT3 INT2)")
    ap.add_argument("--prompts", type=int, default=20, help="벤치마크 프롬프트 수")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4,
                    help="추론 배치 chunk 크기 (T4 OOM 방지, 줄일수록 VRAM 안전)")
    ap.add_argument("--eval_subset", default="synthetic")
    ap.add_argument("--eval_questions", type=int, default=20,
                    help="synthetic 정확도 평가 질문 수 (Colab 시간 절약용)")
    ap.add_argument("--no_eval", action="store_true",
                    help="정확도 평가 생략 (시간 절약)")
    ap.add_argument("--output", default=MULTI_CSV_PATH,
                    help="실측 결과 CSV (다중 모델 파일, 기본값)")
    ap.add_argument("--main-csv", default=CSV_PATH,
                    help="주요 모델(Llama-3-8B) 분석용 results_raw.csv")
    ap.add_argument("--update-main", action="store_true", default=True,
                    help="측정 모델이 results_raw.csv 에 있으면 함께 갱신")
    ap.add_argument("--drive-dir", default=None,
                    help="Google Drive 마운트 경로 (매 측정 즉시 CSV 미러링)")
    ap.add_argument("--checkpoint", default=None,
                    help="체크포인트 JSON 경로 (기본: <output>.checkpoint.json)")
    ap.add_argument("--resume", action="store_true",
                    help="완료된 (모델,정밀도) 건너뛰고 이어서 실행")
    ap.add_argument("--force", action="store_true",
                    help="--resume 이어도 완료분 재측정")
    ap.add_argument("--tdp", type=float, default=None,
                    help="torch.cuda 기반 전력 추정 폴백 시 TDP(W) 오버라이드")
    ap.add_argument("--energy_norm", action="store_true",
                    help="측정 에너지를 J/1,000 tokens 로 정규화 (문헌 앵커와 단위 일치)")
    ap.add_argument("--notes", default="Colab T4 GPU 실측 (Source=Measured-GPU)")
    args = ap.parse_args()

    if args.model:
        args.models = [args.model]

    torch, PowerMeter, eval_harness = _lazy_imports()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU가 없습니다. 실측은 GPU 필요. 참고 데이터는 generate_results.py 사용.")

    device = "cuda"
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} / VRAM {props.total_memory / 1024 ** 3:.1f} GB")
    free, total = torch.cuda.mem_get_info()
    print(f"  free={free / 1024 ** 3:.1f} GB / total={total / 1024 ** 3:.1f} GB")

    meter = PowerMeter()
    print(f"PowerMeter available={meter.available} "
          f"(pynvml={meter._use_pynvml}, nvidia-smi={meter._use_smi})")
    if not meter.available:
        print("경고: 실전력 계측 불가 → torch.cuda(TDP) 기반 추정값으로 기록")

    checkpoint_path = args.checkpoint or (args.output + ".checkpoint.json")
    completed = load_checkpoint(checkpoint_path)

    prompts = [f"지시: 다음 질문에 답하세요. 질문 {i}: 대한민국의 수도는?"
               for i in range(args.prompts)]

    for model_name in args.models:
        for prec in args.precisions:
            key = f"{model_name}::{prec}"
            if args.resume and not args.force and key in completed:
                print(f"[skip] {key} (체크포인트에 완료 기록됨)")
                continue

            print(f"\n=== Loading {model_name} as {prec} "
                  f"({key}) ===")
            torch.cuda.empty_cache()
            gc.collect()
            model, tokenizer = load_model(model_name, prec, device)
            warmup(model, tokenizer, device, batch_size=args.batch_size)

            bench = {}

            def _bench():
                # 전력 계측과 성능 측정을 동일 호출에서 수행 (에너지-성능 정합)
                bench["res"] = run_inference(model, tokenizer, prompts,
                                             args.max_new_tokens, device, args.batch_size)
                return bench["res"]

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            if meter.available:
                avg_power, energy = meter.measure(_bench)
            else:
                avg_power, energy = None, None
                _bench()
            latency, tps, total_new = bench.get("res", (None, None, None))
            dt = time.time() - t0
            vram = measure_vram_gb()

            # torch.cuda 기반 전력/에너지 추정 폴백 (pynvml/nvidia-smi 실패 시)
            power_estimated = False
            if avg_power is None or energy is None:
                est_p, est_e = meter.estimate(dt, tdp=args.tdp)
                if est_p is not None:
                    power_estimated = True
                    if avg_power is None:
                        avg_power = est_p
                    if energy is None:
                        energy = est_e
            if energy is None and avg_power is not None:
                energy = avg_power * dt
            # 에너지 단위 정규화: 배치 전체 J → J/1,000 tokens (문헌 앵커와 비교 가능)
            energy_norm = False
            if args.energy_norm and energy is not None and total_new:
                energy = energy * 1000.0 / max(1, total_new)
                energy_norm = True

            accuracy = None
            if not args.no_eval:
                accuracy = eval_harness.run_eval(
                    model, tokenizer, device, subset=args.eval_subset,
                    num_questions=args.eval_questions)
            # --no_eval 시 Accuracy_Score 는 빈 값으로 기록 → 분석 단계에서 스킵.
            # (0.0 을 넣으면 A0=0 이 되어 Power Wall 판정 전체가 붕괴하므로 금지)

            notes = (
                f"{args.notes}. max_new_tokens={args.max_new_tokens}, "
                f"prompts={args.prompts}, batch={args.batch_size}, "
                f"eval={args.eval_subset if not args.no_eval else 'skipped'}, "
                f"eval_questions={args.eval_questions if not args.no_eval else 0}"
            )
            if prec in ("INT3", "INT2"):
                notes += (f", quant=lowbit naive per-channel RTN {prec} "
                          "(uncalibrated, not GPTQ/AWQ)")
            if energy_norm:
                notes += ", E=J/1000 tokens"
            if power_estimated:
                notes += ", power=torch.cuda TDP estimate (not real power draw)"

            row = {
                "Precision": prec,
                "Model_Name": model_name,
                "Latency_ms": round(latency, 4),
                "Throughput_tps": round(tps, 4),
                "Avg_Power_W": round(avg_power, 2) if avg_power else "",
                "Total_Energy_J": round(energy, 2) if energy else "",
                "Accuracy_Score": round(accuracy, 3) if accuracy is not None else "",
                "VRAM_GB": round(vram, 3) if vram else "",
                "Source": "Measured-GPU",
                "Notes": notes,
            }
            # 매 측정 즉시 저장 (세션 끊김 대비 체크포인트)
            upsert_row(args.output, row)
            copy_to_drive(args.output, args.drive_dir)
            if args.update_main:
                main_rows = read_rows(args.main_csv)
                if any(r.get("Model_Name") == model_name for r in main_rows):
                    upsert_row(args.main_csv, row)
                    copy_to_drive(args.main_csv, args.drive_dir)
            completed.add(key)
            save_checkpoint(checkpoint_path, completed)
            copy_to_drive(checkpoint_path, args.drive_dir)

            acc_str = f"{accuracy:.2f}%" if accuracy is not None else "n/a"
            print(f"{prec}: acc={acc_str} latency={latency:.2f}ms/tok "
                  f"tps={tps:.1f} power={avg_power}W energy={energy}J vram={vram}GB")
            print(f"  [save] {args.output} + checkpoint")

            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\n결과 저장: {args.output}")
    print(f"체크포인트: {checkpoint_path} (완료 {len(completed)}건)")
    if args.drive_dir:
        print(f"Drive 미러: {args.drive_dir}")


if __name__ == "__main__":
    main()