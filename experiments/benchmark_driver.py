# experiments/benchmark_driver.py
# GPU 추론 벤치마크 드라이버 (무료 Colab T4 16GB 최적화, 학술적 실측 구조).
#
# ★ 방법론 설계 (균일 PCAG 곡선):
#   양자화 방법을 섞지 않는다. 기본(--quant-method rtn)은 모든 비트폭
#   {FP16, INT8, INT6, INT5, INT4, INT3, INT2}를 한 가지 per-channel RTN
#   (lowbit.py)로 통일 측정 → 방법론 혼합에 오염되지 않는 미세 비트 PCAG 곡선.
#   표준 방법 비교 세트(--quant-method bnb)는 bitsandbytes INT8/INT4/FP4 를
#   별도 CSV로 기록해 곡선과 분리한다.
#
# 지원 Precision:
#   rtn (기본): FP16 / INT8 / INT6 / INT5 / INT4 / INT3 / INT2
#     - FP16 은 torch fp16, 나머지는 lowbit.py packed per-channel RTN
#   bnb : FP16 / INT8 / INT4 / FP4 (bitsandbytes, 표준 방법 검증 세트)
#
# Colab 세션 대응:
#   - 모델별 분할 실행: --models 에 실행할 모델만 지정 (세션당 1모델 권장)
#   - 정밀도별 즉시 저장: 각 precision 측정이 끝나는 즉시 CSV/체크포인트 기록
#     → 세션이 끊겨도 지금까지 측정분은 보존되고 --resume 으로 이어서 가능
#   - Google Drive 연동: --drive-dir <마운트 경로> 로 매 측정마다 결과 미러링
#   - OOM 방지: PYTORCH_CUDA_ALLOC_CONF, 배치 chunking, empty_cache, gc
#   - 반복 측정: --repeats 로 평균/표준편차 확보 (실측 신뢰구간)
#
# 사용법 예시 (Colab, 최적 프로토콜):
#   # 1. 균일 RTN PCAG 곡선 (7비트폭, 표준평가 ARC-Easy, 3회 반복)
#   python benchmark_driver.py --models meta-llama/Llama-3-8B \
#       --quant-method rtn --precisions FP16 INT8 INT6 INT5 INT4 INT3 INT2 \
#       --eval arc_easy --eval_questions 100 --repeats 3 --energy_norm \
#       --update-main --drive-dir /content/drive/MyDrive/pcag_results --resume
#   # 2. 표준 방법 검증 세트 (별도 CSV, results_raw.csv 는 건드리지 않음)
#   python benchmark_driver.py --models meta-llama/Llama-3-8B \
#       --quant-method bnb --precisions INT8 INT4 \
#       --drive-dir /content/drive/MyDrive/pcag_results --resume
import argparse
import gc
import os
import time
import csv
import json
import statistics

# torch는 지연 import (이 모듈은 분석/참고데이터 생성 시에도 import되므로 무거운 의존성 격리)
from schema import RESULTS_COLUMNS, CSV_PATH

# (모델, 정밀도) 정렬용 순서 — 비트폭 내림차순 (FP16 > INT8 > INT6 > INT5 > INT4 > INT3 > INT2)
PREC_ORDER = {"FP16": 0, "INT8": 1, "INT6": 2, "INT5": 3,
              "INT4": 4, "FP4": 4, "INT3": 5, "INT2": 6}
DEFAULT_MODELS = ["meta-llama/Llama-3-8B"]
RTN_PRECISIONS = ["FP16", "INT8", "INT6", "INT5", "INT4", "INT3", "INT2"]
BNB_PRECISIONS = ["FP16", "INT8", "INT4"]
RTN_BITS = {"INT8": 8, "INT6": 6, "INT5": 5, "INT4": 4, "INT3": 3, "INT2": 2}
MULTI_CSV_PATH = os.path.join(os.path.dirname(__file__), "results_multimodel_raw.csv")
BNB_CSV_PATH = os.path.join(os.path.dirname(__file__), "results_multimodel_bnb_raw.csv")


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
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True,max_split_size_mb:128")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_model(model_name, precision, device="cuda", method="rtn"):
    """precision: 'FP16'|'INT8'|'INT6'|'INT5'|'INT4'|'INT3'|'INT2'|'FP4'.

    method='rtn' (기본): 모든 저비트를 lowbit.py per-channel RTN 으로 통일
      (PCAG 곡선의 방법론 일관성). FP16 은 torch fp16.
    method='bnb': bitsandbytes 표준 방법 (INT8 / INT4-NF4 / FP4) — 검증 세트.
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
    use_lowbit = method == "rtn" and precision in RTN_BITS
    if use_lowbit:
        # T4(14.6GB) 에서 8B fp16(≈15GB) 은 GPU/CPU 어디에도 전체 상주 불가
        # (ISSUE-CODE-12). 체크포인트 shard 를 스트리밍으로 읽어 Linear 를
        # 즉시 packed 저비트로 변환 → 상주 메모리는 packed 가중치 + 임베딩뿐.
        from lowbit import load_lowbit_from_checkpoint
        return load_lowbit_from_checkpoint(model_name, RTN_BITS[precision], device)
    elif precision == "FP16":
        load_kwargs["torch_dtype"] = torch.float16
    elif method == "bnb" and precision in ("INT8", "INT4", "FP4"):
        if not HAS_BNB:
            raise RuntimeError(
                f"{precision} 필요하나 bitsandbytes 미설치 (pip install bitsandbytes)")
        load_kwargs["torch_dtype"] = torch.float16
        if precision == "INT8":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif precision == "INT4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True)
        else:  # FP4
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="fp4")
    else:
        raise ValueError(f"({method}) 에서 지원하지 않는 정밀도: {precision}")

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
    ap.add_argument("--quant-method", choices=["rtn", "bnb"], default="rtn",
                    help="rtn: lowbit 균일 per-channel RTN(PCAG 곡선용, 기본) | "
                         "bnb: bitsandbytes 표준 방법(검증 세트)")
    ap.add_argument("--precisions", nargs="+", default=None,
                    help="실행할 정밀도 (기본: rtn→FP16 INT8 INT6 INT5 INT4 INT3 INT2, "
                         "bnb→FP16 INT8 INT4)")
    ap.add_argument("--prompts", type=int, default=20, help="벤치마크 프롬프트 수")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4,
                    help="추론 배치 chunk 크기 (T4 OOM 방지, 줄일수록 VRAM 안전)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="정밀도당 반복 측정 수 (평균·표준편차 → 실측 신뢰구간)")
    ap.add_argument("--eval_subset", default="arc_easy",
                    help="평가: arc_easy(표준 0-shot 객관식) | synthetic(내장 합성)")
    ap.add_argument("--eval_questions", type=int, default=100,
                    help="정확도 평가 질문 수 (T4 시간 절약용)")
    ap.add_argument("--no_eval", action="store_true",
                    help="정확도 평가 생략 (시간 절약)")
    ap.add_argument("--output", default=None,
                    help="실측 결과 CSV (기본: rtn→results_multimodel_raw.csv, "
                         "bnb→results_multimodel_bnb_raw.csv)")
    ap.add_argument("--main-csv", default=CSV_PATH,
                    help="주요 모델(Llama-3-8B) 분석용 results_raw.csv")
    ap.add_argument("--update-main", action="store_true",
                    help="측정 모델이 results_raw.csv 에 있으면 함께 갱신 "
                         "(PCAG 곡선 기준 파일이므로 rtn 에서만 권장)")
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
    if args.precisions is None:
        args.precisions = list(RTN_PRECISIONS if args.quant_method == "rtn"
                               else BNB_PRECISIONS)
    if args.output is None:
        args.output = MULTI_CSV_PATH if args.quant_method == "rtn" else BNB_CSV_PATH

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
                  f"[method={args.quant_method}] ({key}) ===")
            torch.cuda.empty_cache()
            gc.collect()
            model, tokenizer = load_model(model_name, prec, device,
                                          method=args.quant_method)
            warmup(model, tokenizer, device, batch_size=args.batch_size)

            # 정확도는 결정적이므로 1회 평가 (반복 아님)
            accuracy = None
            if not args.no_eval:
                accuracy = eval_harness.run_eval(
                    model, tokenizer, device, subset=args.eval_subset,
                    num_questions=args.eval_questions)
            # --no_eval 시 Accuracy_Score 는 빈 값으로 기록 → 분석 단계에서 스킵.
            # (0.0 을 넣으면 A0=0 이 되어 Power Wall 판정 전체가 붕괴하므로 금지)

            # 반복 측정: 지연/처리량/전력/에너지 의 mean±std 확보
            latencies, tpss, powers, energies = [], [], [], []
            total_new = 0
            vram = None
            power_estimated = False
            for rep in range(args.repeats):
                bench = {}

                def _bench():
                    # 전력 계측과 성능 측정을 동일 호출에서 수행 (에너지-성능 정합)
                    bench["res"] = run_inference(model, tokenizer, prompts,
                                                 args.max_new_tokens, device,
                                                 args.batch_size)
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
                latencies.append(latency)
                tpss.append(tps)
                if avg_power is not None:
                    powers.append(avg_power)
                if energy is not None:
                    energies.append(energy)

            def _mean(xs):
                return sum(xs) / len(xs) if xs else None

            def _std(xs):
                return (statistics.stdev(xs) if len(xs) > 1 else 0.0) if xs else None

            latency, tps = _mean(latencies), _mean(tpss)
            avg_power = _mean(powers)
            energy = _mean(energies)
            # 에너지 단위 정규화: 배치 전체 J → J/1,000 tokens (문헌 앵커와 비교 가능)
            energy_norm = False
            if args.energy_norm and energy is not None and total_new:
                energy = energy * 1000.0 / max(1, total_new)
                energy_norm = True

            notes = (
                f"{args.notes}. method={args.quant_method}, "
                f"max_new_tokens={args.max_new_tokens}, "
                f"prompts={args.prompts}, batch={args.batch_size}, "
                f"repeats={args.repeats}, "
                f"eval={args.eval_subset if not args.no_eval else 'skipped'}, "
                f"eval_questions={args.eval_questions if not args.no_eval else 0}"
            )
            if args.quant_method == "rtn" and prec in RTN_BITS:
                notes += (f", quant=uniform per-channel RTN {prec} (lowbit, "
                          "uncalibrated — same method across all bit-widths)")
            elif args.quant_method == "bnb" and prec != "FP16":
                notes += f", quant=bitsandbytes {prec} (standard calibrated)"
            if args.repeats > 1:
                notes += (
                    f", std_lat={_std(latencies):.4f}ms, std_tps={_std(tpss):.2f}, "
                    f"std_power={_std(powers):.2f}W, std_energy={_std(energies):.2f}J"
                    if _std(latencies) is not None else ""
                )
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
            # results_raw.csv 는 균일 RTN PCAG 곡선의 기준 파일이므로 rtn 에서만 갱신.
            # bnb 검증 세트가 섞이면 곡선의 방법론 일관성이 깨진다.
            if args.update_main:
                if args.quant_method != "rtn":
                    print("  [경고] --update-main 은 rtn 에서만 유효 → bnb 는 기준 CSV 미갱신")
                else:
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
            if args.repeats > 1:
                print(f"  (repeats={args.repeats}, std_lat={_std(latencies):.4f}, "
                      f"std_energy={_std(energies):.2f})")
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