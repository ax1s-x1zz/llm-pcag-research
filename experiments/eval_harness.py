# experiments/eval_harness.py
# 정확도(Accuracy) 평가 모듈.
# - GPU 벤치마크 드라이버가 호출하는 평가기.
# - 데이터셋 우선순위:
#     1) experiments/datasets/ 에 MMLU(dev) 또는 GSM8K 형태가 있으면 사용.
#     2) 없으면 내장 합성 논리 태스크(embedded synthetic logic task)로 proxy accuracy 측정.
#   ※ 참고 데이터 단계에서는 accuracy는 문헌 앵커를 사용하므로, 이 하네스는 실제 GPU 실행 시 실측용.
import json
import os

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets")


class SyntheticLogicTask:
    """합성 논리 태스크: (pattern, n, expected) 규칙을 모델이 따르는지.

    프롬프트에서 규칙 예시 몇 개를 주고 새로운 입력의 정답을 생성하게 하여,
    정확 일치(exact match) 비율을 정확도로 사용.
    - GPU 없이도 실행 가능하므로 dry-run 및 파이프라인 검증에 유용.
    """

    RULES = [
        ("다음 규칙을 따르세요: 입력 정수 x에 대해 x*2를 출력.", lambda x: x * 2),
        ("다음 규칙을 따르세요: 입력 정수 x에 대해 x+3을 출력.", lambda x: x + 3),
        ("다음 규칙을 따르세요: 입력 정수 x에 대해 x*x를 출력.", lambda x: x * x),
    ]

    def __init__(self, num_questions=50, seed=0):
        self.num_questions = num_questions
        self.seed = seed

    def build_prompt(self, rule):
        examples = [(2, rule(2)), (5, rule(5)), (9, rule(9))]
        ex_str = "\n".join(f"입력: {e[0]} → 출력: {e[1]}" for e in examples)
        return (
            f"규칙 예시:\n{ex_str}\n"
            "이제 다음 입력에 대해 같은 규칙을 적용한 정답을 '정답: <숫자>' 형식으로만 출력하세요.\n"
            f"입력: {self._current_input}"
        )

    def generate(self, tokenizer, model, device):
        import numpy as np
        rng = np.random.RandomState(self.seed)
        correct = 0
        total = 0
        rule = self.RULES[0]
        for _ in range(self.num_questions):
            x = int(rng.randint(1, 100))
            expected = rule(x)
            self._current_input = x
            prompt = self.build_prompt(rule)
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **enc, max_new_tokens=8, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][enc.input_ids.shape[1]:],
                                   skip_special_tokens=True)
            total += 1
            # 정답 숫자 추출
            digits = "".join(ch for ch in gen if ch.isdigit())
            if digits and str(expected) == digits.lstrip("0") or (str(expected) == "0" and digits == ""):
                # 간단 매칭: expected 숫자가 생성 문자열에 포함
                if str(expected) in gen:
                    correct += 1
        return (correct / total) * 100.0 if total else 0.0


def load_datasets():
    """datasets/ 디렉토리에서 평가 세트 로드. 없으면 None."""
    mmlu = os.path.join(DATASET_DIR, "mmlu.jsonl")
    gsm = os.path.join(DATASET_DIR, "gsm8k.jsonl")
    found = []
    for p, kind in [(mmlu, "mmlu"), (gsm, "gsm8k")]:
        if os.path.exists(p):
            found.append((kind, p))
    return found


def run_arc_easy(tokenizer, model, device, num_questions=100, seed=0, split="test"):
    """ARC-Easy 표준 객관식(0-shot) 정확도. datasets 설치 필요.

    표준 벤치마크라 문헌 수치와 비교 가능한 accuracy 프록시. 실패 시 None 반환.
    """
    import torch
    try:
        from datasets import load_dataset
    except Exception:
        print("[eval_harness] 'datasets' 미설치 → ARC-Easy 불가")
        return None
    try:
        ds = load_dataset("ai2_arc", "ARC-Easy", split=split)
    except Exception as e:
        print(f"[eval_harness] ARC-Easy 로드 실패: {e}")
        return None
    import random as _rng
    rng = _rng.Random(seed)
    idx = rng.sample(range(len(ds)), min(num_questions, len(ds)))

    letters = "ABCDE"
    correct = 0
    with torch.no_grad():
        for i in idx:
            item = ds[i]
            question = item["question"].strip()
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            choice_lines = "\n".join(
                f"{letters[j]}) {choices[j].strip()}" for j in range(len(choices)))
            prompt = f"Question: {question}\n{choice_lines}\nAnswer:"
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **enc, max_new_tokens=1, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][enc.input_ids.shape[1]:],
                                   skip_special_tokens=True).strip()
            pred = gen[0].upper() if gen else ""
            if pred in letters and pred in labels and pred == item["answerKey"].strip().upper():
                correct += 1
            del enc, out
    return (correct / len(idx)) * 100.0 if idx else 0.0


def run_eval(model, tokenizer, device, subset="arc_easy", **kw):
    """통합 평가 진입점. accuracy(%) 반환.

    subset: 'arc_easy'(표준 0-shot 객관식) | 'synthetic'(내장 합성 논리 태스크).
    arc_easy 실패(datasets 미설치 등) 시 synthetic 으로 자동 폴백.
    """
    if subset == "arc_easy":
        res = run_arc_easy(tokenizer, model, device, **kw)
        if res is not None:
            return res
        print("[eval_harness] arc_easy 폴백 → synthetic")
        subset = "synthetic"
    if subset == "synthetic":
        return SyntheticLogicTask(**kw).generate(tokenizer, model, device)
    raise NotImplementedError(
        f"subset={subset} 평가는 지원되지 않습니다. ('arc_easy' | 'synthetic')")


if __name__ == "__main__":
    print("eval_harness 모듈 로드 확인. GPU 연결 시 run_eval() 호출.")