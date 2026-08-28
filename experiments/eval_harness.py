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


def run_eval(model, tokenizer, device, subset="synthetic", **kw):
    """통합 평가 진입점. accuracy(%) 반환."""
    if subset == "synthetic":
        return SyntheticLogicTask(**kw).generate(tokenizer, model, device)
    # (실측 데이터 확보 시 확장 지점)
    raise NotImplementedError(
        f"subset={subset} 평가는 데이터셋 준비 후 지원됩니다. "
        "현재는 'synthetic' 사용.")


if __name__ == "__main__":
    print("eval_harness 모듈 로드 확인. GPU 연결 시 run_eval() 호출.")