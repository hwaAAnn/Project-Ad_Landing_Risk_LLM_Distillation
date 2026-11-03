"""Shared pieces for the local (student) model: prompt format, loading, scoring

Used by `finetune_qlora.py` (training) and `predict_local.py` (zero-shot and
fine-tuned inference), so that both see **exactly the same prompt**. If training
and inference formatted the input differently, the fine-tuned score would measure
the formatting gap as much as the learning.

The task is the binary label of `build_trainset.py` (LOW = 0, MEDIUM · HIGH = 1).
The model answers with a single word, and the prediction is read from the
next-token logits of the two label words rather than from free-form generation.
This removes parse failures and gives a probability that can be thresholded.

torch / transformers / peft are imported lazily, so the pure functions here can be
tested without a GPU stack (see `test_local_judge.py`).
"""

from __future__ import annotations

LABEL_WORDS = {0: "SAFE", 1: "RISK"}

# Appended to the sealed judging criteria. The criteria themselves are not edited,
# so the student is conditioned on the same rules the teachers were given.
ANSWER_INSTRUCTION = (
    "\n\n## Output\n"
    "Do not call any tool and do not explain. Answer with exactly one word:\n"
    f"- {LABEL_WORDS[1]} if the page is MEDIUM or HIGH risk under the criteria above\n"
    f"- {LABEL_WORDS[0]} if it is LOW risk"
)

DEFAULT_BASE = "Qwen/Qwen3-4B"
DEFAULT_MAX_LEN = 4096


def system_prompt(spec: dict) -> str:
    return spec["system"] + ANSWER_INSTRUCTION


def messages(system: str, text: str) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": text}]


def render(tokenizer, system: str, text: str) -> str:
    """Chat-template the prompt up to (not including) the answer.

    Qwen3 is a hybrid reasoning model. `enable_thinking=False` makes the template
    emit an empty think block, so the first generated token is the answer itself.
    Tokenizers that do not know the flag ignore it.
    """
    return tokenizer.apply_chat_template(messages(system, text), tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)


def fit_text(text_ids: list[int], budget: int) -> tuple[list[int], bool]:
    """Truncate the user text to `budget` tokens. Returns (ids, truncated).

    The head is kept: the landing URL, the ad link and the start of the fetched
    page come first in the assembled prompt and carry most of the signal. The tail
    is cut first, and the closing instruction line is restored by the caller.
    """
    if budget <= 0:
        raise ValueError(f"no room for the user text (budget={budget}); raise --max-len")
    if len(text_ids) <= budget:
        return text_ids, False
    return text_ids[:budget], True


def encode_prompt(tokenizer, system: str, text: str, max_len: int,
                  reserve: int = 8) -> tuple[list[int], bool]:
    """Token ids of the rendered prompt, at most `max_len - reserve` long.

    `reserve` leaves room for the answer tokens during training. The system prompt
    is never truncated (it is the sealed criteria); only the user text is.
    """
    full = tokenizer(render(tokenizer, system, text), add_special_tokens=False)["input_ids"]
    if len(full) <= max_len - reserve:
        return full, False
    overhead = len(tokenizer(render(tokenizer, system, ""),
                             add_special_tokens=False)["input_ids"])
    text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    kept, _ = fit_text(text_ids, max_len - reserve - overhead - 16)
    short = tokenizer.decode(kept) + "\n…(truncated)"
    ids = tokenizer(render(tokenizer, system, short), add_special_tokens=False)["input_ids"]
    return ids[: max_len - reserve], True


def label_token_ids(tokenizer) -> dict[int, int]:
    """label → id of the first token of its answer word. The two must differ"""
    out = {lab: tokenizer(w, add_special_tokens=False)["input_ids"][0]
           for lab, w in LABEL_WORDS.items()}
    if out[0] == out[1]:
        raise SystemExit(f"label words share a first token ({out}); pick other words")
    return out


def answer_ids(tokenizer, label: int) -> list[int]:
    """Training target: the label word followed by the end-of-turn token"""
    ids = tokenizer(LABEL_WORDS[label], add_special_tokens=False)["input_ids"]
    return ids + [tokenizer.eos_token_id]


def p_risk_from_logits(logit_safe: float, logit_risk: float) -> float:
    """Two-way softmax over the label tokens only"""
    import math
    m = max(logit_safe, logit_risk)
    a, b = math.exp(logit_safe - m), math.exp(logit_risk - m)
    return b / (a + b)


def load_tokenizer(base: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def bnb_config():
    """4-bit NF4 with double quantization and bf16 compute (the QLoRA recipe)"""
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16)


def load_model(base: str, adapter: str | None = None, four_bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM
    kw = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if four_bit:
        kw["quantization_config"] = bnb_config()
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def score(model, tokenizer, ids: list[int], label_ids: dict[int, int]) -> float:
    """P(RISK) for one encoded prompt"""
    import torch
    with torch.no_grad():
        x = torch.tensor([ids], device=model.device)
        logits = model(input_ids=x).logits[0, -1].float()
    return p_risk_from_logits(logits[label_ids[0]].item(), logits[label_ids[1]].item())
