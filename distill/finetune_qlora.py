"""QLoRA fine-tuning of the student on the distillation training set

    python -m distill.finetune_qlora --trainset dataset/trainset_0831.jsonl
    python -m distill.finetune_qlora --trainset dataset/trainset_0831.jsonl --cv-fold 0   # one CV fold

Trains on `split == "train"` rows by default. With `--cv-fold k` it trains on every
row whose `fold != k` instead, so that `predict_local.py --cv-fold k` can score the
held-out fold. The two protocols are independent, as in `build_trainset.py`.

Only the answer tokens contribute to the loss. The prompt (sealed criteria + the
assembled user text) is masked with -100, so the model is not trained to reproduce
the page text, only to map it to a label.

Requires the GPU stack in `requirements-local.txt`. Nothing here calls a remote API.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

from pipeline import build_prompt as B
from distill import local_judge as L

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select(rows: list[dict], cv_fold: int | None) -> list[dict]:
    if cv_fold is None:
        return [r for r in rows if r["split"] == "train"]
    return [r for r in rows if r["fold"] != cv_fold]


def build_examples(rows, tokenizer, system: str, max_len: int) -> tuple[list[dict], int]:
    """rows → [{"input_ids", "labels"}], number of truncated prompts"""
    examples, truncated = [], 0
    for r in rows:
        prompt, cut = L.encode_prompt(tokenizer, system, r["text"], max_len)
        truncated += cut
        answer = L.answer_ids(tokenizer, r["label"])
        examples.append({"input_ids": prompt + answer,
                         "labels": [-100] * len(prompt) + answer})
    return examples, truncated


class Collator:
    """Right-pad a batch. Padding is masked in both attention and loss"""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


def main() -> int:
    ap = argparse.ArgumentParser(description="QLoRA fine-tuning of the student classifier")
    ap.add_argument("--trainset", required=True)
    ap.add_argument("--spec", default=os.path.join(HERE, "dataset", "prompt_spec.json"))
    ap.add_argument("--base", default=L.DEFAULT_BASE)
    ap.add_argument("--out", default="", help="adapter directory (default: adapters/<base>-qlora[-foldK])")
    ap.add_argument("--cv-fold", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=L.DEFAULT_MAX_LEN)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    random.seed(args.seed)

    rows = select(load_rows(args.trainset), args.cv_fold)
    if not rows:
        raise SystemExit("no training rows selected — check --trainset and --cv-fold")
    print(f"train rows: {len(rows)}  labels: {dict(Counter(r['label'] for r in rows))}")

    spec = B.load_spec(args.spec)
    system = L.system_prompt(spec)
    tok = L.load_tokenizer(args.base)
    examples, truncated = build_examples(rows, tok, system, args.max_len)
    print(f"prompts truncated to --max-len {args.max_len}: {truncated}/{len(examples)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=L.bnb_config(), device_map="auto",
        torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    name = args.base.split("/")[-1].lower() + "-qlora"
    if args.cv_fold is not None:
        name += f"-fold{args.cv_fold}"
    out = args.out or os.path.join(HERE, "adapters", name)

    trainer = Trainer(
        model=model,
        train_dataset=examples,
        data_collator=Collator(tok.pad_token_id),
        args=TrainingArguments(
            output_dir=out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            lr_scheduler_type="cosine", warmup_ratio=0.05, bf16=True,
            optim="paged_adamw_8bit", logging_steps=5, save_strategy="no",
            report_to=[], remove_unused_columns=False, seed=args.seed))
    trainer.train()

    model.save_pretrained(out)
    tok.save_pretrained(out)
    with open(os.path.join(out, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"base": args.base, "trainset": os.path.basename(args.trainset),
                   "cv_fold": args.cv_fold, "rows": len(rows), "max_len": args.max_len,
                   "truncated": truncated, "spec_id": spec.get("spec_id"),
                   "system_sha256": spec.get("system_sha256"),
                   **{k: getattr(args, k) for k in
                      ("epochs", "lr", "batch", "grad_accum", "lora_r",
                       "lora_alpha", "lora_dropout", "seed")}}, f, indent=2)
    print(f"adapter → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
