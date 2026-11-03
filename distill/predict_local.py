"""Score the evaluation rows with a local model (zero-shot or fine-tuned)

    python -m distill.predict_local --trainset dataset/trainset_0831.jsonl --name qwen3-4b-zeroshot
    python -m distill.predict_local --trainset dataset/trainset_0831.jsonl --name qwen3-4b-qlora \
        --adapter adapters/qwen3-4b-qlora
    python -m distill.predict_local --trainset dataset/trainset_0831.jsonl --name qwen3-4b-qlora-f0 \
        --adapter adapters/qwen3-4b-qlora-fold0 --cv-fold 0

Writes `predictions/{name}.jsonl`, one line per case:
`{"case_id", "label", "pred", "p_risk", "truncated"}`. `eval_binary.py` reads it.

Zero-shot and fine-tuned runs use the same prompt (`local_judge.py`), so the only
difference between the two rows of the results table is the adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from pipeline import build_prompt as B
from distill import local_judge as L

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root


def eval_rows(rows: list[dict], split: str, cv_fold: int | None) -> list[dict]:
    """The rows to score: the held-out fold under CV, else the requested split"""
    if cv_fold is not None:
        return [r for r in rows if r["fold"] == cv_fold]
    return [r for r in rows if r["split"] == split]


def main() -> int:
    ap = argparse.ArgumentParser(description="local model → predictions/{name}.jsonl")
    ap.add_argument("--trainset", required=True)
    ap.add_argument("--name", required=True, help="output file name and table row label")
    ap.add_argument("--spec", default=os.path.join(HERE, "dataset", "prompt_spec.json"))
    ap.add_argument("--base", default=L.DEFAULT_BASE)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir. Omit for zero-shot")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cv-fold", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=L.DEFAULT_MAX_LEN)
    ap.add_argument("--no-4bit", action="store_true", help="load the base in bf16")
    args = ap.parse_args()

    with open(args.trainset, encoding="utf-8") as f:
        rows = eval_rows([json.loads(l) for l in f if l.strip()], args.split, args.cv_fold)
    if not rows:
        raise SystemExit("no rows to score — check --split / --cv-fold")

    spec = B.load_spec(args.spec)
    system = L.system_prompt(spec)
    tok = L.load_tokenizer(args.base)
    label_ids = L.label_token_ids(tok)
    model = L.load_model(args.base, args.adapter, four_bit=not args.no_4bit)

    out_dir = os.path.join(HERE, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.name}.jsonl")
    t0 = time.monotonic()
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for i, r in enumerate(rows, 1):
            ids, cut = L.encode_prompt(tok, system, r["text"], args.max_len, reserve=0)
            p = L.score(model, tok, ids, label_ids)
            f.write(json.dumps({"case_id": r["case_id"], "label": r["label"],
                                "pred": int(p >= args.threshold), "p_risk": round(p, 6),
                                "truncated": cut, "source": r.get("source", "")},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}  {time.monotonic() - t0:.0f}s")
    print(f"predictions → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
