"""Binary metrics for teachers and students on the same evaluation rows

    python -m distill.eval_binary --trainset dataset/trainset_0831.jsonl \
        --run "Gemini 2.5 Flash=results/gemini-flash/<run_id>.jsonl" \
        --run "Claude Haiku 4.5=results/claude/<run_id>.jsonl" \
        --run "GPT-5 mini=results/openai-mini/<run_id>.jsonl" \
        --pred "Qwen3-4B zero-shot=predictions/qwen3-4b-zeroshot.jsonl" \
        --pred "Qwen3-4B QLoRA=predictions/qwen3-4b-qlora.jsonl"

`--run` reads a `run_eval.py` result file (commercial LLM, 3-grade verdict) and maps
the grade to the binary label with `build_trainset.RISK_TO_LABEL`. `--pred` reads a
`predict_local.py` file (already binary). Ground truth always comes from the
training set, restricted to the evaluation rows (`--split`, or `--cv-fold`), so every
model is scored on the same cases.

Positive class is RISK (MEDIUM · HIGH). Reported per model:

| metric | definition |
| :-- | :-- |
| Accuracy | correct / evaluated |
| RISK Recall | TP / (TP + FN) |
| FN | danger pages missed: gold `HIGH` (original three-grade label) predicted SAFE, including failed calls |
| RISK Precision | TP / (TP + FP) |
| Macro-F1 | mean of the F1 of SAFE and RISK |

FN is counted on the original three-grade labels, not on the binary ones, because
letting a danger page through is the error that matters most. It is therefore not the
complement of RISK Recall. The binary miss count is kept as `missed` in `--json`.

A case a model failed on (schema failure, missing row) is **not dropped**: it counts
as wrong. On a RISK case that is a false negative, which is the conservative reading
for a safety classifier. Dropping it would flatter the model that fails most.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from pipeline.build_trainset import RISK_TO_LABEL

FAIL = None  # prediction value for a failed / missing case


def parse_named(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"expected NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


def eval_truth(rows: list[dict], split: str, cv_fold: int | None) -> dict[str, int]:
    """case_id → gold binary label for the evaluation rows"""
    keep = (lambda r: r["fold"] == cv_fold) if cv_fold is not None \
        else (lambda r: r["split"] == split)
    return {r["case_id"]: r["label"] for r in rows if keep(r)}


def danger_ids(rows: list[dict]) -> set[str]:
    """case_ids whose gold grade is HIGH on the original three-grade scale"""
    return {r["case_id"] for r in rows if r.get("risk") == "HIGH"}


def preds_from_run(lines: list[dict], raw: bool = False) -> dict[str, int | None]:
    """run_eval.py rows → case_id → binary prediction (None if the call failed)"""
    out: dict[str, int | None] = {}
    for r in lines:
        v = r.get("verdict_raw" if raw else "verdict") or {}
        risk = v.get("risk")
        out[r["case_id"]] = RISK_TO_LABEL.get(risk, FAIL) if r.get("ok") else FAIL
    return out


def preds_from_local(lines: list[dict]) -> dict[str, int | None]:
    return {r["case_id"]: r.get("pred", FAIL) for r in lines}


def metrics(truth: dict[str, int], pred: dict[str, int | None],
            danger: set[str] = frozenset()) -> dict:
    c = Counter()
    for cid, y in truth.items():
        p = pred.get(cid, FAIL)
        if p is FAIL:
            c["fail"] += 1
            p = 1 - y                       # failure counts as wrong
        if cid in danger and p == 0:
            c["fn_danger"] += 1
        c[("tp", "fn", "fp", "tn")[(0 if p else 1) if y else (2 if p else 3)]] += 1
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    n = tp + fn + fp + tn

    def f1(prec, rec):
        return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    rec_r = tp / (tp + fn) if tp + fn else 0.0
    prec_r = tp / (tp + fp) if tp + fp else 0.0
    rec_s = tn / (tn + fp) if tn + fp else 0.0
    prec_s = tn / (tn + fn) if tn + fn else 0.0
    return {"n": n, "accuracy": (tp + tn) / n if n else 0.0,
            "risk_recall": rec_r, "fn": c["fn_danger"], "missed": fn, "risk_precision": prec_r,
            "macro_f1": (f1(prec_r, rec_r) + f1(prec_s, rec_s)) / 2,
            "tp": tp, "fp": fp, "tn": tn, "fail": c["fail"]}


def table(results: list[tuple[str, str, dict]]) -> str:
    lines = ["| Model | Type | Accuracy | RISK Recall | FN | RISK Precision | Macro-F1 |",
             "| :-- | :-- | --: | --: | --: | --: | --: |"]
    for name, kind, m in results:
        lines.append(f"| {name} | {kind} | {m['accuracy']:.1%} | {m['risk_recall']:.1%} | "
                     f"{m['fn']} | {m['risk_precision']:.1%} | {m['macro_f1']:.3f} |")
    return "\n".join(lines)


def _read(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="binary metrics on a shared evaluation split")
    ap.add_argument("--trainset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--cv-fold", type=int, default=None)
    ap.add_argument("--run", action="append", default=[], help="NAME=run_eval result jsonl")
    ap.add_argument("--pred", action="append", default=[], help="NAME=predict_local jsonl")
    ap.add_argument("--raw", action="store_true", help="score verdict_raw (before post-processing)")
    ap.add_argument("--json", default="", help="also write metrics to this path")
    args = ap.parse_args()

    rows = _read(args.trainset)
    truth = eval_truth(rows, args.split, args.cv_fold)
    if not truth:
        raise SystemExit("no evaluation rows — check --split / --cv-fold")
    danger = danger_ids(rows) & truth.keys()
    print(f"evaluation rows: {len(truth)}  (RISK {sum(truth.values())}, of which danger "
          f"{len(danger)} · SAFE {len(truth) - sum(truth.values())})\n")

    results = []
    for spec in args.run:
        name, path = parse_named(spec)
        results.append((name, "Commercial LLM", metrics(truth, preds_from_run(_read(path), args.raw), danger)))
    for spec in args.pred:
        name, path = parse_named(spec)
        results.append((name, "Local", metrics(truth, preds_from_local(_read(path)), danger)))
    if not results:
        raise SystemExit("nothing to score — pass --run and/or --pred")

    print(table(results))
    for name, _kind, m in results:
        if m["fail"]:
            print(f"  ! {name}: {m['fail']} failed / missing case(s) counted as wrong")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({n: m for n, _k, m in results}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
