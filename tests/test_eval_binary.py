"""eval_binary.py and the pure parts of local_judge.py — small hand-made inputs

    python -m tests.test_eval_binary

Same shape as the other test_*.py files: each check returns (failures, total) and
main sums them into the exit code. No network, no GPU stack.
"""

from __future__ import annotations

from distill import eval_binary as E
from distill import finetune_qlora as F
from distill import local_judge as L
from distill import predict_local as P


def _check(label: str, got, want) -> int:
    if got == want:
        return 0
    print(f"  FAIL {label}\n    want: {want!r}\n    got : {got!r}")
    return 1


def _close(label: str, got: float, want: float) -> int:
    return _check(label, round(got, 6), round(want, 6))


def test_metrics():
    fails = 0
    # 4 RISK (3 caught, 1 missed) and 4 SAFE (1 false alarm)
    truth = {f"r{i}": 1 for i in range(4)} | {f"s{i}": 0 for i in range(4)}
    pred = {"r0": 1, "r1": 1, "r2": 1, "r3": 0, "s0": 1, "s1": 0, "s2": 0, "s3": 0}
    m = E.metrics(truth, pred)
    fails += _check("counts", (m["tp"], m["missed"], m["fp"], m["tn"]), (3, 1, 1, 3))
    fails += _close("accuracy", m["accuracy"], 6 / 8)
    fails += _close("risk recall", m["risk_recall"], 3 / 4)
    fails += _close("risk precision", m["risk_precision"], 3 / 4)
    fails += _close("macro f1", m["macro_f1"], 0.75)
    fails += _check("binary misses kept", m["missed"], 1)
    fails += _check("no danger set → FN 0", m["fn"], 0)
    return fails, 7


def test_fn_danger():
    """FN counts only HIGH pages let through, on the original three-grade labels"""
    fails = 0
    rows = [{"case_id": "h0", "label": 1, "risk": "HIGH", "split": "test", "fold": 0},
            {"case_id": "h1", "label": 1, "risk": "HIGH", "split": "test", "fold": 0},
            {"case_id": "m0", "label": 1, "risk": "MEDIUM", "split": "test", "fold": 0},
            {"case_id": "l0", "label": 0, "risk": "LOW", "split": "test", "fold": 0},
            {"case_id": "h9", "label": 1, "risk": "HIGH", "split": "train", "fold": 1}]
    truth = E.eval_truth(rows, "test", None)
    danger = E.danger_ids(rows) & truth.keys()
    fails += _check("danger limited to eval rows", danger, {"h0", "h1"})
    m = E.metrics(truth, {"h0": 0, "h1": None, "m0": 0, "l0": 0}, danger)
    fails += _check("missed and failed HIGH both count", m["fn"], 2)
    fails += _check("missed MEDIUM is not FN", m["missed"], 3)
    m = E.metrics(truth, {"h0": 1, "h1": 1, "m0": 0, "l0": 0}, danger)
    fails += _check("caught HIGH is not FN", m["fn"], 0)
    return fails, 4


def test_failures_count_as_wrong():
    fails = 0
    truth = {"r0": 1, "r1": 1, "s0": 0}
    m = E.metrics(truth, {"r0": 1, "r1": None})          # r1 failed, s0 missing
    fails += _check("failed RISK is a false negative", m["missed"], 1)
    fails += _check("missing SAFE is a false positive", m["fp"], 1)
    fails += _check("fail count", m["fail"], 2)
    fails += _check("nothing dropped", m["n"], 3)
    return fails, 4


def test_preds_from_run():
    fails = 0
    rows = [
        {"case_id": "a", "ok": True, "verdict": {"risk": "HIGH"}, "verdict_raw": {"risk": "HIGH"}},
        {"case_id": "b", "ok": True, "verdict": {"risk": "LOW"}, "verdict_raw": {"risk": "MEDIUM"}},
        {"case_id": "c", "ok": False, "verdict": None, "verdict_raw": None},
        {"case_id": "d", "ok": True, "verdict": {"risk": "UNKNOWN"}},
    ]
    fails += _check("post-processed", E.preds_from_run(rows),
                    {"a": 1, "b": 0, "c": None, "d": None})
    fails += _check("raw", E.preds_from_run(rows, raw=True)["b"], 1)
    return fails, 2


def test_eval_rows_agree():
    """Training, prediction and scoring must carve the same rows"""
    fails = 0
    rows = [{"case_id": f"c{i}", "label": i % 2, "split": "test" if i < 3 else "train",
             "fold": i % 5} for i in range(10)]
    fails += _check("split: predict == truth",
                    sorted(r["case_id"] for r in P.eval_rows(rows, "test", None)),
                    sorted(E.eval_truth(rows, "test", None)))
    fails += _check("cv: predict == truth",
                    sorted(r["case_id"] for r in P.eval_rows(rows, "test", 2)),
                    sorted(E.eval_truth(rows, "test", 2)))
    train = {r["case_id"] for r in F.select(rows, 2)}
    held = {r["case_id"] for r in P.eval_rows(rows, "test", 2)}
    fails += _check("cv: train and held-out fold are disjoint", train & held, set())
    fails += _check("cv: together they cover everything", len(train | held), 10)
    train = {r["case_id"] for r in F.select(rows, None)}
    held = {r["case_id"] for r in P.eval_rows(rows, "test", None)}
    fails += _check("split: train and test are disjoint", train & held, set())
    return fails, 5


def test_local_judge_pure():
    fails = 0
    fails += _close("p_risk even", L.p_risk_from_logits(1.0, 1.0), 0.5)
    fails += _check("p_risk ordered", L.p_risk_from_logits(0.0, 3.0) > 0.9, True)
    fails += _check("p_risk stable", 0.0 < L.p_risk_from_logits(1000.0, 999.0) < 0.5, True)
    fails += _check("fit keeps short text", L.fit_text([1, 2, 3], 5), ([1, 2, 3], False))
    fails += _check("fit keeps the head", L.fit_text([1, 2, 3, 4, 5], 2), ([1, 2], True))
    fails += _check("criteria kept verbatim",
                    L.system_prompt({"system": "CRITERIA"}).startswith("CRITERIA"), True)
    fails += _check("named arg", E.parse_named("Qwen3-4B QLoRA=p/q.jsonl"),
                    ("Qwen3-4B QLoRA", "p/q.jsonl"))
    return fails, 7


def main() -> int:
    total_f = total_n = 0
    for name, fn in [("metrics", test_metrics),
                     ("danger FN", test_fn_danger),
                     ("failures", test_failures_count_as_wrong),
                     ("run parsing", test_preds_from_run),
                     ("row selection", test_eval_rows_agree),
                     ("local_judge", test_local_judge_pure)]:
        f, n = fn()
        print(f"{name}: {n - f}/{n} passed")
        total_f += f
        total_n += n
    print(f"\ntotal {total_n - total_f}/{total_n} passed")
    return 1 if total_f else 0


if __name__ == "__main__":
    raise SystemExit(main())
