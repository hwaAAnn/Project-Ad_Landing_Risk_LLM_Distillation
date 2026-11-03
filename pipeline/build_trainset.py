"""Gold 라벨 + 조립 프롬프트 → fine-tuned classifier 학습셋

    python -m pipeline.build_trainset --tag 0831

`gold/gold_251.jsonl` 의 등급을 2클래스로 묶어 라벨로 삼고, `build_prompt.py` 가
조립한 user 프롬프트를 입력 텍스트로 쓴다. 3사에게 실제로 보낸 것과 **문자 단위로 같은
입력**이라야 `score_gold.py` 의 같은 표에서 비교해도 공정하다.

## 왜 2클래스인가

Gold 251건의 분포가 이 결정을 강제한다 — HIGH 1건, impersonation 1건, urgency 2건.
train/test 로 나누면 한쪽에만 들어가거나 아예 빠져서 학습도 평가도 안 된다. 등급을
LOW(0) 대 MEDIUM·HIGH(1) 로 묶으면 129 대 122 로 균형이 맞는다. 원본 risk·type·source 는
각 행에 그대로 실어, 데이터가 늘어 3클래스로 갈 때 **다시 조립하지 않고 라벨만 바꿔**
재산출할 수 있게 한다.

## 왜 sha 를 검증하는가

계산해서 남기기만 하면 아무도 안 본다. 조립 규칙이 바뀌어도 모른 채 다른 입력으로
학습하게 되므로, `results/*/*.jsonl` 의 기록과 **전건 대조해 하나라도 어긋나면 중단**한다.
조립 파라미터(input_mode·excluded_tools)도 하드코딩하지 않고 참조 run 에서 읽는다 —
설계 중 input_mode 를 'oneshot' 으로 잘못 쓴 실수를 실제로 밟았고, 이 데이터에서는
우연히 같은 결과가 나왔지만 하드코딩했다면 조용히 다른 조립을 썼을 것이다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter

from pipeline import build_prompt as B
from pipeline import schema as S

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root

# 등급 2클래스. HIGH 가 1건뿐이라 MEDIUM 과 묶는다 — 자세한 이유는 위 docstring
RISK_TO_LABEL = {"LOW": 0, "MEDIUM": 1, "HIGH": 1}


def label_of(risk: str) -> int:
    """등급 → 2클래스 라벨. 모르는 등급은 조용히 넘기지 않는다"""
    if risk not in RISK_TO_LABEL:
        raise SystemExit(f"모르는 등급: {risk!r} (가능: {', '.join(RISK_TO_LABEL)})")
    return RISK_TO_LABEL[risk]


def collect_reference(results_dir: str) -> tuple[dict[str, dict], list[str]]:
    """`results/*/*.jsonl` 에서 case_id → {"sha", "input_mode", "exclude"} 와 충돌 목록

    **세 필드를 다 가진 행만 참조가 된다.** dryrun 행은 prompt_sha256 은 있어도
    input_mode·excluded_tools 를 기록하지 않아 조립 파라미터를 줄 수 없고, merged 의
    compare 행은 prompt_sha256 자체가 없다. 참조는 "어떤 조립으로 무엇이 나왔는지"를
    둘 다 말해 줄 수 있어야 한다
    """
    ref: dict[str, dict] = {}
    conflicts: list[str] = []
    if not os.path.isdir(results_dir):
        return ref, conflicts
    for sub in sorted(os.listdir(results_dir)):
        d = os.path.join(results_dir, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    sha = r.get("prompt_sha256")
                    mode = r.get("input_mode")
                    exc = r.get("excluded_tools")
                    if not (sha and mode and exc is not None):
                        continue
                    entry = {"sha": sha, "input_mode": mode, "exclude": tuple(exc)}
                    cid = r["case_id"]
                    prev = ref.get(cid)
                    if prev is None:
                        ref[cid] = entry
                    elif prev != entry:
                        conflicts.append(
                            f"{cid}: 참조가 서로 다르다 — {prev} vs {entry} ({sub}/{fn})")
    return ref, conflicts


def assembly_params(reference: dict[str, dict]) -> tuple[str, tuple[str, ...]]:
    """참조에서 (input_mode, exclude) 를 도출한다. 하나로 정해지지 않으면 중단

    이 값을 스크립트에 박아 두지 않는 이유는, 박아 두면 실제 실행과 다른 조립을 써도
    아무도 모르기 때문이다. 결과 행에 이미 기록돼 있으므로 그것이 정본이다
    """
    if not reference:
        raise SystemExit(
            "참조할 run 결과가 없다 — results/<provider>/*.jsonl 이 있어야 한다.\n"
            "  학습셋 입력이 3사에게 보낸 것과 같은지 확인할 방법이 없으므로 만들지 않는다.")
    combos = {(e["input_mode"], e["exclude"]) for e in reference.values()}
    if len(combos) > 1:
        raise SystemExit(
            "참조 run 들의 조립 파라미터가 서로 다르다 — 어느 쪽으로 조립할지 정할 수 없다.\n  "
            + "\n  ".join(str(c) for c in sorted(combos, key=str))
            + "\n  같은 조립으로 돈 run 만 남기고 다시 실행하라.")
    return combos.pop()


def assemble(cases: list[dict], spec: dict, input_mode: str,
             exclude: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    """case_id → (조립된 user 프롬프트, 그 sha256)

    `run_eval.py` 와 같은 함수를 같은 파라미터로 부른다. 조립이 실패하는 case 는
    조용히 빠뜨리지 않고 중단한다 — 251건 중 몇 건이 사라지면 학습셋 건수가 어긋난다
    """
    out: dict[str, tuple[str, str]] = {}
    for case in cases:
        cid = case["case_id"]
        try:
            _system, user = B.build_messages(case, spec, input_mode=input_mode,
                                             exclude=exclude)
        except Exception as e:
            raise SystemExit(f"{cid} 조립 실패: {type(e).__name__}: {e}")
        out[cid] = (user, hashlib.sha256(user.encode("utf-8")).hexdigest())
    return out


def verify(assembled: dict[str, tuple[str, str]],
           reference: dict[str, dict]) -> list[str]:
    """조립 결과가 3사에게 보낸 것과 같은지 전건 대조. 어긋남 설명 목록을 돌려준다

    참조에 없는 case 도 어긋남으로 센다. 그 case 는 3사가 판정한 적이 없다는 뜻이고,
    그러면 "3사와 같은 입력"이라는 전제 자체가 깨진다 — 조용히 넘어가면 학습셋 일부만
    검증되지 않은 조립으로 들어간다
    """
    bad = []
    for cid in sorted(assembled):
        _text, sha = assembled[cid]
        entry = reference.get(cid)
        if entry is None:
            bad.append(f"{cid}: 참조 run 이 없다 (3사가 판정한 적이 없는 case)")
        elif entry["sha"] != sha:
            bad.append(f"{cid}: sha 불일치 — 기록 {entry['sha'][:16]}… vs 지금 {sha[:16]}…")
    return bad


def stratified_assign(items: list[tuple[str, str]], test_ratio: float,
                      folds: int, seed: int) -> dict[str, tuple[str, int]]:
    """(case_id, 계층키) 목록 → case_id → (split, fold)

    계층은 `label|source` 다. source 를 계층에 넣는 이유는 auto_agree 와 human_review 가
    성격이 다른 데이터이기 때문이다 — 한쪽에 몰리면 구간별 점수를 낼 수 없다.

    **split 과 fold 는 서로 독립이다.** fold 는 split 과 무관하게 전건에 배정한다
    (test 행도 fold 를 가진다). 둘은 다른 평가 방식이라 섞으면 안 된다 — "train 중에서
    fold 나누기"를 하면 CV 가 전체를 덮지 못해 human_review 를 전부 평가한다는 목적이
    깨진다
    """
    groups: dict[str, list[str]] = {}
    for cid, key in items:
        groups.setdefault(key, []).append(cid)

    out: dict[str, tuple[str, int]] = {}
    for key in sorted(groups):
        ids = sorted(groups[key])            # 입력 순서에 의존하지 않게 먼저 정렬
        random.Random(f"{seed}|{key}").shuffle(ids)
        n_test = round(len(ids) * test_ratio)
        for i, cid in enumerate(ids):
            split = "test" if i < n_test else "train"
            out[cid] = (split, i % folds)    # fold 는 계층 안에서 돌아가며 — 고르게 퍼진다
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Gold 라벨 + 조립 프롬프트 → 학습셋")
    ap.add_argument("--tag", required=True, help="산출 파일 이름표 (trainset_{tag}.jsonl)")
    ap.add_argument("--gold", default=os.path.join(HERE, "gold", "gold_251.jsonl"))
    ap.add_argument("--cases", default=os.path.join(HERE, "dataset",
                                                    "cases_251_20260829.jsonl"))
    ap.add_argument("--spec", default=os.path.join(HERE, "dataset", "prompt_spec.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"),
                    help="참조 sha 를 모을 곳")
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gold = {}
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                gold[r["case_id"]] = r

    reference, conflicts = collect_reference(args.results)
    if conflicts:
        raise SystemExit(
            f"참조 run 의 prompt_sha256 이 서로 어긋난다 ({len(conflicts)}건).\n  "
            + "\n  ".join(conflicts[:10])
            + "\n  같은 조립으로 돈 run 만 남기고 다시 실행하라.")
    input_mode, exclude = assembly_params(reference)
    print(f"참조 run: {len(reference)}건 · 조립 {input_mode} · 제외 {list(exclude) or '없음'}")

    cases = S.load_cases(args.cases, input_mode)
    spec = B.load_spec(args.spec)
    assembled = assemble(cases, spec, input_mode, exclude)

    bad = verify(assembled, reference)
    if bad:
        raise SystemExit(
            f"검증 실패 — 조립 결과가 3사에게 보낸 것과 다르다 ({len(bad)}건).\n  "
            + "\n  ".join(bad[:10])
            + "\n\n  원인 후보는 셋뿐이다:\n"
              "   1. 조립 규칙이 바뀌었다        → build_prompt.py\n"
              "   2. 판정 기준의 assembly 가 바뀌었다 → dataset/prompt_spec.json 의 assembly\n"
              "   3. 다른 cases 파일을 주고 있다  → --cases 인자")
    print(f"sha 검증 통과: {len(assembled)}건 전부 기록과 일치")

    # gold 에 없는 case 는 라벨이 없으므로 학습셋에 넣을 수 없다
    missing = [c for c in assembled if c not in gold]
    if missing:
        raise SystemExit(f"gold 에 라벨이 없는 case {len(missing)}건: {missing[:10]}")

    items = [(cid, f"{label_of(gold[cid]['risk'])}|{gold[cid].get('source', '')}")
             for cid in sorted(assembled)]
    assign = stratified_assign(items, args.test_ratio, args.folds, args.seed)

    rows = []
    for cid in sorted(assembled):
        text, sha = assembled[cid]
        g = gold[cid]
        split, fold = assign[cid]
        rows.append({"case_id": cid, "text": text, "label": label_of(g["risk"]),
                     "risk": g["risk"], "type": g.get("type", ""),
                     "source": g.get("source", ""),
                     "split": split, "fold": fold, "prompt_sha256": sha})

    out_path = os.path.join(HERE, "dataset", f"trainset_{args.tag}.jsonl")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n학습셋 {len(rows)}건 → {out_path}")
    print(f"라벨: {dict(Counter(r['label'] for r in rows))}  (0=LOW · 1=MEDIUM·HIGH)")
    print("label × source")
    cross = Counter((r["label"], r["source"]) for r in rows)
    for (lab, src), n in sorted(cross.items()):
        print(f"  label={lab}  {src:22s} {n:4d}")
    print(f"split: {dict(Counter(r['split'] for r in rows))}")
    print(f"fold : {dict(sorted(Counter(r['fold'] for r in rows).items()))}")

    n_auto = sum(1 for r in rows if r["source"].startswith("auto_agree"))
    print(f"\n  ! 이 학습셋의 라벨 {n_auto}/{len(rows)}건은 3사 합의로 만들어졌다.")
    print("    이것으로 학습한 분류기는 상당 부분 3사를 모방하도록 배운다 — 지식 증류에")
    print("    가깝지 사람이 정한 정답을 배우는 것이 아니다. 순환이 없는 신호는")
    print(f"    human_review {sum(1 for r in rows if r['source'] == 'human_review')}건뿐이다.")
    print("    평가할 때 source 로 갈라 구간별 점수를 따로 내라 (score_gold.py 와 같은 방식).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
