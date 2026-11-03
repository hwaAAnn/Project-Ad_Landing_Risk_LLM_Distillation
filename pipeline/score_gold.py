"""Gold 기준 3사 채점

    python -m pipeline.score_gold results/merged/compare_0830.jsonl

`gold/gold_251.jsonl` 을 정답으로 놓고 provider 별 정확도를 낸다. 예측은 compare
파일의 provider 셀에서 읽는다 — 어느 run 을 쓸지 고르는 규칙(최신 우선·성공 우선)이
`merge_results.py:58` 에 이미 있고, 두 군데 두면 언젠가 어긋난다.

## 왜 구간을 나누는가

Gold 251건 중 220건은 3사 합의로 자동 확정된 것이다. 그 구간을 정답으로 놓고 채점하면
세 모델 모두 **정의상** 100% 가 나온다. 전체 정확도만 내면 이 순환이 숫자에 가려지므로
전체 / 3사 합의 / 사람 확정 세 구간으로 나누어 낸다. 실제 변별력은 사람 확정 구간에서만
나온다.

## 등급과 유형은 구간이 다르다

`auto_agree+human_type` (case_131 · case_204) 은 등급이 3사 합의, 유형이 사람 확정이다.
하나의 구간 분류를 둘에 같이 쓰면 사람이 정한 유형이 순환 구간에 섞여 100% 판정을 받는다.
그래서 `risk_bucket` 과 `type_bucket` 을 따로 둔다.

| | consensus | human |
| :-- | --: | --: |
| 등급 | auto_agree + auto_agree+human_type | human_review |
| 유형 | auto_agree | human_review + auto_agree+human_type |
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from pipeline.providers.base import PROVIDER_NAMES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root

# merge_results.py:37 과 같은 값. 등급은 순서가 있어 "틀린 방향"을 판정할 수 있다
RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

CONSENSUS = "consensus"
HUMAN = "human"


def risk_bucket(source: str) -> str:
    """등급의 출처. auto_agree+human_type 은 **등급만은** 3사 합의로 정해졌다"""
    return HUMAN if source == "human_review" else CONSENSUS


def type_bucket(source: str) -> str:
    """유형의 출처. auto_agree+human_type 은 유형을 사람이 채웠다 — 순환이 아니다"""
    return CONSENSUS if source == "auto_agree" else HUMAN


def load_jsonl(path: str) -> dict[str, dict]:
    """jsonl 을 case_id → 행 으로. 같은 case_id 가 두 번 나오면 뒤엣것이 이긴다"""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["case_id"]] = r
    return out


def join(gold: dict[str, dict], compare: dict[str, dict]) -> tuple[list[str], list[str]]:
    """(양쪽에 다 있는 case_id, gold 에만 있는 case_id)

    gold 에만 있는 case 를 오답으로 세면 "예측이 없다"와 "틀렸다"가 한 숫자에 섞인다.
    빼고 건수만 따로 보고한다
    """
    both = sorted(cid for cid in gold if cid in compare)
    gold_only = sorted(cid for cid in gold if cid not in compare)
    return both, gold_only


def direction(gold_risk: str, pred_risk: str) -> str | None:
    """틀린 방향. 맞으면 None

    등급은 순서가 있어 단순 정답/오답보다 방향이 중요하다. Gold 가 HIGH 인데 LOW 로
    본 것(미탐)과 Gold 가 LOW 인데 HIGH 로 본 것(과탐)은 서비스에서 의미가 다르다
    """
    if gold_risk == pred_risk:
        return None
    return "under" if RISK_ORDER[pred_risk] < RISK_ORDER[gold_risk] else "over"


def _pred(compare_row: dict, provider: str, raw: bool) -> dict | None:
    """provider 셀에서 예측을 꺼낸다. 실패했으면 None — 오답이 아니라 '채점 불가'다"""
    cell = compare_row.get(provider) or {}
    if not cell.get("ok"):
        return None
    return {"risk": cell.get("risk_raw" if raw else "risk"),
            "type": cell.get("type_raw" if raw else "type")}


def score_risk(gold: dict[str, dict], compare: dict[str, dict], cids: list[str],
               providers: list[str], raw: bool = False) -> dict[str, dict]:
    """provider → 등급 채점 결과

    실패한 provider 는 skipped 로 따로 센다. 오답으로 세면 모델의 판단력과 API 실패가
    한 숫자에 섞인다
    """
    out: dict[str, dict] = {}
    for p in providers:
        s = {"n": 0, "hit": 0, "under": 0, "over": 0, "skipped": 0, "confusion": {}}
        for cid in cids:
            pred = _pred(compare[cid], p, raw)
            if pred is None or not pred["risk"]:
                s["skipped"] += 1
                continue
            g = gold[cid]["risk"]
            s["n"] += 1
            key = (g, pred["risk"])
            s["confusion"][key] = s["confusion"].get(key, 0) + 1
            d = direction(g, pred["risk"])
            if d is None:
                s["hit"] += 1
            else:
                s[d] += 1
        out[p] = s
    return out


def score_type(gold: dict[str, dict], compare: dict[str, dict], cids: list[str],
               providers: list[str], raw: bool = False) -> dict[str, dict]:
    """provider → 유형 채점 결과. 틀린 (정답, 예측) 쌍을 세어 어디서 갈리는지 본다

    Gold 의 type 이 비어 있는 행은 채점하지 않는다 — 정답이 없는 것을 틀렸다고 할 수
    없다. 지금 251건에는 빈 type 이 없지만, 확정 규칙이 바뀌면 다시 생길 수 있다
    """
    out: dict[str, dict] = {}
    for p in providers:
        s = {"n": 0, "hit": 0, "skipped": 0, "pairs": {}}
        for cid in cids:
            g = gold[cid].get("type") or ""
            pred = _pred(compare[cid], p, raw)
            if not g or pred is None or not pred["type"]:
                s["skipped"] += 1
                continue
            s["n"] += 1
            if pred["type"] == g:
                s["hit"] += 1
            else:
                key = (g, pred["type"])
                s["pairs"][key] = s["pairs"].get(key, 0) + 1
        out[p] = s
    return out


def verify_consensus(risk_scores: dict[str, dict], raw: bool) -> list[str]:
    """3사 합의 구간에서 전원 100% 인지 본다. 어긋난 provider 설명 목록을 돌려준다

    이 구간의 정답은 세 모델이 합의한 값 그 자체다. 100% 가 아니라면 모델이 틀린 게
    아니라 **채점이 틀린** 것이다 — 조인이 어긋났거나, compare 파일이 Gold 를 만든 것과
    다르거나, 후처리 기준이 다르다. 어느 쪽이든 나머지 숫자를 믿을 수 없으므로
    조용히 이상한 값을 내놓는 것보다 멈추는 쪽이 낫다.

    `--raw` 를 준 때는 기준이 다른 것이 의도된 것이므로 검사하지 않는다. Gold 는
    후처리 후 값으로 만들어졌기 때문에 raw 기준에서는 어긋나는 게 정상이다
    """
    if raw:
        return []
    bad = []
    for p, s in risk_scores.items():
        if s["n"] and s["hit"] != s["n"]:
            bad.append(f"{p}: 3사 합의 구간 {s['hit']}/{s['n']} — 100% 여야 한다")
    return bad


def consensus_gate(bad: list[str], experiment: bool) -> tuple[str, str] | None:
    """검산 어긋남을 치명(중단)으로 볼지 경고로 볼지. 어긋남이 없으면 None

    기본은 치명이다 — 3사 합의 구간에서 100% 가 아니면 채점이 틀린 것이다.

    `--experiment` 는 그 전제가 성립하지 않는 실행을 위한 것이다. 프롬프트를 바꿔
    다시 돌린 결과를 지금 Gold 로 채점하면, 합의 구간에서 판정이 달라지는 것이 오히려
    **실험의 목적**이다. 여기서 중단하면 실험 자체를 채점할 수 없다. 대신 경고를 남겨
    "이 숫자는 Gold 를 만든 판정이 아니다"가 출력에 드러나게 한다
    """
    if not bad:
        return None
    if experiment:
        return ("warn", "이 compare 는 Gold 를 만든 판정이 아니다 (--experiment) — "
                        "합의 구간이 100% 가 아닌 것은 예상된 것이다")
    return ("fatal", "검산 실패 — 3사 합의 구간은 정의상 전원 100% 여야 한다")


def error_rows(gold: dict[str, dict], compare: dict[str, dict], cids: list[str],
               providers: list[str], raw: bool = False) -> list[dict]:
    """한 provider 라도 틀린 case 만. 숫자에서 바로 "어떤 case 에서 틀렸는데"로 넘어가려는 것

    틀린 모델만 싣지 않고 **세 provider 를 다** 싣는다 — 나란히 놓고 봐야 왜 갈렸는지
    보인다. `why` 는 사람이 라벨링 때 적은 근거로, human_review 행에만 있다
    """
    rows = []
    for cid in cids:
        g = gold[cid]
        cells, wrong = {}, False
        for p in providers:
            pred = _pred(compare[cid], p, raw)
            if pred is None:
                cells[f"{p}_risk"] = cells[f"{p}_type"] = ""
                cells[f"{p}_direction"] = "실패"
                continue
            d = direction(g["risk"], pred["risk"]) if pred["risk"] else None
            cells[f"{p}_risk"] = pred["risk"] or ""
            cells[f"{p}_type"] = pred["type"] or ""
            cells[f"{p}_direction"] = d or ""
            if d or (g.get("type") and pred["type"] and pred["type"] != g["type"]):
                wrong = True
        if not wrong:
            continue
        rows.append({"case_id": cid, "url": compare[cid].get("url") or g.get("url") or "",
                     "gold_risk": g["risk"], "gold_type": g.get("type", ""),
                     "gold_source": g.get("source", ""), "why": g.get("why", ""),
                     **cells})
    return rows


def _tag_from(compare_path: str) -> str:
    """`compare_0830.jsonl` → `0830`. 채점 결과가 어느 병합본에서 나왔는지 이름에 남긴다"""
    stem = os.path.basename(compare_path)
    for cut in (".jsonl", ".json"):
        stem = stem[: -len(cut)] if stem.endswith(cut) else stem
    return stem[len("compare_"):] if stem.startswith("compare_") else stem or "latest"


def _pct(hit: int, n: int) -> str:
    return f"{hit}/{n} ({hit / n * 100:5.1f}%)" if n else f"{hit}/0 (  —  )"


def _print_risk(title: str, note: str, scores: dict[str, dict], providers: list[str]) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))
    if note:
        print(f"   {note}")
    for p in providers:
        s = scores[p]
        line = f"  {p:8s} 정확도 {_pct(s['hit'], s['n'])}"
        if s["n"]:
            line += f" · 미탐 {s['under']} · 과탐 {s['over']}"
        if s["skipped"]:
            line += f" · 채점제외 {s['skipped']}"
        print(line)
        if not s["n"]:
            continue
        order = [r for r in RISK_ORDER if any(k[0] == r for k in s["confusion"])]
        if order:
            print("             gold\\pred " + "".join(f"{r:>8s}" for r in RISK_ORDER))
            for gr in order:
                cells = "".join(f"{s['confusion'].get((gr, pr), 0):8d}" for pr in RISK_ORDER)
                print(f"             {gr:9s}" + cells)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gold 기준 3사 채점")
    ap.add_argument("compare_jsonl", help="merge_results.py 가 만든 compare_{tag}.jsonl")
    ap.add_argument("--gold", default=os.path.join(HERE, "gold", "gold_251.jsonl"))
    ap.add_argument("--tag", default=None, help="기본은 compare 파일 이름에서 따온다")
    ap.add_argument("--providers", nargs="+", default=list(PROVIDER_NAMES))
    ap.add_argument("--raw", action="store_true",
                    help="후처리 전(risk_raw) 기준. Gold 는 후처리 후 값으로 만들어졌으므로 "
                         "기준이 어긋난다 — 검산을 끄고 경고만 낸다")
    ap.add_argument("--experiment", action="store_true",
                    help="Gold 를 만든 판정이 아닌 실행(프롬프트 변경 등)을 채점한다. "
                         "3사 합의 구간이 100%% 가 아닌 것을 중단이 아니라 경고로 다룬다")
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    compare = load_jsonl(args.compare_jsonl)
    cids, gold_only = join(gold, compare)
    if not cids:
        raise SystemExit(f"조인된 case 가 없다 — {args.gold} 와 {args.compare_jsonl} 의 "
                         "case_id 가 서로 다른 dataset 에서 나온 것 같다")
    tag = args.tag or _tag_from(args.compare_jsonl)

    risk_c = [c for c in cids if risk_bucket(gold[c].get("source", "")) == CONSENSUS]
    risk_h = [c for c in cids if risk_bucket(gold[c].get("source", "")) == HUMAN]
    type_c = [c for c in cids if type_bucket(gold[c].get("source", "")) == CONSENSUS]
    type_h = [c for c in cids if type_bucket(gold[c].get("source", "")) == HUMAN]

    print(f"gold {args.gold} — {len(gold)}건")
    print(f"예측 {args.compare_jsonl} — 조인 {len(cids)}건"
          + (f" (gold 에만 있는 case {len(gold_only)}건은 채점에서 제외)" if gold_only else ""))
    print(f"기준: {'후처리 전(raw)' if args.raw else '후처리 후'} · 이름표 {tag}")
    if args.raw:
        print("  ! --raw 는 Gold(후처리 후)와 기준이 다르다 — 검산을 끄고 낸다")

    s_all = score_risk(gold, compare, cids, args.providers, args.raw)
    s_con = score_risk(gold, compare, risk_c, args.providers, args.raw)
    s_hum = score_risk(gold, compare, risk_h, args.providers, args.raw)

    gate = consensus_gate(verify_consensus(s_con, args.raw), args.experiment)
    if gate and gate[0] == "fatal":
        raise SystemExit(
            gate[1] + ".\n  "
            + "\n  ".join(verify_consensus(s_con, args.raw))
            + "\n  조인이 어긋났거나, 이 compare 파일이 Gold 를 만든 것과 다르다.\n"
              "  build_gold.py 에 넣었던 compare 파일과 같은 것을 주고 있는지 확인하라.\n"
              "  프롬프트를 바꿔 다시 돌린 결과를 채점하는 것이라면 --experiment 를 줘라.")
    if gate:
        print(f"  ! {gate[1]}")

    print("\n" + "=" * 62)
    print(f"등급 (risk) — 3사 합의 {len(risk_c)}건 · 사람 확정 {len(risk_h)}건")
    print("=" * 62)
    _print_risk(f"전체 {len(cids)}건", "순환 포함 — Gold 의 대부분이 3사 합의라 참고용이다",
                s_all, args.providers)
    _print_risk(f"3사 합의 {len(risk_c)}건", "정의상 전원 100% (검산 통과)",
                s_con, args.providers)
    _print_risk(f"사람 확정 {len(risk_h)}건", "★ 순환이 없는 유일한 구간 — 실제 변별력",
                s_hum, args.providers)

    t_con = score_type(gold, compare, type_c, args.providers, args.raw)
    t_hum = score_type(gold, compare, type_h, args.providers, args.raw)
    print("\n" + "=" * 62)
    print(f"유형 (type) — 3사 합의 {len(type_c)}건 · 사람 확정 {len(type_h)}건")
    print("  등급과 구간이 다르다 — auto_agree+human_type 행은 유형을 사람이 채웠다")
    print("=" * 62)
    for label, sc in (("3사 합의", t_con), ("사람 확정", t_hum)):
        print(f"\n── {label} " + "─" * 50)
        for p in args.providers:
            s = sc[p]
            print(f"  {p:8s} 정확도 {_pct(s['hit'], s['n'])}"
                  + (f" · 채점제외 {s['skipped']}" if s["skipped"] else ""))
            top = sorted(s["pairs"].items(), key=lambda kv: -kv[1])[:3]
            for (gt, pt), c in top:
                print(f"             {gt} → {pt}  {c}건")

    rows = error_rows(gold, compare, cids, args.providers, args.raw)
    out_dir = os.path.join(HERE, "results", "scored")
    os.makedirs(out_dir, exist_ok=True)
    cpath = os.path.join(out_dir, f"errors_{tag}.csv")
    cols = (["case_id", "url", "gold_risk", "gold_type", "gold_source", "why"]
            + [f"{p}_{k}" for p in args.providers for k in ("risk", "type", "direction")])
    with open(cpath, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n한 provider 라도 틀린 case {len(rows)}건 → {cpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
