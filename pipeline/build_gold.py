"""3사 판정 + Human Review → Gold Dataset

    python -m pipeline.build_gold results/merged/compare_0829.jsonl \
        --labeled dataset/to_label_0829.json \
        --out gold/gold_251.jsonl

팀 가이드라인 §2:
 - 3사 판정이 **모두** 일치 → 자동 Gold 확정
 - 하나라도 불일치 → to_label 파일에 사람이 채운 expect 를 Gold 로 사용
 - expect 가 비어 있으면(라벨링 미완료) Gold 에서 제외하고 경고

## "모두 일치"를 compare 파일의 agree_risk 로 판단하면 안 된다

`merge_results.py` 의 `agree_risk` 는 **결과 파일이 있는 provider** 기준이다
(merge_results.py:86 의 `have`). Claude 하나만 돌린 시점에 병합하면 `have = ['claude']`
이므로 전건이 `agree_risk=True` 가 되고, 그대로 믿으면 Claude 단독 판정이 "3사 합의 Gold"
라는 이름을 달게 된다. 가이드라인 §4 가 순차 실행을 허용하므로 실제로 밟기 쉬운 길이다.

그래서 여기서는 agree_risk 를 읽지 않고 provider 셀에서 직접 센다. 그리고 두 가지를
구분한다.

| 상황 | 처리 |
| :-- | :-- |
| compare 파일 **전체**에 3사가 안 들어 있음 (아직 안 돌림) | 즉시 중단 |
| 특정 case 에서만 한 provider 가 실패 (스키마 위반 등) | 그 case 만 Human Review |

## Gold 행에 출처를 남긴다

나중에 Historical Reference(가이드라인 §8)가 준비되거나 확정 규칙이 바뀌었을 때,
251건을 다시 라벨링하지 않고 규칙만 바꿔 재산출할 수 있어야 한다. 그러려면 각 행이
"무엇에 근거해 이 값이 됐는지"를 들고 있어야 한다 — `source` · `agreed_by` ·
`provider_risks` 가 그 기록이다.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from pipeline.providers.base import PROVIDER_NAMES

RISKS = ("LOW", "MEDIUM", "HIGH")


def main() -> int:
    ap = argparse.ArgumentParser(description="3사 합의 + Human Review 로 Gold 확정")
    ap.add_argument("compare_jsonl", help="merge_results.py가 만든 compare_{tag}.jsonl")
    ap.add_argument("--labeled", required=True, help="사람이 expect 를 채운 to_label 파일")
    ap.add_argument("--out", default="gold/gold_251.jsonl")
    ap.add_argument("--providers", nargs="+", default=list(PROVIDER_NAMES))
    ap.add_argument("--raw", action="store_true",
                    help="후처리 전(verdict_raw) 기준 — merge_results·extract_disagreements 와 맞춰라")
    args = ap.parse_args()

    key = "risk_raw" if args.raw else "risk"
    tkey = "type_raw" if args.raw else "type"

    rows = []
    with open(args.compare_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{args.compare_jsonl} 이 비어 있다")

    # ── 3사가 실제로 다 돌았는지 먼저 본다 ──────────────────────────────────
    answered = {p: sum(1 for r in rows if (r.get(p) or {}).get("ok")) for p in args.providers}
    missing = [p for p, n in answered.items() if n == 0]
    if missing:
        raise SystemExit(
            f"compare 파일에 {', '.join(missing)} 판정이 한 건도 없다 — 아직 돌리지 않았다.\n"
            f"  현재: {answered}\n"
            f"  이대로 Gold 를 만들면 {len(args.providers) - len(missing)}개 모델의 판정이"
            f" '3사 합의'라는 이름을 달게 된다 (팀 가이드라인 §2 위반).\n"
            f"  남은 provider 를 돌린 뒤 merge_results.py 를 다시 실행하라.")

    with open(args.labeled, encoding="utf-8") as f:
        labeled = {item["case_id"]: item for item in json.load(f)}

    gold: list[dict] = []
    skipped: list[str] = []
    bad_label: list[str] = []
    auto = human = 0
    audit: list[str] = []
    type_split = 0
    human_type: list[str] = []

    for r in rows:
        cid = r["case_id"]
        cells = {p: (r.get(p) or {}) for p in args.providers}
        ok = [p for p in args.providers if cells[p].get("ok")]
        risks = {cells[p].get(key) for p in ok}
        types = {cells[p].get(tkey) for p in ok}
        provider_risks = {p: (cells[p].get(key) if cells[p].get("ok") else "FAIL")
                          for p in args.providers}

        base = {"case_id": cid, "url": r.get("url"),
                "provider_risks": provider_risks}

        if len(ok) == len(args.providers) and len(risks) == 1:
            risk = risks.pop()
            # 유형은 등급과 따로 센다. 등급이 같아도 유형은 갈릴 수 있고, 그때 유형을
            # 임의로 하나 고르면 근거 없는 라벨이 된다 — 비워 두고 표시만 남긴다
            agreed_type = types.pop() if len(types) == 1 else ""
            source = "auto_agree"
            if not agreed_type:
                # 유형만 갈린 case 도 `extract_disagreements.py --include-type-split` 로
                # 라벨링에 실을 수 있다. 그렇게 사람이 채워 둔 값이 있으면 그것을 쓴다 —
                # 여기서 안 보면 사람이 채운 유형이 조용히 버려지고 type 이 빈칸으로 남는다.
                # 등급은 어디까지나 3사 합의로 정해진 것이라 source 에 둘 다 드러낸다
                item = labeled.get(cid)
                labeled_type = (item or {}).get("expect_type", "").strip()
                if labeled_type:
                    agreed_type = labeled_type
                    source = "auto_agree+human_type"
                    human_type.append(cid)
                else:
                    type_split += 1
            gold.append({**base, "risk": risk, "type": agreed_type,
                         "type_agreed": source == "auto_agree" and bool(agreed_type),
                         "source": source,
                         "agreed_by": list(args.providers),
                         # 가이드라인에 없는 항목이지만 계획서 §8-3 의 경고 지점이다 —
                         # 세 모델이 같은 이유로 함께 틀릴 수 있어 HIGH 합의는 표본 검수 대상
                         "needs_audit": risk == "HIGH"})
            auto += 1
            if risk == "HIGH":
                audit.append(cid)
            continue

        item = labeled.get(cid)
        if not item or not item.get("expect"):
            skipped.append(cid)
            continue
        expect = str(item["expect"]).strip().upper()
        if expect not in RISKS:
            # 오타를 조용히 Gold 에 넣으면 채점이 통째로 어긋난다
            bad_label.append(f"{cid}: expect={item['expect']!r}")
            continue
        gold.append({**base, "risk": expect,
                     "type": (item.get("expect_type") or "").strip(),
                     "type_agreed": bool((item.get("expect_type") or "").strip()),
                     "source": "human_review",
                     "agreed_by": [],
                     "why": item.get("why", ""),
                     "needs_audit": False})
        human += 1

    if bad_label:
        raise SystemExit(
            f"라벨 값이 잘못된 case {len(bad_label)}건 — 허용값은 {RISKS}\n  "
            + "\n  ".join(bad_label[:10]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print(f"provider별 판정 건수: {answered}")
    print(f"Gold {len(gold)}건 → {args.out}  (자동확정 {auto} · 사람검수 {human})")
    print(f"등급 분포: {dict(Counter(g['risk'] for g in gold))}")
    if human_type:
        print(f"  등급은 3사 합의 · 유형만 사람이 채운 case {len(human_type)}건 "
              f"(source=auto_agree+human_type): {', '.join(human_type[:10])}"
              + (" …" if len(human_type) > 10 else ""))
    if type_split:
        print(f"  등급은 합의했으나 유형이 갈린 case {type_split}건 — type 을 비워 두었다")
    if audit:
        print(f"  HIGH 자동확정 {len(audit)}건 — 표본 검수 권장 (needs_audit=true): "
              f"{', '.join(audit[:10])}" + (" …" if len(audit) > 10 else ""))
    if skipped:
        print(f"라벨 미완료로 제외된 case {len(skipped)}건: {skipped[:10]}"
              + (" …" if len(skipped) > 10 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
