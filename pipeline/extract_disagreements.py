"""등급이 갈린 case만 뽑아 사람 라벨링용 파일로

    python -m pipeline.extract_disagreements results/merged/compare_0829.jsonl \
        --out dataset/to_label_0829.json

팀 가이드라인 §2 — 3사 중 하나라도 판정이 다르면 자동 확정하지 않고 Human Review로 보낸다.
그 Human Review 대상만 여기서 뽑는다.

## 사람이 채우는 칸은 두 개가 아니라 **세 개**다

`expect`(등급)만 받으면 `build_gold.py`가 유형을 채울 곳이 없어 Gold의 `type`이 통째로
빈 문자열이 된다. 251건을 다 라벨링한 뒤 발견하면 유형만 다시 채워야 하므로,
빈 칸을 처음부터 `expect` · `expect_type` · `why` 셋으로 낸다.

## 등급은 같은데 유형만 갈린 case

`build_gold.py` 는 이런 case를 자동 Gold 로 넣되 `type` 을 빈 문자열로 두고
`type_agreed=false` 로 표시한다 (근거 없이 유형 하나를 고르지 않기 위해). 그 결과
**아무도 채우지 않은 빈 유형이 Gold 에 남는다** — Fine-tuning 에서 유형이 학습 대상이면
그만큼이 통째로 빠진다.

§2 의 "하나라도 불일치"를 유형까지로 읽으면 이것도 Human Review 대상이다. 기본값은
기존 동작(등급 기준)을 그대로 두고 `--include-type-split` 로 켠다. 켜지 않아도 몇 건이
그렇게 빠지는지는 **항상 출력한다** — 조용히 사라지면 "전건 처리했다"로 읽힌다.

## 정렬

`risk_spread`(가장 위험하게 본 모델과 가장 안전하게 본 모델의 등급 차)가 큰 것부터.
LOW↔HIGH로 갈린 건이 판단이 가장 크게 엇갈린 곳이고, 사람이 먼저 봐야 할 곳이다.

## reference(과거 Claude 판정)

`merge_results.py --reference` 로 붙여 둔 값이 compare 행에 있으면 라벨링 파일에도
`reference_risk` · `reference_type` 으로 실어 준다. 가이드라인 §5-2·§8 이 과거 판정을
**검수 과정의 참고 자료**로 쓰는 것을 허용한다 (금지하는 것은 모델 *입력* 에 넣는 것).
사람이 판단을 맞춰 갈 값이 아니라 참고값이라는 것이 이름에서 드러나야 한다.
"""

from __future__ import annotations

import argparse
import json
import os

from pipeline.providers.base import PROVIDER_NAMES

# 판정 유형 — server/judge_agent.py 의 출력 스키마와 같은 목록. 라벨러가 오타나
# 새 유형을 만들어 내지 않도록 파일 머리에 적어 둔다
TYPES = ("impersonation", "credentials", "apk", "personal_info", "payment",
         "urgency", "investment", "contentfarm", "unverifiable", "none")


def main() -> int:
    ap = argparse.ArgumentParser(description="3사 판정이 갈린 case를 라벨링 파일로")
    ap.add_argument("compare_jsonl", help="merge_results.py가 만든 compare_{tag}.jsonl")
    ap.add_argument("--out", default="dataset/to_label.json")
    ap.add_argument("--providers", nargs="+", default=list(PROVIDER_NAMES))
    ap.add_argument("--raw", action="store_true",
                    help="후처리 전(verdict_raw) 기준으로 불일치를 판단 — merge_results 와 맞춰라")
    ap.add_argument("--include-type-split", action="store_true",
                    help="등급은 같은데 유형만 갈린 case도 Human Review 로 (기본: 등급 기준만)")
    args = ap.parse_args()

    key = "risk_raw" if args.raw else "risk"
    tkey = "type_raw" if args.raw else "type"

    rows = []
    with open(args.compare_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # merge_results 의 agree_risk 는 "결과 파일이 있는 provider" 기준이라, 한 provider만
    # 돌린 상태에서도 True 가 된다. 여기서는 그 값을 믿지 않고 셀에서 다시 센다
    todo = []
    type_split_skipped: list[str] = []
    for r in rows:
        cells = {p: (r.get(p) or {}) for p in args.providers}
        ok = [p for p in args.providers if cells[p].get("ok")]
        risks = {cells[p].get(key) for p in ok}
        types = {cells[p].get(tkey) for p in ok}
        all_answered = len(ok) == len(args.providers)
        risk_agreed = all_answered and len(risks) == 1
        type_only_split = risk_agreed and len(types) > 1
        # 전원이 답했고 등급이 하나로 모인 case만 자동 확정 대상 — 나머지는 전부 사람에게
        if risk_agreed and not (type_only_split and args.include_type_split):
            if type_only_split:
                type_split_skipped.append(r["case_id"])
            continue
        item = {"case_id": r["case_id"], "url": r.get("url")}
        for p in args.providers:
            c = cells[p]
            item[f"{p}_risk"] = c.get(key) if c.get("ok") else "FAIL"
            item[f"{p}_type"] = c.get(tkey) if c.get("ok") else ""
        item["risk_spread"] = r.get("risk_spread")
        item["missing_providers"] = [p for p in args.providers if p not in ok]
        item["type_only_split"] = type_only_split
        # 과거 Claude 판정 — 검수 참고용이지 맞춰야 할 정답이 아니다 (§5-2)
        ref = r.get("reference") or {}
        if ref:
            item["reference_risk"] = ref.get("risk")
            item["reference_type"] = ref.get("type")
        item["expect"] = ""          # ← 사람이 채운다: LOW · MEDIUM · HIGH
        item["expect_type"] = ""     # ← 사람이 채운다: TYPES 중 하나
        item["why"] = ""             # ← 사람이 채운다: 그렇게 본 이유
        todo.append(item)

    todo.sort(key=lambda r: (r.get("risk_spread") or 0), reverse=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(todo, f, ensure_ascii=False, indent=2)

    print(f"전체 {len(rows)}건 중 Human Review 대상 {len(todo)}건 → {args.out}")
    if todo:
        spread2 = sum(1 for r in todo if r.get("risk_spread") == 2)
        failed = sum(1 for r in todo if r["missing_providers"])
        print(f"  등급 2단계 차이(LOW↔HIGH) {spread2}건 — 최우선")
        if failed:
            # 판정이 갈린 게 아니라 provider가 답을 못 한 case. 라벨링 난이도가 다르므로
            # 섞여 있다는 사실을 먼저 알린다
            print(f"  provider 응답 누락·실패 포함 {failed}건 — 판정 불일치와 원인이 다르다")
        if args.include_type_split:
            n = sum(1 for r in todo if r.get("type_only_split"))
            print(f"  등급은 같고 유형만 갈린 것 {n}건 포함 (--include-type-split)")
    if type_split_skipped:
        # 켜지 않았을 때도 몇 건이 이렇게 빠지는지는 반드시 보인다
        print(f"  ! 등급은 같은데 유형이 갈린 {len(type_split_skipped)}건은 여기에 없다 — "
              f"Gold 의 type 이 빈칸으로 남는다 ({', '.join(type_split_skipped[:5])}"
              f"{' …' if len(type_split_skipped) > 5 else ''})\n"
              f"    유형까지 사람이 정하려면 --include-type-split")
    print(f"\n채울 칸: expect(등급) · expect_type(유형) · why(이유)")
    print(f"  등급 LOW · MEDIUM · HIGH")
    print(f"  유형 {' · '.join(TYPES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
