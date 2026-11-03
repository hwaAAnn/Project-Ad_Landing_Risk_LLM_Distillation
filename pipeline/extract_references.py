"""Fixed Dataset 안에 남아 있는 **과거 Claude 판정**을 사후 비교용으로 분리해 낸다

    python -m pipeline.extract_references
    python -m pipeline.extract_references --cases dataset/cases_251_20260829.jsonl \
        --out references/historical_claude_251.jsonl

팀 가이드라인 §5-2 · §8 을 그대로 옮긴 것이다.

수집 당시의 `tool_outputs` 에는 `lookup_cache` 도구의 출력이 들어 있고, 그 안의
`previous` 에 과거 Claude 판정(risk·type·reason·at)이 남아 있는 case가 있다.
모델 비교에서는 이것을 **입력에서 제외**하지만(`run_eval.py` 의 DEFAULT_EXCLUDE),
검수 단계에서 "예전에는 뭐라고 봤나"를 대조할 때는 필요하다.

## 이 스크립트가 지키는 선

 1. **원본 Dataset은 읽기만 한다** (§5-2 규칙 1). 출력은 `references/` 로만 나간다.
 2. 뽑아낸 판정은 Gold 가 아니다. `source` 를 `lookup_cache.previous` 로 박아 두어
    3사 신규 판정(Provider Results)·확정 정답(Gold)과 섞이지 않게 한다.
 3. `previous` 가 `null` 인 case는 줄을 만들지 않는다 — "과거 판정이 LOW였다"와
    "과거 판정이 없다"는 다른 것이라 빈 줄로 채우면 나중에 구분이 안 된다.

## 왜 모델 입력 파일에 도로 합치지 않는가

§3 이 금지한다. `cases_251_with_ref.jsonl` 같은 합본을 만들어 판정 입력으로 쓰면
다른 provider가 과거 Claude 판정을 힌트로 보게 되어 독립 비교가 깨진다.
이 파일은 판정이 **끝난 뒤** case_id 로 조인해서 쓴다.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
DEFAULT_CASES = os.path.join(HERE, "dataset", "cases_251_20260829.jsonl")
DEFAULT_OUT = os.path.join(HERE, "references", "historical_claude_251.jsonl")
VERDICT_FIELDS = ("risk", "type", "reason", "advice", "evidence", "at")


def _previous_of(case: dict) -> dict | None:
    """case 하나에서 lookup_cache.previous 를 꺼낸다. 없으면 None

    `output` 은 JSON **문자열**로 저장돼 있다 (도구가 돌려준 그대로). 파싱이 안 되는
    행은 조용히 넘기지 않고 예외로 올린다 — 데이터가 상한 것을 못 보고 지나가면
    "과거 판정 없음"으로 잘못 집계된다
    """
    for tool in case.get("input", {}).get("tool_outputs", []):
        if tool.get("tool") != "lookup_cache":
            continue
        raw = tool.get("output")
        if not raw:
            continue
        payload = json.loads(raw) if isinstance(raw, str) else raw
        prev = payload.get("previous")
        if prev:
            return {"site": payload.get("site"), **prev}
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fixed Dataset의 lookup_cache.previous → references/ (사후 비교용)")
    ap.add_argument("--cases", default=DEFAULT_CASES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    with open(args.cases, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    rows = []
    for case in cases:
        prev = _previous_of(case)
        if prev is None:
            continue
        rows.append({
            "case_id": case["case_id"],
            "url": case.get("url"),
            "site": prev.get("site"),
            "reference": {k: prev.get(k) for k in VERDICT_FIELDS if prev.get(k) is not None},
            # 이 값이 어디서 나왔는지 — Gold·Provider Result 와 섞이면 안 된다
            "source": "lookup_cache.previous",
            "cases_file": os.path.basename(args.cases),
        })

    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"case {len(cases)}건 중 과거 판정이 있는 것 {len(rows)}건 "
          f"(없음 {len(cases) - len(rows)}건)")
    dist = Counter(r["reference"].get("risk") for r in rows)
    tdist = Counter(r["reference"].get("type") for r in rows)
    print(f"  등급 {dict(dist)}")
    print(f"  유형 {dict(tdist)}")
    missing = [r["case_id"] for r in rows if not r["reference"].get("risk")]
    if missing:
        print(f"  ✗ risk 가 비어 있는 행 {len(missing)}건: {', '.join(missing[:5])}")
    print(f"\n{out}")
    print("이 파일은 모델 입력이 아니다 — 판정이 끝난 뒤 case_id 로 조인해 대조용으로만 쓴다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
