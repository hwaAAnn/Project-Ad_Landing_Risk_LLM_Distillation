"""case_id 기준 병합 · 비교

    python -m pipeline.merge_results                        # 최신 실행끼리
    python -m pipeline.merge_results --tag 0829             # 이름 붙여 저장
    python -m pipeline.merge_results --dir 'results/{provider}-pilot'   # 파일럿 실행끼리

`results/{provider}/*.jsonl` 을 읽어 case_id로 조인한다. provider마다 실행 시점이 다르므로
같은 provider의 파일이 여럿이면 **가장 최근 run**을 쓰고, 그 안에서 성공한 행을 우선한다.
아직 안 돌린 provider는 빈 칸으로 두고 병합 — 세 사람이 각자 다른 시점에 끝내도
그때까지 나온 것으로 비교된다.

읽을 곳은 `--dir` 로 바꾼다. `run_eval.py --out-dir` 로 본 실행과 분리해 둔 결과
(파일럿·재현 실험 등)를 그 묶음끼리만 비교할 때 쓴다. `{provider}` 자리는 provider 이름으로
채워지므로 반드시 들어 있어야 한다 — 세 provider를 한 인자로 가리키기 위한 것이다.
`--tag` 를 안 주면 그 디렉터리 이름에서 따온다 (`results/{provider}-pilot` → `compare_pilot.*`).
기본 병합 결과를 파일럿 결과가 덮어쓰지 않도록.

산출:
 - `results/merged/compare_{tag}.jsonl`  case 한 건 = 한 줄 (세 모델 나란히)
 - `results/merged/compare_{tag}.csv`    같은 내용. 엑셀·시트로 열어 눈으로 볼 용도
 - 콘솔 요약: 3자 일치율 · 쌍별 불일치 · reference 대비 · 지연 · 실패
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

from pipeline import schema as S
from pipeline.providers.base import PROVIDER_NAMES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
RESULTS = os.path.join(HERE, "results")
RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


DEFAULT_DIR = "results/{provider}"


def _provider_dir(template: str, provider: str) -> str:
    """`--dir` 템플릿의 {provider} 를 채워 실제 경로로. 상대 경로는 저장소 기준"""
    path = template.format(provider=provider)
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def _tag_from(template: str) -> str:
    """`--tag` 를 안 준 때 쓸 이름표. `results/{provider}-pilot` → `pilot`.
    기본 경로로 낸 `compare_latest.*` 를 파일럿 결과가 조용히 덮어쓰지 않게 하려는 것"""
    if template == DEFAULT_DIR:
        return "latest"
    stem = os.path.basename(template.rstrip("/\\")).replace("{provider}", "")
    return stem.strip("-_ ") or "latest"


def _latest_run(dir_template: str, provider: str, run_id: str | None) -> dict[str, dict]:
    """provider 하나의 결과를 case_id → row 로. 최신 run 우선, 성공 행 우선"""
    d = _provider_dir(dir_template, provider)
    if not os.path.isdir(d):
        return {}
    files = sorted(fn for fn in os.listdir(d) if fn.endswith(".jsonl"))
    if run_id:
        files = [fn for fn in files if fn.startswith(run_id)]
    rows: dict[str, dict] = {}
    for fn in files:                     # 이름이 시각순이라 뒤 파일이 최신
        with open(os.path.join(d, fn), encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                cid = r.get("case_id")
                prev = rows.get(cid)
                # 최신을 쓰되, 최신이 실패이고 이전이 성공이면 성공을 남긴다 —
                # 재시도 실행에서 일부만 다시 돌린 경우를 위한 것
                if prev is None or r.get("ok") or not prev.get("ok"):
                    rows[cid] = r
    return rows


def _cell(row: dict | None) -> dict:
    if row is None:
        return {}
    if not row.get("ok"):
        return {"ok": False, "error": (row.get("error") or "")[:120],
                "latency_ms": row.get("latency_ms"), "model": row.get("model")}
    v = row["verdict"]
    raw = row.get("verdict_raw") or {}
    return {
        "ok": True,
        "risk": v["risk"], "type": v["type"],
        "reason": v.get("reason", ""), "advice": v.get("advice", ""), "evidence": v.get("evidence", ""),
        "risk_raw": raw.get("risk"), "type_raw": raw.get("type"),
        "ground_downgraded": (row.get("postprocess") or {}).get("ground_downgraded"),
        "fill_type_applied": (row.get("postprocess") or {}).get("fill_type_applied"),
        "latency_ms": row.get("latency_ms"), "model": row.get("model"),
    }


def _load_reference(path: str | None) -> dict[str, dict]:
    """`extract_references.py` 가 낸 파일을 case_id → reference 로

    가이드라인 §3 대로 Fixed Dataset 에는 과거 판정이 들어 있지 않다. 그래서 대조용
    reference 는 **판정이 끝난 뒤** 이렇게 옆에서 붙인다 — 모델 입력에 합치지 않는다.
    파일이 없으면 조용히 빈 대조로 넘어가지 않고 중단한다. `--reference` 를 준 것은
    "대조해서 보겠다"는 뜻이라, 경로를 틀렸을 때 0% 일치로 읽히면 안 된다
    """
    if not path:
        return {}
    p = path if os.path.isabs(path) else os.path.join(HERE, path)
    if not os.path.isfile(p):
        raise SystemExit(f"--reference 파일이 없다: {p}\n"
                         "  python -m pipeline.extract_references 로 먼저 만들어라")
    out: dict[str, dict] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            # reference 를 감싸 둔 형태(extract_references.py)와 평평한 형태 둘 다 받는다
            out[r["case_id"]] = r.get("reference") or {
                k: r[k] for k in ("risk", "type", "reason", "at") if k in r}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="case_id 기준으로 provider별 결과 병합")
    ap.add_argument("--cases", default=os.path.join(HERE, "dataset", "cases.jsonl"))
    ap.add_argument("--dir", dest="dir_template", default=DEFAULT_DIR,
                    help="결과를 읽을 곳. {provider} 자리가 provider 이름으로 채워진다 "
                         f"(기본: {DEFAULT_DIR} · 예: 'results/{{provider}}-pilot')")
    ap.add_argument("--tag", default=None,
                    help="산출 파일 이름표. 기본은 --dir 에서 따온다 (기본 경로면 latest)")
    ap.add_argument("--reference", default=None,
                    help="사후 대조용 과거 판정 파일 (예: references/historical_claude_251.jsonl). "
                         "모델 입력이 아니라 병합 결과에만 붙는다 — 가이드라인 §3·§8")
    ap.add_argument("--run-id", default=None, help="특정 run_id만 (provider별 접두사 일치)")
    ap.add_argument("--raw", action="store_true", help="후처리 전(verdict_raw) 기준으로 비교")
    args = ap.parse_args()

    if "{provider}" not in args.dir_template:
        raise SystemExit("--dir 에 {provider} 자리가 있어야 한다 — 세 provider를 한 인자로 "
                         f"가리키기 위한 것이다. 예: 'results/{{provider}}-pilot'")
    tag = args.tag or _tag_from(args.dir_template)

    cases = S.load_cases(args.cases)
    reference = _load_reference(args.reference)
    per = {p: _latest_run(args.dir_template, p, args.run_id) for p in PROVIDER_NAMES}
    have = [p for p in PROVIDER_NAMES if per[p]]
    if not have:
        raise SystemExit("결과가 없다 — "
                         f"{_provider_dir(args.dir_template, '<provider>')}/*.jsonl 을 먼저 만들어라")
    print(f"읽는 곳: {args.dir_template}  ·  이름표: {tag}")
    if args.reference:
        print(f"사후 대조 reference: {args.reference} — {len(reference)}건 "
              "(모델 입력에는 들어가지 않았다)")
    print(f"병합 대상: {', '.join(have)}"
          + (f"  (없음: {', '.join(p for p in PROVIDER_NAMES if not per[p])})" if len(have) < 3 else ""))
    key = "risk_raw" if args.raw else "risk"
    tkey = "type_raw" if args.raw else "type"
    print(f"비교 기준: {'모델 원본(verdict_raw)' if args.raw else '후처리 후(verdict)'}\n")

    merged = []
    for case in cases:
        cid = case["case_id"]
        cells = {p: _cell(per[p].get(cid)) for p in PROVIDER_NAMES}
        risks = [c.get(key) for c in cells.values() if c.get("ok")]
        types = [c.get(tkey) for c in cells.values() if c.get("ok")]
        row = {
            "case_id": cid,
            "url": case.get("url"),
            # --reference 를 준 때는 그 파일이 정본. 안 준 때만 case 안의 값을 본다
            # (blind dataset 이면 그 값은 비어 있고, 그대로 "대조 없음"이 된다)
            "reference": reference.get(cid) or case.get("reference") or {},
            "reference_source": (args.reference if reference.get(cid)
                                 else ("cases" if case.get("reference") else None)),
            **cells,
            "agree_risk": len(set(risks)) == 1 and len(risks) == len(have),
            "agree_type": len(set(types)) == 1 and len(types) == len(have),
            "majority_risk": Counter(risks).most_common(1)[0][0] if risks else None,
            # 가장 위험하게 본 모델과 가장 안전하게 본 모델의 등급 차 — 큰 것부터 사람이 본다
            "risk_spread": (max(RISK_ORDER.get(r, 0) for r in risks)
                            - min(RISK_ORDER.get(r, 0) for r in risks)) if risks else None,
        }
        merged.append(row)

    os.makedirs(os.path.join(RESULTS, "merged"), exist_ok=True)
    jpath = os.path.join(RESULTS, "merged", f"compare_{tag}.jsonl")
    with open(jpath, "w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cpath = os.path.join(RESULTS, "merged", f"compare_{tag}.csv")
    cols = (["case_id", "url", "ref_risk", "ref_type"]
            + [f"{p}_{k}" for p in PROVIDER_NAMES for k in ("risk", "type", "latency_ms")]
            + ["agree_risk", "agree_type", "majority_risk", "risk_spread"])
    with open(cpath, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in merged:
            flat = {"case_id": row["case_id"], "url": row["url"],
                    "ref_risk": row["reference"].get("risk"), "ref_type": row["reference"].get("type")}
            for p in PROVIDER_NAMES:
                c = row[p]
                flat[f"{p}_risk"] = c.get(key) if c.get("ok") else "FAIL"
                flat[f"{p}_type"] = c.get(tkey) if c.get("ok") else ""
                flat[f"{p}_latency_ms"] = c.get("latency_ms")
            for k in ("agree_risk", "agree_type", "majority_risk", "risk_spread"):
                flat[k] = row[k]
            w.writerow(flat)

    # ── 요약 ────────────────────────────────────────────────────────────────
    n = len(merged)
    full = [r for r in merged if all(r[p].get("ok") for p in have)]
    print(f"case {n}건 · 세 결과가 모두 있는 것 {len(full)}건")
    if full:
        ar = sum(r["agree_risk"] for r in full)
        at = sum(r["agree_type"] for r in full)
        print(f"등급 전원 일치 {ar}/{len(full)} ({100 * ar // len(full)}%) · "
              f"유형 전원 일치 {at}/{len(full)} ({100 * at // len(full)}%)")
        print(f"등급 2단계 차이(LOW↔HIGH) {sum(r['risk_spread'] == 2 for r in full)}건 — 먼저 볼 것")

    for p in have:
        rows = [r[p] for r in merged if r[p]]
        ok = [c for c in rows if c.get("ok")]
        dist = Counter(c.get(key) for c in ok)
        lats = sorted(c["latency_ms"] for c in ok if c.get("latency_ms") is not None)
        med = lats[len(lats) // 2] if lats else 0
        down = sum(1 for c in ok if c.get("ground_downgraded"))
        print(f"  {p:7} 성공 {len(ok)}/{len(rows)} · {dict(dist)} · 지연 중앙 {med}ms · 근거강등 {down}")

    refs = [r for r in merged if r["reference"].get("risk")]
    if refs:
        print(f"\nreference(기존 판정) 대비 등급 일치")
        print(f"  과거 판정이 있는 {len(refs)}/{n}건 중, 그 provider가 실제로 판정한 case만 센다")
        for p in have:
            m = [r for r in refs if r[p].get("ok")]
            hit = sum(1 for r in m if r[p].get(key) == r["reference"]["risk"])
            if m:
                print(f"  {p:7} {hit}/{len(m)} ({100 * hit // len(m)}%)")

    if len(have) > 1:
        print("\n쌍별 등급 불일치")
        for i, a in enumerate(have):
            for b in have[i + 1:]:
                m = [r for r in merged if r[a].get("ok") and r[b].get("ok")]
                diff = [r for r in m if r[a][key] != r[b][key]]
                print(f"  {a} ↔ {b}: {len(diff)}/{len(m)}"
                      + (f"  예) {', '.join(r['case_id'] for r in diff[:3])}" if diff else ""))

    print(f"\n{jpath}\n{cpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
