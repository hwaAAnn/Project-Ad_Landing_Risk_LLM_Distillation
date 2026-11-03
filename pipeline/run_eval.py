"""모델 판정 실행 — provider 하나를 골라 전 case를 돌린다

    export GEMINI_API_KEY=...
    python -m pipeline.run_eval --provider gemini
    python -m pipeline.run_eval --provider openai --model gpt-5
    python -m pipeline.run_eval --provider claude --limit 10 --resume

읽는 것은 두 파일뿐이다.
 - `dataset/cases.jsonl`      raw tool_outputs가 들어 있는 case 묶음
 - `dataset/prompt_spec.json` server/에서 1회 추출해 봉인한 판정 기준

## 입력 조립

case에는 **raw tool_outputs**가 그대로 들어 있고, user 프롬프트는 실행 시점에
`build_prompt.py`가 서비스와 같은 규칙으로 조립한다(슬롯 순서·라벨·3000/1200자 trim·
조건부 지시문). trim된 값을 case에 저장하지 않는 이유는, 자르기가 판정을 얼마나 바꾸는지
`--input-mode raw`로 바로 비교해 보기 위해서다.

## 여기서 네트워크는 모델 API 하나뿐이다

페이지를 다시 열지 않고, 광고를 다시 누르지 않고, 차단 목록·RDAP도 조회하지 않는다.
case의 `input.prompt_text`가 이미 완성된 user 메시지이므로 그대로 넣는다.
`server/` 모듈은 import조차 하지 않는다 — 그쪽을 올리면 requests 세션과 RDAP 스레드풀이
실행 경로에 함께 올라온다(judge_agent.py:87·294).

문서로만 두지 않고 `--guard`(기본 켬)로 강제한다. provider API 호스트 외의 TCP 연결은
그 자리에서 예외 — 실수로 수집 코드가 섞여도 조용히 통과하지 않는다.

## 결과

`results/{provider}/{run_id}.jsonl` 에 case 한 건당 한 줄. **실패도 남긴다** —
빼먹으면 "스키마를 못 지킨 5건"이 파일에서 사라지고 남은 95건의 일치율만 공정해 보인다
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone

from pipeline import build_prompt
from pipeline import postprocess
from pipeline import schema as S
from pipeline.providers.base import PROVIDER_NAMES, Request, get_provider

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
DEFAULT_CASES = os.path.join(HERE, "dataset", "cases.jsonl")
DEFAULT_SPEC = os.path.join(HERE, "dataset", "prompt_spec.json")

# provider API 외의 바깥 연결을 막는다. 페이지 재수집·차단 목록·RDAP이 섞여 들어오는 것을
# 코드 리뷰가 아니라 실행이 잡아내게 하는 장치
_ALLOWED_HOST_SUFFIXES = (
    "anthropic.com",
    "googleapis.com", "google.com",
    "openai.com",
)


def _install_network_guard() -> None:
    real = socket.create_connection

    def guarded(address, *a, **kw):
        host = (address[0] if isinstance(address, tuple) else str(address)).lower()
        if not any(host == s or host.endswith("." + s) for s in _ALLOWED_HOST_SUFFIXES):
            raise RuntimeError(
                f"모델 비교 단계에서 허용되지 않은 연결: {host}\n"
                "  이 파이프라인은 저장된 snapshot만 읽는다 — 페이지 재수집·광고 클릭 금지.\n"
                "  provider API가 새 호스트를 쓴다면 _ALLOWED_HOST_SUFFIXES에 추가하라."
            )
        return real(address, *a, **kw)

    socket.create_connection = guarded


def _block_all_network() -> None:
    """dry-run 전용 — **어떤** 연결도 막는다. 모델 API도 포함

    조립만 하는 모드라 바깥으로 나갈 이유가 하나도 없다. 실수로 provider 호출이 섞이면
    비용이 발생하므로, 문서가 아니라 실행이 막게 한다
    """
    def blocked(address, *a, **kw):
        host = (address[0] if isinstance(address, tuple) else str(address))
        raise RuntimeError(f"--dry-run 은 네트워크를 쓰지 않는다 (차단: {host})")

    socket.create_connection = blocked


def _pct(vals: list[int], q: float) -> int:
    v = sorted(vals)
    return v[min(len(v) - 1, int(len(v) * q))] if v else 0


def dry_run(args, spec: dict, exclude: tuple[str, ...]) -> int:
    """provider API를 부르지 않고 조립까지만. API 키·SDK 없이 돌아간다

    100건을 넣은 뒤 **돈을 쓰기 전에** 입력이 제대로 만들어지는지 보는 용도다.
    각 case의 프롬프트 전문과 진단을 `results/dryrun/`에 남긴다
    """
    _block_all_network()
    cases, problems = S.inspect_cases(args.cases, input_mode=args.input_mode)
    total_lines = len(cases) + len(problems)

    print(f"입력  {args.cases}")
    print(f"기준  {spec['spec_id']} ({spec['system_sha256'][:12]}…) · system {len(spec['system'])}자")
    print(f"모드  {args.input_mode} · 제외 {list(exclude) or '없음'}\n")

    print(f"1. case 로드    {len(cases)}/{total_lines}건 정상"
          + (f"  ✗ 실패 {len(problems)}건" if problems else "  ✓"))
    for msg in problems[:10]:
        print(f"     - {msg}")
    if len(problems) > 10:
        print(f"     … 외 {len(problems) - 10}건")

    built = 0
    failed: list[tuple[str, str]] = []
    lengths: list[int] = []
    trim_fetch: list[str] = []
    trim_other: dict[str, list[str]] = {}
    parse_failed: list[tuple[str, list[str]]] = []
    unknown: list[tuple[str, list[str]]] = []
    collided: list[tuple[str, list[str]]] = []
    missing_core: list[tuple[str, list[str]]] = []
    slot_count: dict[str, int] = {}

    os.makedirs(os.path.join(HERE, "results", "dryrun"), exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(HERE, "results", "dryrun", f"{stamp}-{args.input_mode}.jsonl")

    with open(out_path, "w", encoding="utf-8") as out:
        for case in cases:
            cid = case["case_id"]
            inp = case.get("input") or {}
            url = inp.get("url") or case.get("url") or ""
            tools = inp.get("tool_outputs") or []

            rep = build_prompt.slot_report(tools, url) if args.input_mode != "stored" else {}
            cut = build_prompt.trim_report(tools, url) if args.input_mode != "stored" else {}
            try:
                _, user = build_prompt.build_messages(
                    case, spec, input_mode=args.input_mode, exclude=exclude)
            except ValueError as e:
                failed.append((cid, str(e)))
                out.write(json.dumps({"case_id": cid, "ok": False, "error": str(e),
                                      "diagnostics": rep}, ensure_ascii=False) + "\n")
                continue

            built += 1
            lengths.append(len(user))
            for slot in rep.get("slots", []):
                slot_count[slot] = slot_count.get(slot, 0) + 1
            if "fetch" in cut:
                trim_fetch.append(cid)
            for slot in cut:
                if slot != "fetch":
                    trim_other.setdefault(slot, []).append(cid)
            if rep.get("parse_failed"):
                parse_failed.append((cid, rep["parse_failed"]))
            if rep.get("unknown_tools"):
                unknown.append((cid, rep["unknown_tools"]))
            if rep.get("collisions"):
                collided.append((cid, rep["collisions"]))
            if rep.get("missing_core"):
                missing_core.append((cid, rep["missing_core"]))

            out.write(json.dumps({
                "case_id": cid, "ok": True, "url": url,
                "prompt_chars": len(user),
                "prompt_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
                "trimmed": {k: {"chars": v[0], "cut": v[1]} for k, v in cut.items()},
                "diagnostics": rep,
                "prompt": user,          # 전문 — 눈으로 확인할 수 있게
            }, ensure_ascii=False) + "\n")

    print(f"\n2. 프롬프트 생성 성공 {built}건 · 실패 {len(failed)}건"
          + ("  ✓" if not failed else "  ✗"))
    for cid, why in failed[:10]:
        print(f"     - {cid}: {why}")

    n_other = sum(len(v) for v in trim_other.values())
    print(f"\n3. trim 발생 (production-trim 규칙)")
    print(f"     fetch_page  3000자 초과 {len(trim_fetch)}건"
          + (f"  예) {', '.join(trim_fetch[:3])}" if trim_fetch else ""))
    print(f"     그 밖 도구  1200자 초과 {n_other}건"
          + (f"  ({', '.join(f'{k} {len(v)}건' for k, v in sorted(trim_other.items()))})" if trim_other else ""))
    if args.input_mode == "raw":
        print("     (raw 모드라 실제로는 자르지 않는다 — 위는 production-trim이면 잘릴 곳)")

    if lengths:
        print(f"\n4. user prompt 길이 (문자)")
        print(f"     min {min(lengths):,} · median {_pct(lengths, 0.5):,} · "
              f"p90 {_pct(lengths, 0.9):,} · max {max(lengths):,}")
        buckets = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 6000), (6000, 10**9)]
        for lo, hi in buckets:
            n = sum(1 for x in lengths if lo <= x < hi)
            if n:
                label = f"{lo:,}~{hi:,}" if hi < 10**9 else f"{lo:,}~"
                print(f"       {label:>13}  {'█' * max(1, round(40 * n / len(lengths)))} {n}건")

    print(f"\n5. tool_outputs 파싱   "
          + (f"✗ 실패 {len(parse_failed)}건" if parse_failed else "✓ 전건 정상"))
    for cid, items in parse_failed[:5]:
        print(f"     - {cid}: {items[:2]}")

    print(f"\n6. 슬롯 복원")
    if slot_count:
        order = [k for k, _ in build_prompt.PREFETCH_SLOTS]
        print("     자리별 보유 건수  "
              + " · ".join(f"{k} {slot_count.get(k, 0)}" for k in order if slot_count.get(k)))
    issues = 0
    for label, rows in (("핵심 슬롯(fetch·signals) 누락", missing_core),
                        ("같은 자리 중복 — 뒤엣것이 버려짐", collided),
                        ("모르는 도구 이름 — 어느 자리에도 못 들어감", unknown)):
        if rows:
            issues += len(rows)
            print(f"     ✗ {label} {len(rows)}건")
            for cid, what in rows[:5]:
                print(f"        - {cid}: {what}")
        else:
            print(f"     ✓ {label} 없음")

    ok = not problems and not failed and not parse_failed and issues == 0
    print("\n" + "─" * 62)
    print(("전건 정상 — 이대로 실제 실행 가능" if ok
           else "위 ✗ 항목을 먼저 확인하라 (실제 실행은 첫 문제에서 중단된다)"))
    print(f"case별 프롬프트 전문·진단: {out_path}")
    print("\n실제 실행은 provider SDK와 API 키가 필요하다:")
    print("  pip install anthropic   # 또는 google-genai · openai")
    print("  export ANTHROPIC_API_KEY=...")
    print("  python -m pipeline.run_eval --provider claude --limit 3")
    return 0 if ok else 1


def _run_id(provider: str, spec: dict) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{provider}-{spec['system_sha256'][:6]}"


def _done_case_ids(out_dir: str) -> set[str]:
    """--resume 용. 같은 provider의 이전 실행에서 **성공한** case만 건너뛴다 —
    실패한 것은 다시 시도해야 한다"""
    done: set[str] = set()
    if not os.path.isdir(out_dir):
        return done
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(out_dir, fn), encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("ok"):
                    done.add(r.get("case_id"))
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description="provider 하나로 전 case 판정")
    ap.add_argument("--provider", choices=list(PROVIDER_NAMES),
                    help="실제 실행에 필수. --dry-run 에는 필요 없다")
    ap.add_argument("--dry-run", action="store_true",
                    help="provider API를 부르지 않고 조립까지만 — API 키·SDK 없이 실행 가능")
    ap.add_argument("--model", default=None, help="기본 모델 대신 쓸 모델 이름")
    ap.add_argument("--cases", default=DEFAULT_CASES)
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--input-mode", choices=list(build_prompt.INPUT_MODES), default="production-trim",
                    help="production-trim: 서비스와 동일(3000/1200 trim) · raw: trim 없음 · stored: 저장된 prompt_text 사용")
    ap.add_argument("--include-lookup-cache", action="store_true",
                    help="과거 판정(lookup_cache)을 입력에 포함 — 서비스 재현 검증용. 기본은 제외")
    ap.add_argument("--temperature", type=float, default=None,
                    help="기본은 spec의 값(운영은 미지정). 재현성 실험에서만 0 같은 값을 준다")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="기본은 spec의 값(450 — 운영 Claude 기준). 추론 모델(gpt-5·gemini 3.x)은 "
                         "추론에 그 예산을 다 써 본문이 안 나오므로 2000 정도가 필요하다")
    ap.add_argument("--out-dir", default=None, help="기본: results/{provider}")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N건만 (0=전부)")
    ap.add_argument("--resume", action="store_true", help="이전 실행에서 성공한 case는 건너뛴다")
    ap.add_argument("--retries", type=int, default=2, help="일시적 실패 재시도 횟수")
    ap.add_argument("--sleep", type=float, default=0.0, help="호출 간 대기(초) — 속도 제한 회피")
    ap.add_argument("--no-guard", action="store_true", help="네트워크 가드 해제 (권장하지 않음)")
    args = ap.parse_args()

    spec = build_prompt.load_spec(args.spec)
    exclude = () if args.include_lookup_cache else build_prompt.DEFAULT_EXCLUDE
    if args.dry_run:
        return dry_run(args, spec, exclude)
    if not args.provider:
        raise SystemExit("--provider 를 지정하라 (또는 --dry-run 으로 조립만 확인)")
    cases = S.load_cases(args.cases, input_mode=args.input_mode)
    # spec의 temperature가 정본. --temperature 를 준 때만 덮어쓴다 —
    # 그 사실은 결과 행에 남아 나중에 "이건 운영과 다른 설정으로 잰 것"임을 알 수 있다
    temperature = args.temperature if args.temperature is not None else spec["params"]["temperature"]
    # max_tokens 도 같은 규칙 — spec 이 정본, CLI 를 준 때만 덮어쓰고 그 사실을 결과 행에 남긴다.
    # spec 의 450 은 운영 Claude(haiku-4-5)에서 뽑은 값인데, 추론 모델은 그 예산을 **추론에만**
    # 다 쓴다(실측: gpt-5 450/450, gemini-3.1-pro 사고 428). 한도에 걸린 것과 모델이 답을 못
    # 낸 것은 다른 실패라서, 어느 한도로 쟀는지가 결과에 남아 있어야 한다
    max_tokens = args.max_tokens if args.max_tokens is not None else spec["params"]["max_tokens"]
    if args.resume:
        done = _done_case_ids(args.out_dir or os.path.join(HERE, "results", args.provider))
        before = len(cases)
        cases = [c for c in cases if c["case_id"] not in done]
        print(f"resume: 이미 성공한 {before - len(cases)}건 건너뜀")
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("돌릴 case가 없다")
        return 0

    if not args.no_guard:
        _install_network_guard()

    provider = get_provider(args.provider)
    model = args.model or provider.default_model()
    run_id = _run_id(args.provider, spec)
    out_dir = args.out_dir or os.path.join(HERE, "results", args.provider)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.jsonl")

    print(f"provider {args.provider} · model {model} · spec {spec['spec_id']} "
          f"({spec['system_sha256'][:12]}…)")
    print(f"입력 {args.input_mode} · 제외 {list(exclude) or '없음'} · temperature "
          f"{temperature if temperature is not None else '미지정(운영과 동일)'}"
          f" · max_tokens {max_tokens}"
          + ("" if args.max_tokens is None else f" (spec {spec['params']['max_tokens']} 를 덮어씀)"))
    print(f"case {len(cases)}건 → {out_path}\n")

    counts = {"ok": 0, "fail": 0}
    risks: dict[str, int] = {}
    lats: list[int] = []

    with open(out_path, "w", encoding="utf-8") as out:
        for i, case in enumerate(cases, 1):
            try:
                system, user = build_prompt.build_messages(
                    case, spec, input_mode=args.input_mode, exclude=exclude)
            except ValueError as e:
                # 조립 자체가 안 되는 case도 결과 파일에 남긴다 — 조용히 빠지면 건수가 어긋난다
                row = S.make_result(case_id=case["case_id"], run_id=run_id, provider=args.provider,
                                    model=model, spec=spec, input_mode=args.input_mode,
                                    excluded=exclude, temperature=temperature, max_tokens=max_tokens,
                                    prompt_sha256="",
                                    verdict_raw=None, verdict=None,
                                    postprocess={"fill_type_applied": None, "ground_downgraded": None},
                                    latency_ms=0, usage={}, attempts=0, error=f"조립 실패: {e}")
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                counts["fail"] += 1
                print(f"[{i:3}/{len(cases)}] {case['case_id']:12}     -  조립 실패 {e}")
                continue
            prompt_sha = hashlib.sha256(user.encode("utf-8")).hexdigest()
            req = Request(system=system, user=user, schema=spec["output_schema"],
                          model=model,
                          max_tokens=max_tokens,
                          temperature=temperature)

            verdict_raw = verdict = None
            post: dict = {"fill_type_applied": None, "ground_downgraded": None}
            usage: dict = {}
            error: str | None = None
            attempts = 0
            t0 = time.monotonic()
            for attempt in range(1, args.retries + 2):
                attempts = attempt
                try:
                    resp = provider.generate(req)
                    usage = resp.usage
                    if resp.error:
                        error = resp.error
                        break          # 스키마 준수 실패는 다시 물어도 같다 — 재시도 낭비
                    verdict_raw = S.validate_verdict(resp.verdict, spec["output_schema"])
                    # 근거 대조도 모델에게 **보여 준 것**으로만 — 프롬프트에서 뺀 도구는 여기서도 뺀다
                    verdict, post = postprocess.apply(verdict_raw, S.tool_outputs_of(case, exclude))
                    error = None
                    break
                except ValueError as e:
                    error = f"스키마 위반: {e}"
                    break
                except Exception as e:
                    error = f"{type(e).__name__}: {str(e)[:200]}"
                    if attempt <= args.retries:
                        time.sleep(min(8.0, 2.0 ** attempt))
                        continue
                    break
            latency = int((time.monotonic() - t0) * 1000)

            row = S.make_result(case_id=case["case_id"], run_id=run_id, provider=args.provider,
                                model=model, spec=spec, input_mode=args.input_mode,
                                excluded=exclude, temperature=temperature, max_tokens=max_tokens,
                                prompt_sha256=prompt_sha,
                                verdict_raw=verdict_raw, verdict=verdict,
                                postprocess=post, latency_ms=latency, usage=usage,
                                attempts=attempts, error=error)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()          # 중간에 끊겨도 여기까지는 남는다 — --resume이 이어받는다

            if row["ok"]:
                counts["ok"] += 1
                risks[verdict["risk"]] = risks.get(verdict["risk"], 0) + 1
                lats.append(latency)
                mark = f"{verdict['risk']:6} {verdict['type']}"
                if post["ground_downgraded"]:
                    mark += " (근거 없어 강등)"
            else:
                counts["fail"] += 1
                mark = f"실패 {error[:60]}"
            print(f"[{i:3}/{len(cases)}] {case['case_id']:12} {latency:5}ms  {mark}")

            if args.sleep:
                time.sleep(args.sleep)

    print("\n" + "─" * 60)
    print(f"성공 {counts['ok']} · 실패 {counts['fail']} · 등급 분포 {risks}")
    if lats:
        ls = sorted(lats)
        p = lambda q: ls[min(len(ls) - 1, int(len(ls) * q))]
        print(f"지연 중앙값 {p(0.5)}ms · p90 {p(0.9)}ms · 최대 {ls[-1]}ms")
    print(f"기록: {out_path}")
    print(f"세 provider가 끝나면:  python -m pipeline.merge_results")
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
