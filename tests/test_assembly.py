"""조립 검증 — build_prompt.py의 결과가 실제 서비스와 같은가

두 갈래로 확인한다. 둘 다 네트워크를 쓰지 않는다.

| 검사 | 정답의 출처 | 무엇을 덮는가 |
| :-- | :-- | :-- |
| A. 실제 페이지 대조 | `collect_device/out/.../offline_refetch_sample.jsonl`의 `prompt_text` | 실제 페이지에서 나온 진짜 도구 결과 |
| B. 서비스 코드 직접 실행 | `server/judge_agent.py::_prefetch` 를 가짜 run_tool로 호출 | trim 경계·4xx·js_only·공식도메인·슬롯 누락 |

**정답의 무게가 다르다.** A의 `prompt_text`는 서버가 아니라 오프라인 수집기
(`experiments/collect_dataset.py` — 이 저장소에 없다)가 만든 것이라, 서버 조립과 이미
다른 부분이 있다. 그래서 A는 불일치를 **분류**한다 — 알려진 수집기 차이면 설명된 것으로,
그 밖이면 실패로 센다. 서비스와 같은지를 판정하는 것은 B다.

B는 `_prefetch`가 `run_tool`을 **인자로 받는** 함수라서 가능하다 — 도구를 가짜로 넣으면
네트워크 없이 서비스의 요약 조립 코드 그 자체가 돌아간다. 정답을 손으로 적지 않고
원본 코드에서 뽑으므로, 원본이 바뀌면 이 검사가 먼저 깨진다.

    python -m tests.test_assembly
"""

from __future__ import annotations

import json
import os
import re
import sys

from pipeline import build_prompt as B

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root

# 운영 저장소의 위치 — **선택 사항.** A·B 검사에만 쓴다.
# 없으면 그 둘을 건너뛰고 C(우리 코드만 보는 검사)는 그대로 돈다
#
#     export JUDGE_SERVICE_ROOT=~/judge-service
JUDGE_SERVICE_ROOT = os.path.expanduser(os.environ.get("JUDGE_SERVICE_ROOT", ""))
SAMPLE = (os.path.join(JUDGE_SERVICE_ROOT, "collect_device", "out", "0828-server100",
                       "offline_refetch_sample.jsonl") if JUDGE_SERVICE_ROOT else "")


def diff_report(want: str, got: str, label: str) -> str:
    """어디서부터 갈렸는지 — 문자 위치와 앞뒤 문맥"""
    if want == got:
        return ""
    n = min(len(want), len(got))
    i = next((k for k in range(n) if want[k] != got[k]), n)
    return (f"\n    {label} 불일치 (길이 want={len(want)} got={len(got)}, 첫 차이 {i}번째 글자)\n"
            f"      want …{want[max(0, i - 60):i + 60]!r}\n"
            f"      got  …{got[max(0, i - 60):i + 60]!r}")


# ── A. 실제 판정 프롬프트와 대조 ─────────────────────────────────────────────


def _case_from_sample(row: dict) -> dict:
    """수집본의 `tools`(도구 이름 → 결과 dict)를 cases.jsonl의 tool_outputs 형태로

    브랜드 조회는 `brands`에 {브랜드: 결과}로 뭉쳐 있으므로 하나씩 풀어 준다.
    args는 서비스가 실제로 넘긴 것과 같게 채운다 — 슬롯 판별이 args를 보기 때문
    """
    t = row["tools"]
    url, final_url = row["url"], row.get("final_url") or row["url"]
    outs = []
    if "fetch_page" in t:
        outs.append({"tool": "fetch_page", "args": {"url": url}, "output": t["fetch_page"]})
    if "page_signals" in t:
        outs.append({"tool": "page_signals", "args": {"url": final_url}, "output": t["page_signals"]})
    if "official" in t:
        outs.append({"tool": "official_domain_of",
                     "args": {"brand": (t["official"] or {}).get("domain", "")},
                     "output": t["official"]})
    for brand, out in (t.get("brands") or {}).items():
        outs.append({"tool": "official_domain_of", "args": {"brand": brand}, "output": out})
    if "check_blocklist" in t:
        outs.append({"tool": "check_blocklist",
                     "args": {"host": (t["check_blocklist"] or {}).get("host", "")},
                     "output": t["check_blocklist"]})
    if "lookup_cache" in t:
        outs.append({"tool": "lookup_cache",
                     "args": {"site": (t["lookup_cache"] or {}).get("site", "")},
                     "output": t["lookup_cache"]})
    if "domain_age" in t:
        outs.append({"tool": "domain_age",
                     "args": {"host": (t["domain_age"] or {}).get("domain", "")},
                     "output": t["domain_age"]})
    if "domain_age_final" in t:
        outs.append({"tool": "domain_age",
                     "args": {"host": (t["domain_age_final"] or {}).get("domain", "")},
                     "output": t["domain_age_final"]})
    # 스레드 때문에 실제 순서는 뒤섞인다 — 그래도 같은 결과가 나와야 한다.
    # 일부러 거꾸로 넣어 순서 의존이 없음을 함께 확인
    outs.reverse()
    return {"case_id": row["url"][:40], "input": {"url": url, "tool_outputs": outs},
            "url": url}


# 오프라인 수집기가 서버와 다르게 구는 지점. 서버는 조회하지 않은 슬롯에도
# "(시간 안에 못 받음)" 줄을 내지만(judge_agent.py:799), 수집기는 그 줄을 빼고 만든다.
# `_prefetch`를 직접 돌려 확인했다 — 검사 B의 "평범한 쇼핑몰" 시나리오가 그 증거다
KNOWN_COLLECTOR_DIFF = "- domain_age(도착지): (시간 안에 못 받음)\n"


def test_real_prompts() -> tuple[int, int]:
    if not JUDGE_SERVICE_ROOT:
        print("A. 실제 페이지 대조 — 건너뜀 (JUDGE_SERVICE_ROOT 미설정)")
        return (0, 0)
    if not os.path.exists(SAMPLE):
        print(f"A. 실제 페이지 대조 — 건너뜀 (없음: {SAMPLE})")
        return (0, 0)
    rows = [json.loads(l) for l in open(SAMPLE, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("prompt_text")]
    print(f"A. 실제 페이지 대조 — {len(rows)}건 "
          f"(정답: 오프라인 수집기가 만든 prompt_text. 서버 조립과 일부 다름)")
    ok = 0
    for r in rows:
        case = _case_from_sample(r)
        # 서비스 재현이므로 lookup_cache를 **포함**해 조립한다
        got = B.build_user_prompt(case, input_mode="production-trim", exclude=())
        want = r["prompt_text"]
        trims = got.count("…(+")
        if got == want:
            ok += 1
            note = "완전 일치"
        elif got.replace(KNOWN_COLLECTOR_DIFF, "") == want:
            ok += 1     # 설명된 차이 — 우리 쪽이 서버와 같고 수집기가 그 줄을 뺀 것
            note = "설명됨: 수집기가 age_final '못 받음' 줄을 뺀 차이뿐 (서버는 낸다)"
        else:
            note = "설명 안 되는 차이" + diff_report(want, got, "prompt")
        flags = B.prefetch_flags(B.slots_of(case["input"]["tool_outputs"], case["url"]))
        print(f"  {'✓' if '설명 안' not in note else '✗'} {r['url'][:46]:48} {len(want):5}자 "
              f"trim {trims}곳  unreadable={str(flags['unreadable']):5} · {note}")
    return (ok, len(rows))


# ── B. 서비스 코드(_prefetch)를 직접 돌려 대조 ───────────────────────────────


SCENARIOS = {
    "평범한 쇼핑몰 (본문 있음·브랜드 없음)": {
        "url": "https://shop.example.com/sale",
        "fetch": {"final_url": "https://shop.example.com/sale", "status": 200,
                  "chain": ["https://shop.example.com/sale"], "text": "여름 세일 최대 70%", "text_chars": 12},
        "signals": {"title": "샵", "host": "shop.example.com", "text_chars": 12, "js_only": False,
                    "meta_refresh": False, "forms": [], "password_field": False, "asks_rrn": False,
                    "asks_card": False, "external_scripts": [], "iframes": [], "apk_links": [],
                    "brand_mentions": [], "brand_domain_mismatch": []},
        "blocklist": {"host": "shop.example.com", "listed": False, "list": None},
        "cache": {"site": "example.com", "previous": None},
        "age": {"domain": "example.com", "age_days": 4000, "young": False, "note": "ok"},
    },
    "본문 3000자 초과 → fetch trim 경계": {
        "url": "https://news.example.com/a",
        # 실제 서비스는 fetch_chain이 본문을 MAX_PAGE_CHARS(2500)로 먼저 자르므로 fetch JSON이
        # 3000자를 넘는 일이 흔치 않다. 여기서는 trim 경계를 확인하려고 일부러 넘긴다
        "fetch": {"final_url": "https://news.example.com/a", "status": 200,
                  "chain": ["https://news.example.com/a"], "text": "가" * 4000, "text_chars": 4000},
        "signals": {"title": "뉴스", "host": "news.example.com", "text_chars": 4000, "js_only": False,
                    "meta_refresh": False, "forms": [], "password_field": False, "asks_rrn": False,
                    "asks_card": False, "external_scripts": [], "iframes": [], "apk_links": [],
                    "brand_mentions": [], "brand_domain_mismatch": []},
        "blocklist": {"host": "news.example.com", "listed": False, "list": None},
        "cache": {"site": "example.com", "previous": None},
        "age": {"domain": "example.com", "age_days": 4000, "young": False, "note": "ok"},
    },
    "JS 렌더링(js_only) + 공식 도메인 확인됨": {
        "url": "https://www.axa.co.kr/",
        "fetch": {"final_url": "https://www.axa.co.kr/", "status": 200,
                  "chain": ["https://www.axa.co.kr/"], "text": "", "text_chars": 0},
        "signals": {"title": "", "host": "www.axa.co.kr", "text_chars": 0, "js_only": True,
                    "meta_refresh": False, "forms": [], "password_field": False, "asks_rrn": False,
                    "asks_card": False, "external_scripts": [], "iframes": [], "apk_links": [],
                    "brand_mentions": [], "brand_domain_mismatch": []},
        "official": {"domain": "www.axa.co.kr", "brand": "AXA손해보험",
                     "official_domains": ["axa.co.kr"], "known": True,
                     "note": "www.axa.co.kr은(는) AXA손해보험의 공식 도메인이다"},
        "blocklist": {"host": "www.axa.co.kr", "listed": False, "list": None},
        "cache": {"site": "axa.co.kr", "previous": None},
        "age": {"domain": "axa.co.kr", "age_days": None, "young": False, "note": "RDAP 미지원 TLD (.kr)"},
    },
    "403 차단(unreadable) + 공식 도메인 표에 없음": {
        "url": "https://weird-shop.top/x",
        "fetch": {"final_url": "https://weird-shop.top/x", "status": 403,
                  "chain": ["https://weird-shop.top/x"], "text": "", "text_chars": 0},
        "signals": {"title": "", "host": "weird-shop.top", "text_chars": 0, "js_only": False,
                    "meta_refresh": False, "forms": [], "password_field": False, "asks_rrn": False,
                    "asks_card": False, "external_scripts": [], "iframes": [], "apk_links": [],
                    "brand_mentions": [], "brand_domain_mismatch": []},
        "official": {"domain": "weird-shop.top", "known": False,
                     "note": "weird-shop.top은(는) 공식 도메인 표에 없다 — 표에 없다는 것은 사칭의 근거가 아니다"},
        "blocklist": {"host": "weird-shop.top", "listed": False, "list": None},
        "cache": {"site": "weird-shop.top", "previous": None},
        "age": {"domain": "weird-shop.top", "age_days": 3, "young": True, "note": "2026-08-25 등록"},
    },
    "브랜드 언급 + 도착지 도메인 다름(age_final)": {
        "url": "https://cyad1.nate.com/click.kti/abc",
        "fetch": {"final_url": "https://kb-star.top/login", "status": 200,
                  "chain": ["https://cyad1.nate.com/click.kti/abc", "https://kb-star.top/login"],
                  "text": "KB국민은행 로그인 아이디 비밀번호", "text_chars": 21},
        "signals": {"title": "KB국민은행", "host": "kb-star.top", "text_chars": 21, "js_only": False,
                    "meta_refresh": False,
                    "forms": [{"action_host": "kb-star.top", "external_action": False,
                               "fields": ["text:id", "password:pw"], "asks": ["password"]}],
                    "password_field": True, "asks_rrn": False, "asks_card": False,
                    "external_scripts": [], "iframes": [], "apk_links": [],
                    "brand_mentions": ["KB국민은행", "국민은행"],
                    "brand_domain_mismatch": [{"brand": "KB국민은행", "official": ["kbstar.com"],
                                               "actual_host": "kb-star.top"}]},
        "brands": {"KB국민은행": {"brand": "KB국민은행", "official_domains": ["kbstar.com"], "known": True},
                   "국민은행": {"brand": "국민은행", "official_domains": ["kbstar.com"], "known": True}},
        "blocklist": {"host": "cyad1.nate.com", "listed": False, "list": None},
        "cache": {"site": "nate.com", "previous": None},
        "age": {"domain": "nate.com", "age_days": 9000, "young": False, "note": "ok"},
        "age_final": {"domain": "kb-star.top", "age_days": 2, "young": True, "note": "2026-08-26 등록"},
    },
    "도구 일부가 시간 안에 못 옴 (슬롯 누락)": {
        "url": "https://slow.example.com/",
        "fetch": {"final_url": "https://slow.example.com/", "status": 200,
                  "chain": ["https://slow.example.com/"], "text": "느린 사이트", "text_chars": 6},
        "signals": {"title": "느림", "host": "slow.example.com", "text_chars": 6, "js_only": False,
                    "meta_refresh": False, "forms": [], "password_field": False, "asks_rrn": False,
                    "asks_card": False, "external_scripts": [], "iframes": [], "apk_links": [],
                    "brand_mentions": [], "brand_domain_mismatch": []},
        # blocklist·cache·age 전부 없음 → "(시간 안에 못 받음)" 세 줄이 나와야 한다
    },
}


def _service_truth(sc: dict, oneshot: bool = True) -> tuple[dict, list[dict]]:
    """실제 `judge_agent._prefetch`를 가짜 run_tool로 돌려 정답을 얻는다

    네트워크는 쓰지 않는다 — run_tool이 미리 준비한 결과를 돌려줄 뿐이다.
    동시에 그 호출들을 tool_outputs 형태로 모아, 우리 조립기의 입력으로 쓴다
    """
    import judge_agent as J

    captured: list[dict] = []
    url = sc["url"]
    fsite = J.site_of(J.host_of(sc["fetch"]["final_url"]) or "")

    def run_tool(name: str, args: dict) -> str:
        if name == "fetch_page":
            out = sc["fetch"]
        elif name == "page_signals":
            out = sc.get("signals", {"error": "없음"})
        elif name == "check_blocklist":
            out = sc.get("blocklist")
        elif name == "lookup_cache":
            out = sc.get("cache")
        elif name == "domain_age":
            asked = J.site_of(args.get("host", "")) or args.get("host", "")
            out = sc.get("age_final") if asked == fsite and "age_final" in sc else sc.get("age")
        elif name == "official_domain_of":
            b = args.get("brand", "")
            out = sc.get("brands", {}).get(b) if not J._looks_like_domain(b) else sc.get("official")
        else:
            out = {"error": f"unknown {name}"}
        if out is None:
            # 서비스에서 "시간 안에 못 받음"이 되는 경로 — got에 키가 안 생기게 예외를 던진다
            raise TimeoutError(f"{name} 없음")
        text = json.dumps(out, ensure_ascii=False)
        captured.append({"tool": name, "args": args, "output": text})
        return text

    class FakeDeps:
        db_path = ":memory:"
        blocked_by = staticmethod(lambda u: None)
        cache_lookup = staticmethod(lambda s: None)

    budget = J._Budget(__import__("time").monotonic())
    pre = J._prefetch(url, FakeDeps(), budget, run_tool, oneshot=oneshot)
    return pre, captured


def test_service_code() -> tuple[int, int]:
    if not JUDGE_SERVICE_ROOT:
        print("\nB. 서비스 코드 대조 — 건너뜀 (JUDGE_SERVICE_ROOT 미설정)")
        return (0, 0)
    sys.path.insert(0, os.path.join(JUDGE_SERVICE_ROOT, "server"))
    try:
        import judge_agent  # noqa: F401
    except ImportError as e:
        print(f"\nB. 서비스 코드 대조 — 건너뜀 ({e}; pip install requests beautifulsoup4)")
        return (0, 0)

    print(f"B. 서비스 코드 대조 — {len(SCENARIOS)}가지 "
          f"(정답: server/judge_agent.py::_prefetch 를 그대로 실행한 결과)")
    ok = 0
    for label, sc in SCENARIOS.items():
        pre, captured = _service_truth(sc)
        case = {"case_id": label, "url": sc["url"],
                "input": {"url": sc["url"], "tool_outputs": list(reversed(captured))}}

        got_slots = B.slots_of(case["input"]["tool_outputs"], sc["url"])
        got_summary = B.summary_of(got_slots, trim_fetch=B.TRIM_FETCH, trim_other=B.TRIM_OTHER)
        got_flags = B.prefetch_flags(got_slots)

        want_flags = {"js_only": pre.get("js_only"), "unreadable": pre.get("unreadable"),
                      "official_known": pre.get("official_known")}
        d_sum = diff_report(pre["summary"], got_summary, "summary")
        d_flag = "" if got_flags == want_flags else f"\n    플래그 불일치 want={want_flags} got={got_flags}"

        # trim 경계 확인 — 서비스가 자른 자리와 우리가 자른 자리가 같은가
        trimmed = [ln.split(": ", 1)[0] for ln in pre["summary"].split("\n") if "…(+" in ln]
        same = not d_sum and not d_flag
        ok += same
        print(f"  {'✓' if same else '✗'} {label:38} 요약 {len(pre['summary']):5}자  "
              f"trim된 슬롯 {trimmed or '없음'}{d_sum}{d_flag}")
    return (ok, len(SCENARIOS))


# ── C. input_mode 동작 ───────────────────────────────────────────────────────


def scenario_tools(sc: dict) -> list[dict]:
    """시나리오 → tool_outputs. **운영 저장소 없이** 만든다

    B 검사는 서비스의 `_prefetch`가 실제로 부른 도구를 잡아 쓰지만, C는 우리 조립기만
    보는 검사이므로 서비스가 필요 없다. 도구 인자는 서비스가 넘기는 것과 같게 채운다
    """
    url, fetch = sc["url"], sc["fetch"]
    fhost = re.sub(r"^https?://", "", fetch["final_url"]).split("/")[0]
    outs = [{"tool": "fetch_page", "args": {"url": url}, "output": json.dumps(fetch, ensure_ascii=False)}]
    if "signals" in sc:
        outs.append({"tool": "page_signals", "args": {"url": fetch["final_url"]},
                     "output": json.dumps(sc["signals"], ensure_ascii=False)})
    if "official" in sc:
        outs.append({"tool": "official_domain_of", "args": {"brand": fhost},
                     "output": json.dumps(sc["official"], ensure_ascii=False)})
    for brand, out in (sc.get("brands") or {}).items():
        outs.append({"tool": "official_domain_of", "args": {"brand": brand},
                     "output": json.dumps(out, ensure_ascii=False)})
    for key, tool, argk in (("blocklist", "check_blocklist", "host"),
                            ("cache", "lookup_cache", "site"),
                            ("age", "domain_age", "host"),
                            ("age_final", "domain_age", "host")):
        if key not in sc:
            continue
        v = sc[key]
        arg = v.get("host") or v.get("site") or v.get("domain") or ""
        outs.append({"tool": tool, "args": {argk: arg}, "output": json.dumps(v, ensure_ascii=False)})
    return outs


def test_modes() -> tuple[int, int]:
    print("\nC. input_mode · lookup_cache 제외  (운영 저장소 불필요)")
    sc = SCENARIOS["본문 3000자 초과 → fetch trim 경계"]
    captured = scenario_tools(sc)
    case = {"case_id": "m", "url": sc["url"], "input": {"url": sc["url"], "tool_outputs": captured}}

    trimmed = B.build_user_prompt(case, input_mode="production-trim", exclude=())
    raw = B.build_user_prompt(case, input_mode="raw", exclude=())
    no_cache = B.build_user_prompt(case, input_mode="production-trim")

    checks = [
        ("production-trim이 fetch를 자른다", "…(+" in trimmed),
        ("raw는 자르지 않는다", "…(+" not in raw),
        ("raw가 더 길다", len(raw) > len(trimmed)),
        ("조립 형식은 같다 (조건부 문단·마무리 동일)",
         raw.endswith("이것으로 final_verdict를 내라.") and trimmed.endswith("이것으로 final_verdict를 내라.")),
        ("기본은 lookup_cache 제외", "- lookup_cache:" not in no_cache),
        ("--include-lookup-cache 면 포함", "- lookup_cache:" in trimmed),
        ("제외해도 다른 슬롯은 그대로",
         no_cache.replace("- lookup_cache: " + json.dumps(sc["cache"], ensure_ascii=False) + "\n", "") == trimmed.replace(
             "- lookup_cache: " + json.dumps(sc["cache"], ensure_ascii=False) + "\n", "")),
    ]
    for label, got in checks:
        print(f"  {'✓' if got else '✗'} {label}")
    return (sum(c[1] for c in checks), len(checks))


def main() -> int:
    a_ok, a_n = test_real_prompts()
    b_ok, b_n = test_service_code()
    c_ok, c_n = test_modes()
    total_ok, total_n = a_ok + b_ok + c_ok, a_n + b_n + c_n
    print("\n" + "─" * 66)
    print(f"A 실제 프롬프트 {a_ok}/{a_n} · B 서비스 코드 {b_ok}/{b_n} · C 모드 {c_ok}/{c_n}"
          f"   합계 {total_ok}/{total_n}")
    if not JUDGE_SERVICE_ROOT:
        print("\nA·B는 운영 저장소가 있어야 돈다 — 조립이 서비스와 같은지 확인하려면:")
        print("  export JUDGE_SERVICE_ROOT=~/judge-service")
        print("  python -m tests.test_assembly")
    return 0 if total_ok == total_n else 1


if __name__ == "__main__":
    raise SystemExit(main())
