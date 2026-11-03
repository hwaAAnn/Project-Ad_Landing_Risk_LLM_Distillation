"""판정 기준 고정 + raw tool_outputs → user prompt 조립

세 가지 일을 한다.

| 언제 | 무엇 |
| :-- | :-- |
| **필요할 때만** (`--from-server`) | 운영 서버의 판정 기준을 뽑아 `dataset/prompt_spec.json` 재생성 |
| **매 실행** (`load_spec`) | 그 파일을 읽고 봉인(sha256) 확인 |
| **case마다** (`build_messages`) | raw `tool_outputs` → 서비스와 **같은 규칙으로** user 메시지 조립 |

## 왜 case에는 raw를 넣고 조립은 여기서 하는가

`cases.jsonl`에 조립·trim된 완성 프롬프트를 넣어 두면 입력 규칙을 바꿔 보는 실험이
불가능하다 — 3000자 자르기가 판정을 얼마나 바꾸는지 보려면 원본이 있어야 한다.
그래서 **case는 raw를 그대로 보존**하고, 자르기와 조립은 실행 시점에 여기서 한다.

## input_mode

| 모드 | 무엇 | 쓰임 |
| :-- | :-- | :-- |
| `production-trim` (기본) | 현재 서비스와 **동일** — fetch 3000자·나머지 1200자 trim + 조건부 지시문 | 서비스 baseline 재현 |
| `raw` | 같은 조립, **trim 없음** | 잘린 본문이 판정을 바꾸는지 보는 실험 |
| `stored` | case의 `input.prompt_text`를 그대로 사용 | 판정 당시 프롬프트가 통째로 저장돼 있을 때 |

## 조립 규칙의 출처

`server/judge_agent.py:_prefetch`(요약 만들기)와 `run_agent`(첫 user 메시지 조립)를 재현한다.
슬롯 순서·라벨·trim 길이·조건부 문장이 전부 그쪽에서 온 것이고, 한 글자라도 다르면
"서비스와 같은 입력"이 아니게 된다. `test_assembly.py`가 실제 서비스 출력과 대조한다.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from datetime import datetime, timezone

from pipeline import domains

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
DEFAULT_SPEC = os.path.join(HERE, "dataset", "prompt_spec.json")

# 운영 서버 소스의 위치 — **선택 사항.** 기준을 다시 뽑을 때만 쓴다.
# 판정 실행(run_eval.py)은 봉인된 prompt_spec.json만 읽으므로 이 값이 없어도 무관하다
#
#     export JUDGE_SERVER_SRC=~/judge-service/server
JUDGE_SERVER_SRC = os.path.expanduser(os.environ.get("JUDGE_SERVER_SRC", ""))

INPUT_MODES = ("production-trim", "raw", "stored")

# 판정 결과가 아닌 것만 남긴다. lookup_cache는 **과거 모델의 판정**이 새 모델 입력으로
# 흘러드는 유일한 통로 — 세 모델을 같은 조건에서 보려면 빼야 한다.
# (서비스와의 byte 일치를 검증할 때는 exclude_tools=() 로 껐다가 켠다)
DEFAULT_EXCLUDE = ("lookup_cache",)


# ── server/ 소스에서 상수 꺼내기 ──────────────────────────────────────────────


def _module(path: str) -> ast.Module:
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


def _const(tree: ast.Module, name: str):
    """모듈 최상단의 `NAME = <리터럴>` 하나를 값으로"""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"{name} 을(를) 찾지 못했다 — server/ 쪽에서 이름이 바뀌었는지 확인")


def _server_sets_temperature(server_dir: str) -> bool:
    """운영 호출부가 temperature를 지정하는지. 지정하지 않으면 실험도 지정하지 않는다 —
    임의로 0을 박으면 baseline이 서비스와 다른 설정으로 돌아간다"""
    for fn in ("judge_agent.py", "main.py"):
        with open(os.path.join(server_dir, fn), encoding="utf-8") as f:
            if "temperature" in f.read():
                return True
    return False


def extract_spec(server_dir: str = "", mode: str = "oneshot") -> dict:
    """server/main.py · judge_agent.py 에서 판정 기준 일습을 뽑아 spec dict로

    운영 서버 소스가 있어야 하는 **유일한 기능**이다. 판정 실행에는 필요 없다
    """
    server_dir = server_dir or JUDGE_SERVER_SRC
    if not server_dir:
        raise SystemExit(
            "판정 기준을 다시 뽑으려면 운영 서버 소스의 위치가 필요하다.\n"
            "  export JUDGE_SERVER_SRC=~/judge-service/server\n"
            "  python -m pipeline.build_prompt --from-server --force\n\n"
            "판정 실행에는 필요 없다 — dataset/prompt_spec.json 이 이미 봉인돼 있다.")
    for fn in ("main.py", "judge_agent.py"):
        if not os.path.exists(os.path.join(server_dir, fn)):
            raise SystemExit(f"{server_dir} 에 {fn} 이 없다 — JUDGE_SERVER_SRC 경로를 확인하라")
    main = _module(os.path.join(server_dir, "main.py"))
    agent = _module(os.path.join(server_dir, "judge_agent.py"))

    criteria = _const(main, "CRITERIA")
    if mode == "oneshot":
        rules = _const(agent, "ONESHOT_RULES")
    else:
        rules = _const(agent, "AGENT_RULES").format(max_calls=_const(agent, "MAX_TOOL_CALLS"))
    system = criteria + rules

    # 출력 스키마의 원본은 main.SCHEMA가 아니라 **final_verdict 도구의 input_schema**다.
    # 실제로 모델의 출력을 강제하는 것이 그쪽이고, 필드 순서(risk 먼저)까지 그것이 정한다
    final = next(t for t in _const(agent, "TOOLS") if t["name"] == "final_verdict")

    # 운영은 temperature를 지정하지 않는다(server/ 전체에 그 이름이 없다) → API 기본값.
    # 재현 모드에서도 **지정하지 않는 것**이 서비스와 같은 조건이다. null = 파라미터 자체를 빼고 호출.
    # 표집 난수를 없애고 싶으면 run_eval.py --temperature 0 으로 그때만 덮어쓴다
    temperature = 0.0 if _server_sets_temperature(server_dir) else None

    return {
        "spec_id": f"{mode}-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "mode": mode,
        "system": system,
        "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "output_schema": final["input_schema"],
        "params": {
            "max_tokens": _const(agent, "MAX_TOKENS"),
            "temperature": temperature,
            "temperature_note": (
                "운영(server/)이 temperature를 지정하지 않으므로 여기서도 지정하지 않는다 — "
                "provider 기본값으로 호출. 재현성이 필요한 실험은 run_eval.py --temperature 로 덮어쓴다"
                if temperature is None else "운영이 지정하는 값을 그대로 따른다"
            ),
        },
        "assembly": {
            # _prefetch의 요약 루프에 박힌 리터럴 (judge_agent.py:797) — MAX_PAGE_CHARS(본문 자르기)와
            # 다른 값이다. 본문은 fetch_page 안에서 2500자로 잘린 뒤, 그 JSON 전체가 여기서 3000자로 또 잘린다
            "trim_fetch": 3000,
            "trim_other": 1200,
            "source": "server/judge_agent.py::_prefetch + run_agent",
        },
        "source": {
            "criteria": "server/main.py::CRITERIA",
            "rules": f"server/judge_agent.py::{'ONESHOT_RULES' if mode == 'oneshot' else 'AGENT_RULES'}",
            "output_schema": "server/judge_agent.py::TOOLS[final_verdict].input_schema",
            "max_tokens": "server/judge_agent.py::MAX_TOKENS",
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }


def load_spec(path: str = DEFAULT_SPEC) -> dict:
    """고정된 spec을 읽고 봉인 확인. 해시가 어긋나면 그 자리에서 중단"""
    if not os.path.exists(path):
        raise SystemExit(
            f"판정 기준 파일이 없다: {path}\n"
            f"  python -m pipeline.build_prompt --from-server   로 최초 1회 생성하라"
        )
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    got = hashlib.sha256(spec["system"].encode("utf-8")).hexdigest()
    if got != spec.get("system_sha256"):
        raise SystemExit(
            "판정 기준이 봉인과 다르다 — 세 provider가 같은 기준을 쓴다는 보증이 깨졌다\n"
            f"  기록된 해시: {spec.get('system_sha256')}\n"
            f"  실제 해시:   {got}\n"
            "  파일을 되돌리거나, 팀 전체가 새 spec으로 다시 돌려야 한다"
        )
    return spec


# ── prefetch 요약 재조립 ─────────────────────────────────────────────────────
#
# 아래 셋은 server/judge_agent.py:_prefetch 의 것을 그대로 옮긴 것이다.
# 순서·라벨·trim 길이가 요약의 형태를 결정한다 — 한 글자라도 다르면 다른 입력이 된다

# _prefetch의 `for key, label in (...)` 루프 (judge_agent.py:796)
PREFETCH_SLOTS = (
    ("fetch", "fetch_page"),
    ("signals", "page_signals"),
    ("official", "official_domain_of(도착 도메인)"),
    ("brands", "official_domain_of(언급된 브랜드)"),
    ("blocklist", "check_blocklist"),
    ("cache", "lookup_cache"),
    ("age", "domain_age"),
    ("age_final", "domain_age(도착지)"),
)
# 본문이 있을 때는 애초에 조회하지 않는 슬롯 — 없어도 "(시간 안에 못 받음)"을 쓰지 않는다
CONDITIONAL_SLOTS = ("official", "brands")
NO_BRANDS = "(본문에 표 안의 브랜드 언급 없음)"
MISSING = "(시간 안에 못 받음)"

TRIM_FETCH = 3000
TRIM_OTHER = 1200


def trim(s: str, n: int | None) -> str:
    """server/judge_agent.py:716 `_trim` — n이 None이면 자르지 않는다(input_mode=raw)"""
    if n is None:
        return s
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n}자)"


def _text(output) -> str:
    """도구 결과를 서비스가 저장한 형태(JSON 문자열)로. dict로 풀려 있어도 받는다 —
    `record()`가 `json.dumps(result, ensure_ascii=False)` 한 것과 같은 문자열이 나온다"""
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)


def _parse(text: str) -> dict:
    try:
        got = json.loads(text)
        return got if isinstance(got, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def slots_of(tool_outputs: list[dict], url: str, oneshot: bool = True) -> dict:
    """raw tool_outputs → `_prefetch`의 `got` 딕셔너리 재구성

    tool_outputs의 **순서는 믿을 수 없다** — prefetch가 스레드 셋(page·fast·age)에서
    동시에 append하므로 실행마다 뒤섞인다(judge_agent.py:786). 그래서 순서가 아니라
    (도구 이름, 인자)로 슬롯을 가른다.

    같은 도구가 두 슬롯에 걸리는 경우가 둘 있고, 서비스와 같은 규칙으로 판별한다.
     - `domain_age`: 인자 host가 출발지 등록 도메인이면 `age`, 아니면 `age_final`
     - `official_domain_of`: 인자가 도메인 꼴이면 `official`(도착 도메인), 이름이면 `brands`
    """
    got: dict[str, str] = {}
    brand_hits: list[tuple[str, str]] = []

    for item in tool_outputs:
        if not isinstance(item, dict):
            continue
        tool = item.get("tool")
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        text = _text(item.get("output"))
        if tool == "fetch_page":
            got.setdefault("fetch", text)
        elif tool == "page_signals":
            got.setdefault("signals", text)
        elif tool == "check_blocklist":
            got.setdefault("blocklist", text)
        elif tool == "lookup_cache":
            got.setdefault("cache", text)
        elif tool == "domain_age":
            got.setdefault(_age_slot(args, url), text)
        elif tool == "official_domain_of":
            brand = str(args.get("brand", ""))
            if domains.looks_like_domain(brand):
                got.setdefault("official", text)
            else:
                brand_hits.append((brand, text))

    signals = _parse(got.get("signals", ""))

    # 브랜드 조회는 sg["brand_mentions"] 순서대로 최대 6개 (judge_agent.py:772).
    # 그 순서를 되살린다 — tool_outputs의 등장 순서는 스레드 때문에 믿을 수 없다
    if brand_hits:
        mentions = [b for b in (signals.get("brand_mentions") or []) if isinstance(b, str)][:6]
        by_name = dict(brand_hits)
        ordered = [(b, by_name[b]) for b in mentions if b in by_name]
        ordered += [(b, t) for b, t in brand_hits if b not in mentions]
        got["brands"] = "\n".join(f"  - {b}: {t}" for b, t in ordered[:6])
    elif oneshot and got.get("signals"):
        # 브랜드 조회가 하나도 없다 = 본문에 표 안의 브랜드가 없었다는 뜻.
        # 서비스는 이때도 슬롯을 채운다 — 빈 줄이 아니라 "없음"이라고 적어 준다
        got["brands"] = NO_BRANDS

    return got


def _age_slot(args: dict, url: str) -> str:
    """domain_age 호출이 출발지의 것인지 도착지의 것인지"""
    site = domains.site_of(domains.host_of(url) or url) or ""
    asked = str(args.get("host", ""))
    return "age" if (domains.site_of(asked) or asked) == site else "age_final"


def prefetch_flags(got: dict) -> dict:
    """`_prefetch`가 요약과 함께 돌려주는 판단 셋 (judge_agent.py:801)

    본문을 못 읽었는지(js 렌더링인지 4xx인지), 도착 도메인이 공식 도메인 표에 있는지 —
    이 셋이 첫 메시지의 **조건부 지시문**을 고른다
    """
    fetch = _parse(got.get("fetch", ""))
    signals = _parse(got.get("signals", ""))
    if not got.get("signals") or "error" in fetch:
        # page() 스레드가 signals까지 못 갔다 — unreadable 판단 자체가 없었던 것
        # (judge_agent.py:743·750에서 return으로 빠져나가는 경로)
        return {"js_only": False, "unreadable": None, "official_known": False}
    status = fetch.get("status") or 0
    js_only = bool(signals.get("js_only"))
    if not (js_only or status >= 400):
        return {"js_only": False, "unreadable": None, "official_known": False}
    # 4xx가 우선 — 404의 빈 본문을 "자바스크립트 페이지"로 설명하면 틀린다 (judge_agent.py:769)
    unreadable = f"http {status}" if status >= 400 else "js"
    return {
        "js_only": unreadable == "js",
        "unreadable": unreadable,
        "official_known": _parse(got.get("official", "")).get("known") is True,
    }


def summary_of(got: dict, *, trim_fetch: int | None, trim_other: int | None,
               exclude: tuple[str, ...] = ()) -> str:
    """`_prefetch`의 요약 문자열 (judge_agent.py:795-800)"""
    skip = {"cache"} if "lookup_cache" in exclude else set()
    if "domain_age" in exclude:
        skip |= {"age", "age_final"}
    if "check_blocklist" in exclude:
        skip |= {"blocklist"}
    lines = []
    for key, label in PREFETCH_SLOTS:
        if key in skip:
            continue
        if key in got:
            lines.append(f"- {label}: {trim(got[key], trim_fetch if key == 'fetch' else trim_other)}")
        elif key not in CONDITIONAL_SLOTS:
            lines.append(f"- {label}: {MISSING}")
    return "\n".join(lines)


def assemble_user(url: str, click_url: str | None, summary: str, flags: dict,
                  oneshot: bool = True) -> str:
    """첫 user 메시지 (server/judge_agent.py:988-1013 `run_agent`의 `first`)

    조건부 문장 셋이 붙는다.
     - js_only  본문이 빈 것은 JS 렌더링 때문이지 접속 실패가 아니라는 설명
     - unreadable(4xx)  서버 IP가 막혔을 수 있다는 설명
     - unreadable + 공식 도메인 확인 여부에 따른 지시
    """
    first = f"광고를 눌러 도착한 주소: {url}"
    if click_url and click_url != url:
        first += f"\n누른 광고의 링크: {click_url}"
    first += "\n\n## 미리 모아 둔 것\n" + summary
    if flags.get("js_only"):
        # oneshot일 때 뒤에 붙는 것은 빈 문자열 — 그래서 문장이 **공백으로 끝난다.**
        # 서비스와 byte 단위로 같으려면 이 공백까지 같아야 한다 (judge_agent.py:996)
        first += ("\n\n본문이 비어 있는 것은 페이지를 자바스크립트로 그리기 때문이다 — 접속 실패도 빈 페이지도 아니다. "
                  + ("" if oneshot else "fetch_page를 다시 쓰지 마라(같은 결과가 온다)."))
    elif flags.get("unreadable"):
        first += (f"\n\n페이지를 받지 못했다({flags['unreadable']}) — 이 서버의 접속을 막았을 수 있다. "
                  "본문 없이 판단해야 한다.")
    if flags.get("unreadable"):
        if flags.get("official_known"):
            first += ("\n도착 도메인이 공식 도메인 표에서 확인됐으니 저위험이다. reason에는 어느 회사의 공식 사이트인지 쓰고, "
                      "evidence에는 위 official_domain_of 결과의 note를 그대로 옮겨라.")
        else:
            first += ("\n도착 도메인은 공식 도메인 표에 없다(사칭의 근거는 아니다). page_signals·차단 목록·등록일로 판단하고, "
                      "위험 근거가 없으면 확인 불가(중위험)로 둔다.")
    first += ("\n\n이것으로 final_verdict를 내라." if oneshot
              else "\n\n충분하면 바로 final_verdict. 더 봐야 할 것이 있을 때만 도구를 써라.")
    return first


def build_user_prompt(case: dict, *, input_mode: str = "production-trim",
                      exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
                      oneshot: bool = True) -> str:
    """case 하나 → 모델에 넣을 user 메시지"""
    inp = case.get("input") or {}
    if input_mode == "stored":
        stored = inp.get("prompt_text")
        if not isinstance(stored, str) or not stored.strip():
            raise ValueError(f"case {case.get('case_id')}: input_mode=stored 인데 prompt_text가 없다")
        return stored

    url = inp.get("url") or case.get("url")
    if not url:
        raise ValueError(f"case {case.get('case_id')}: url 이 없다 (input.url 또는 최상위 url)")
    tool_outputs = inp.get("tool_outputs")
    if not isinstance(tool_outputs, list) or not tool_outputs:
        raise ValueError(f"case {case.get('case_id')}: input.tool_outputs 가 비어 있다")

    got = slots_of(tool_outputs, url, oneshot=oneshot)
    flags = prefetch_flags(got)
    if input_mode == "raw":
        tf = to = None            # 자르지 않는다 — 조립 형식은 그대로
    else:
        tf, to = TRIM_FETCH, TRIM_OTHER
    summary = summary_of(got, trim_fetch=tf, trim_other=to, exclude=exclude)
    return assemble_user(url, inp.get("click_url") or case.get("click_url"),
                         summary, flags, oneshot=oneshot)


# ── 진단 (dry-run) ───────────────────────────────────────────────────────────


# 서비스가 실제로 부르는 도구. 이 밖의 이름은 어느 슬롯에도 못 들어가고 조용히 버려진다
KNOWN_TOOLS = ("fetch_page", "page_signals", "check_blocklist", "lookup_cache",
               "official_domain_of", "domain_age")


def slot_report(tool_outputs: list[dict], url: str, oneshot: bool = True) -> dict:
    """조립 전에 case의 상태를 본다 — 무엇이 어느 자리에 들어갔고, 무엇이 못 들어갔는가

    `slots_of`는 `setdefault`로 **먼저 온 것만** 담는다(서비스의 got 딕셔너리와 같다).
    그래서 같은 자리에 둘이 오면 뒤엣것이 버려지는데, 그것이 조용히 일어나면
    "왜 도구 결과가 프롬프트에 없지?"를 나중에 추적하기 어렵다. 여기서 미리 드러낸다
    """
    got = slots_of(tool_outputs, url, oneshot=oneshot)
    unknown: list[str] = []
    parse_failed: list[str] = []
    landed: dict[str, int] = {}

    for item in tool_outputs:
        if not isinstance(item, dict):
            unknown.append(str(type(item).__name__))
            continue
        tool = item.get("tool")
        if tool not in KNOWN_TOOLS:
            unknown.append(str(tool))
            continue
        text = _text(item.get("output"))
        if not _parse(text):
            # 도구 결과는 전부 JSON 객체다. 파싱이 안 되면 플래그 판단(js_only·status·known)이
            # 통째로 빠져 조건부 지시문이 사라진다 — 프롬프트가 조용히 달라지는 경로
            parse_failed.append(f"{tool}: {text[:60]}")
            continue
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        if tool == "domain_age":
            slot = _age_slot(args, url)
        elif tool == "official_domain_of":
            slot = "official" if domains.looks_like_domain(str(args.get("brand", ""))) else "brands"
        else:
            slot = {"fetch_page": "fetch", "page_signals": "signals",
                    "check_blocklist": "blocklist", "lookup_cache": "cache"}[tool]
        landed[slot] = landed.get(slot, 0) + 1

    # brands는 여러 건이 한 자리에 모이는 것이 정상 — 겹침으로 세지 않는다
    collisions = [f"{slot}×{n}" for slot, n in sorted(landed.items())
                  if n > 1 and slot != "brands"]
    return {
        "slots": sorted(got),
        "missing_core": [k for k in ("fetch", "signals") if k not in got],
        "unknown_tools": sorted(set(unknown)),
        "parse_failed": parse_failed,
        "collisions": collisions,
    }


def trim_report(tool_outputs: list[dict], url: str, oneshot: bool = True) -> dict:
    """production-trim이 어느 슬롯을 자르는지. 자르기가 판정에 영향을 주는 case를 세는 용도"""
    got = slots_of(tool_outputs, url, oneshot=oneshot)
    cut: dict[str, tuple[int, int]] = {}
    for key, _label in PREFETCH_SLOTS:
        if key not in got:
            continue
        limit = TRIM_FETCH if key == "fetch" else TRIM_OTHER
        n = len(got[key])
        if n > limit:
            cut[key] = (n, n - limit)      # (원본 길이, 잘려 나간 글자 수)
    return cut


def build_messages(case: dict, spec: dict, *, input_mode: str = "production-trim",
                   exclude: tuple[str, ...] = DEFAULT_EXCLUDE) -> tuple[str, str]:
    """case 하나를 (system, user) 한 쌍으로"""
    oneshot = spec.get("mode", "oneshot") == "oneshot"
    return spec["system"], build_user_prompt(case, input_mode=input_mode,
                                             exclude=exclude, oneshot=oneshot)


def main() -> int:
    ap = argparse.ArgumentParser(description="server/의 판정 기준을 뽑아 prompt_spec.json 고정")
    ap.add_argument("--from-server", action="store_true", help="server/ 소스에서 추출 (최초 1회)")
    ap.add_argument("--server", default="", help="운영 서버 소스 경로 (기본: $JUDGE_SERVER_SRC)")
    ap.add_argument("--mode", choices=["oneshot", "agent"], default="oneshot")
    ap.add_argument("--out", default=DEFAULT_SPEC)
    ap.add_argument("--force", action="store_true", help="이미 있는 spec을 덮어쓴다")
    args = ap.parse_args()

    if not args.from_server:
        spec = load_spec(args.out)
        print(f"{args.out}\n  spec_id {spec['spec_id']} · system {len(spec['system'])}자 "
              f"· sha256 {spec['system_sha256'][:16]}… · 봉인 확인")
        print(f"  params {spec['params']['max_tokens']}토큰 · temperature "
              f"{spec['params']['temperature'] if spec['params']['temperature'] is not None else '미지정(운영과 동일)'}")
        return 0

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"이미 있다: {args.out}\n  정말 새로 뽑으려면 --force (기존 결과와 비교 불가해진다)")

    spec = extract_spec(args.server, args.mode)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"만들었다: {args.out}")
    print(f"  spec_id       {spec['spec_id']}")
    print(f"  system        {len(spec['system'])}자  (CRITERIA + {spec['source']['rules'].split('::')[1]})")
    print(f"  system_sha256 {spec['system_sha256']}")
    print(f"  output_schema {list(spec['output_schema']['properties'])}")
    print(f"  max_tokens    {spec['params']['max_tokens']}")
    print(f"  temperature   {spec['params']['temperature']}  ← {spec['params']['temperature_note']}")
    print("\n이 파일을 커밋해 고정하라 — 세 provider가 같은 기준을 쓴다는 근거다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
