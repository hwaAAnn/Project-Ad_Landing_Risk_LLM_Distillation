"""case · result 스키마 한 곳

세 사람이 각자 파일을 쓰므로, 무엇이 필수이고 무엇이 유효한 값인지는 **여기 하나**에만 둔다.
provider별 파일이 각자 검증하면 "Gemini 결과에만 있는 필드" 같은 것이 생겨 병합이 깨진다
"""

from __future__ import annotations

import json
from typing import Any, Iterator

RISKS = ("LOW", "MEDIUM", "HIGH")
VERDICT_FIELDS = ("risk", "type", "reason", "advice", "evidence")


# ── case (입력) ──────────────────────────────────────────────────────────────


def inspect_cases(path: str, input_mode: str = "production-trim") -> tuple[list[dict], list[str]]:
    """cases.jsonl 읽기 — 문제를 **모아서** 돌려준다 (첫 건에서 죽지 않는다)

    dry-run이 100건 전체의 상태를 한 번에 보여 주려면 첫 오류에서 멈추면 안 된다.
    실제 실행(`load_cases`)은 반대로 하나라도 어긋나면 즉시 중단한다 — 조용히 빠진
    3건 때문에 provider별 건수가 어긋나는 것이 더 나쁘기 때문
    """
    cases: list[dict] = []
    problems: list[str] = []
    seen: set[str] = set()
    for i, line in enumerate(_lines(path), 1):
        try:
            case = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"{i}행 JSON 파싱 실패 — {e}")
            continue
        cid = case.get("case_id")
        if not isinstance(cid, str) or not cid.strip():
            problems.append(f"{i}행 case_id 가 없다 — 병합 키라 반드시 있어야 한다")
            continue
        if cid in seen:
            problems.append(f"{i}행 case_id 중복: {cid} — 병합 시 결과가 덮어써진다")
            continue
        seen.add(cid)
        inp = case.get("input") or {}
        if input_mode == "stored":
            if not isinstance(inp.get("prompt_text"), str) or not inp["prompt_text"].strip():
                problems.append(f"{i}행 ({cid}) input_mode=stored 인데 input.prompt_text 가 없다")
                continue
        else:
            if not (inp.get("url") or case.get("url")):
                problems.append(f"{i}행 ({cid}) input.url 이 없다 — 조립의 첫 줄에 들어간다")
                continue
            tools = inp.get("tool_outputs")
            if not isinstance(tools, list) or not tools:
                problems.append(f"{i}행 ({cid}) input.tool_outputs 가 비어 있다")
                continue
            bad = [t for t in tools if not isinstance(t, dict) or "tool" not in t or "output" not in t]
            if bad:
                problems.append(f"{i}행 ({cid}) tool_outputs 항목에 tool·output 이 없다: {str(bad[0])[:80]}")
                continue
        cases.append(case)
    return cases, problems


def load_cases(path: str, input_mode: str = "production-trim") -> list[dict]:
    """cases.jsonl 읽기 + 검증. 잘못된 줄은 **건너뛰지 않고** 즉시 중단 —
    100건 중 3건이 조용히 빠지면 provider별 건수가 어긋나 병합에서 사라진다

    필요한 필드는 input_mode에 따라 다르다.
     - production-trim · raw: `input.url` + `input.tool_outputs` (raw 원본. trim은 실행 시)
     - stored:             `input.prompt_text` (판정 당시 프롬프트가 통째로 있을 때)
    """
    cases, problems = inspect_cases(path, input_mode)
    if problems:
        raise SystemExit(
            f"{path} 에 문제가 있다 ({len(problems)}건) — 먼저 --dry-run 으로 전체를 확인하라\n  "
            + "\n  ".join(problems[:10])
            + (f"\n  … 외 {len(problems) - 10}건" if len(problems) > 10 else ""))
    if not cases:
        raise SystemExit(f"{path} 에 case가 없다")
    return cases


def _lines(path: str) -> Iterator[str]:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line
    except FileNotFoundError:
        raise SystemExit(f"입력 파일이 없다: {path}")


def tool_outputs_of(case: dict, exclude: tuple[str, ...] = ()) -> list[dict]:
    """근거 검증(_ground)의 대조 원문. 없으면 빈 목록 — 강등을 적용하지 않는 신호

    exclude는 프롬프트에서 뺀 도구와 **같아야 한다.** 모델에게 안 보여 준 lookup_cache를
    근거 대조에는 쓰면, 모델이 볼 수 없었던 문장을 근거로 인정하는 꼴이 된다
    """
    got = (case.get("input") or {}).get("tool_outputs")
    if not isinstance(got, list):
        return []
    return [t for t in got if isinstance(t, dict) and t.get("tool") not in exclude]


# ── verdict (모델 출력) ──────────────────────────────────────────────────────


def validate_verdict(v: Any, output_schema: dict) -> dict:
    """모델이 돌려준 것이 약속한 스키마인지. 아니면 ValueError

    세 모델이 각자 네이티브 방식(tool_choice · response_schema · json_schema)으로
    스키마를 강제하지만 강제의 강도가 다르다 — 여기서 같은 잣대로 한 번 더 본다.
    스키마 준수 실패율 자체가 비교 항목이므로 실패는 숨기지 않고 error로 기록한다
    """
    if not isinstance(v, dict):
        raise ValueError(f"객체가 아니다: {type(v).__name__}")
    props = output_schema.get("properties", {})
    missing = [k for k in VERDICT_FIELDS if k not in v]
    if missing:
        raise ValueError(f"필드 누락: {missing}")
    out = {}
    for k in VERDICT_FIELDS:
        val = v[k]
        if not isinstance(val, str):
            raise ValueError(f"{k} 가 문자열이 아니다: {type(val).__name__}")
        enum = (props.get(k) or {}).get("enum")
        if enum and val not in enum:
            raise ValueError(f"{k}={val!r} 는 허용값이 아니다: {enum}")
        out[k] = val
    extra = [k for k in v if k not in VERDICT_FIELDS]
    if extra:
        # 버리되 조용히 버리지 않는다 — 모델이 스키마 밖 필드를 붙이는 성향도 비교 대상
        out["_extra_keys"] = sorted(extra)
    return out


# ── result (출력) ────────────────────────────────────────────────────────────


def make_result(*, case_id: str, run_id: str, provider: str, model: str, spec: dict,
                input_mode: str, excluded: tuple[str, ...], temperature: float | None,
                max_tokens: int | None = None,
                prompt_sha256: str, verdict_raw: dict | None, verdict: dict | None,
                postprocess: dict, latency_ms: int, usage: dict, attempts: int,
                error: str | None) -> dict:
    """result 한 줄. 성공·실패 **모두** 이 형태로 남긴다

    실패를 빼먹으면 표본이 편향된다 — "Gemini가 5건에서 스키마를 못 지켰다"는 사실이
    파일에서 사라지고, 남은 95건만으로 계산한 일치율이 공정해 보이게 된다
    """
    return {
        "case_id": case_id,
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "spec_id": spec["spec_id"],
        "system_sha256": spec["system_sha256"],
        # 어떤 입력으로 물었는지. 나중에 production-trim ↔ raw 결과를 섞어 보지 않도록,
        # 그리고 세 사람이 정말 같은 프롬프트를 넣었는지 확인할 수 있도록 남긴다
        "input_mode": input_mode,
        "excluded_tools": list(excluded),
        "temperature": temperature,
        # 어느 출력 한도로 쟀는지. 추론 모델은 이 값이 모자라면 추론에만 예산을 다 쓰고
        # 본문을 못 낸다 — "한도 부족"과 "모델이 못 함"을 사후에 구분하려면 남아 있어야 한다
        "max_tokens": max_tokens,
        "prompt_sha256": prompt_sha256,
        "ok": error is None and verdict is not None,
        "verdict_raw": verdict_raw,
        "verdict": verdict,
        "postprocess": postprocess,
        "latency_ms": latency_ms,
        "usage": usage,
        "attempts": attempts,
        "error": error,
    }
