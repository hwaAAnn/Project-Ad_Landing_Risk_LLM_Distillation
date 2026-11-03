"""서비스 후처리 재현 — `_fill_type` · `_ground`

운영에서 모델의 출력은 그대로 화면에 가지 않는다. 두 겹을 더 지난다.

| 순서 | 무엇 | 원본 |
| :-- | :-- | :-- |
| 1 | `_fill_type` — 비어 온 유형을 모델 자신의 reason 문장에서 역추적 | `server/main.py:1281` |
| 2 | `_ground` — 고위험의 evidence가 도구 결과에 실재하지 않으면 중위험으로 강등 | `server/judge_agent.py:1097` |

세 모델을 "모델 그대로" 비교할지 "서비스에 넣었을 때"로 비교할지는 나중에 갈릴 논쟁이므로,
`run_eval.py`는 `verdict_raw`(원본)와 `verdict`(후처리 후)를 **둘 다** 남긴다.

## 왜 import가 아니라 복사인가

`server/main.py` · `judge_agent.py`는 import 시점에 Anthropic 클라이언트 · FastAPI app ·
requests 세션 · RDAP 스레드풀을 만든다. 순수 함수 셋을 쓰자고 그것을 전부 올릴 이유가 없고,
"실험 코드에는 페이지 수집기가 없다"는 보증도 깨진다. 원본이 바뀌면 여기도 손으로 맞춘다 —
아래 주석의 출처 줄번호가 그 대조표다.

## _ground의 대조 원문(haystack)

운영에서는 방금 돌린 도구의 결과가 대조 원문이다. 여기서는 **case에 저장된
`input.tool_outputs`**를 쓴다 — 판정 당시 원문 그대로이므로 같은 결과가 재현된다.
도구를 다시 돌리지 않는다. tool_outputs가 없는 case는 대조할 것이 없으므로
**강등을 적용하지 않고** `ground_downgraded: None`으로 남긴다 (조용히 통과시키지 않는다)
"""

from __future__ import annotations

import json
import re

# ── server/main.py:1281 _fill_type 의 표 ─────────────────────────────────────
_TYPE_TABLE = [
    ("impersonation", ("사칭", "가장한", "위장")),
    ("credentials", ("비밀번호", "주민등록번호", "주민번호", "카드번호")),
    ("apk", ("apk", "설치 파일", "앱 파일")),
    ("personal_info", ("개인정보", "전화번호", "연락처", "이름을 입력", "상담 신청")),
    ("payment", ("결제", "구매", "가입을 유도", "사업자 정보")),
    ("urgency", ("당첨", "무료", "한정", "경품", "이벤트 응모")),
    ("investment", ("투자", "수익", "재테크", "부업")),
    ("contentfarm", ("콘텐츠팜", "미끼", "광고성 글")),
    ("unverifiable", ("확인하지 못", "확인할 수 없", "가져오지 못")),
]


def fill_type(v: dict) -> tuple[dict, bool]:
    """server/main.py:1281 `_fill_type` — 비어 온 유형을 reason 문장에서 역추적

    추측이 아니라 **모델 자신이 쓴 문장**에서 고르므로 새 사실을 만들지 않는다
    """
    if v.get("risk") == "LOW":
        return ({**v, "type": "none"}, v.get("type") != "none")
    if v.get("type") and v["type"] != "none":
        return (v, False)
    reason = v.get("reason", "")
    for kind, words in _TYPE_TABLE:
        if any(w in reason for w in words):
            return ({**v, "type": kind}, True)
    return ({**v, "type": v.get("type") or "none"}, False)


def _tokens_grounded(ev: str, hay: str) -> bool:
    """server/judge_agent.py:1119 `_tokens_grounded`"""
    toks = re.findall(r"[a-z0-9][a-z0-9.\-]{3,}|\d+", ev.lower())
    toks = [t for t in toks if len(t) >= 2]
    if not toks:
        return False
    low = hay.lower()
    return all(t in low for t in toks)


def ground(v: dict, outputs: list[str]) -> tuple[dict, bool | None]:
    """server/judge_agent.py:1097 `_ground` — 근거 없는 고위험은 중위험으로

    반환의 둘째 값: True 강등함 / False 통과 / None 대조 원문이 없어 판정 보류
    """
    v = {"risk": v.get("risk", "UNKNOWN"), "type": v.get("type") or "none",
         "reason": v.get("reason", ""), "advice": v.get("advice", ""),
         "evidence": v.get("evidence", "")}
    if v["risk"] == "LOW":
        v["type"] = "none"
    if v["risk"] != "HIGH":
        return (v, False)
    if not outputs:
        # 대조할 원문이 없다. 강등하면 "근거가 없어서"가 아니라 "데이터가 없어서" 내려간 것이 되고,
        # 그 case만 세 모델 모두 HIGH를 못 받게 되어 비교가 뒤틀린다
        return (v, None)
    ev = re.sub(r"\s+", " ", v["evidence"]).strip()
    hay = re.sub(r"\s+", " ", " ".join(outputs))
    if ev and (ev in hay or _tokens_grounded(ev, hay)):
        return (v, False)
    return ({**v, "risk": "MEDIUM"}, True)


def haystack(tool_outputs: list[dict]) -> list[str]:
    """case의 tool_outputs → `_ground`가 볼 텍스트 목록

    운영의 `outputs` 리스트와 같은 것 — 도구가 돌려준 텍스트 전문(judge_agent.py:905).
    `output`이 문자열(JSON dumps)로 저장된 형태와, 이미 dict로 풀린 형태 둘 다 받는다
    """
    out: list[str] = []
    for item in tool_outputs:
        if not isinstance(item, dict):
            continue
        o = item.get("output")
        if isinstance(o, str):
            out.append(o)
        elif o is not None:
            out.append(json.dumps(o, ensure_ascii=False))
    return out


def apply(verdict_raw: dict, tool_outputs: list[dict]) -> tuple[dict, dict]:
    """운영과 같은 순서로 후처리. (verdict, 무엇이 일어났는지)"""
    v, filled = fill_type(verdict_raw)
    v, downgraded = ground(v, haystack(tool_outputs))
    return v, {"fill_type_applied": filled, "ground_downgraded": downgraded}
