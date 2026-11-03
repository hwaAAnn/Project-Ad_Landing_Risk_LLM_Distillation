"""도메인 조각내기 — `server/judge_agent.py`의 순수 함수 복사본

prefetch 요약을 재조립하려면 "이 domain_age 호출이 출발지의 것인가 도착지의 것인가",
"이 official_domain_of 호출이 도메인으로 물은 것인가 브랜드 이름으로 물은 것인가"를
서비스와 **같은 규칙으로** 갈라야 한다. 그 판별에 쓰이는 함수 셋이다.

import하지 않고 복사한 이유는 postprocess.py와 같다 — `import judge_agent`는 로드 시점에
requests 세션과 RDAP 스레드풀을 만든다(judge_agent.py:87·294). 원본이 바뀌면 여기도
손으로 맞춘다. 아래 출처 줄번호가 대조표다.
"""

from __future__ import annotations

import re


def host_of(url: str) -> str | None:
    """server/judge_agent.py:255"""
    m = re.match(r"https?://([^/?#]+)", url.lower())
    return m.group(1).split(":")[0].strip(".") if m else None


def site_of(host_or_url: str) -> str | None:
    """server/judge_agent.py:260 — 등록 도메인 (co.kr류는 세 조각)"""
    host = host_of(host_or_url) if "://" in host_or_url else host_or_url.lower()
    if not host:
        return None
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    two_level = {"co", "or", "ne", "go", "ac", "re", "pe"}
    take = 3 if len(parts[-1]) <= 3 and parts[-2] in two_level else 2
    return ".".join(parts[-take:])


def looks_like_domain(q: str) -> bool:
    """server/judge_agent.py:236 `_looks_like_domain`

    official_domain_of를 **도메인으로** 물었는지 **브랜드 이름으로** 물었는지 가른다 —
    서비스에서 이 분기가 요약의 라벨('도착 도메인' ↔ '언급된 브랜드')을 결정한다
    """
    q = q.strip().lower()
    return "://" in q or bool(re.fullmatch(r"[a-z0-9.\-]+\.[a-z]{2,}(/.*)?", q))
