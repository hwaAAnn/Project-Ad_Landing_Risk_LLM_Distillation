"""provider 공통 계약 — 세 사람이 같은 인터페이스를 쓰게 하는 곳

각자 한 모델씩 맡아 돌리므로, 반환 형태가 조금이라도 어긋나면 병합에서 드러난다.
그래서 **스키마 변환도 여기 한 곳**에 둔다 — providers/*.py는 SDK 호출만 한다.

## 같은 스키마, 세 가지 강제 방식

세 모델은 구조화 출력을 강제하는 방법이 다르다. 스키마는 하나(prompt_spec.output_schema)이고,
그것을 각 SDK가 알아듣는 형태로 바꾸는 함수만 다르다.

| provider | 강제 방식 | 변환 |
| :-- | :-- | :-- |
| claude | `tools=[final_verdict]` + `tool_choice` 강제 | 원본 그대로 (운영과 동일) |
| gemini | `response_schema` + JSON mime type | type 대문자화 · 미지원 키 제거 |
| openai | `response_format: json_schema` (strict) | `additionalProperties:false` · 전 필드 required |
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Request:
    """모델에 보내는 것 전부. 세 provider가 똑같이 받는다"""
    system: str
    user: str
    schema: dict
    model: str
    max_tokens: int = 450
    # None = **지정하지 않는다.** 운영(server/)이 temperature를 넘기지 않으므로 baseline도 그래야 한다 —
    # 임의로 0을 박으면 서비스와 다른 설정으로 잰 결과가 된다. 재현성 실험은 run_eval.py --temperature 로
    temperature: float | None = None


@dataclass
class Response:
    """모델이 돌려준 것. verdict는 **검증 전 원본** — 검증은 run_eval이 schema.py로"""
    verdict: dict | None = None
    usage: dict = field(default_factory=dict)
    error: str | None = None


class Provider(Protocol):
    name: str

    def default_model(self) -> str: ...

    def generate(self, req: Request) -> Response: ...


# ── 환경변수 ─────────────────────────────────────────────────────────────────


def require_env(var: str, provider: str) -> str:
    """API 키는 환경변수에서만. 코드·DB·파일 어디에도 두지 않는다"""
    val = os.environ.get(var)
    if not val:
        raise SystemExit(
            f"{provider}: 환경변수 {var} 가 없다\n"
            f"  export {var}=...   (키를 파일에 적지 마라 — 결과·로그에도 남기지 않는다)"
        )
    return val


def require_sdk(module: str, pip_name: str, provider: str):
    """SDK는 **맡은 것만** 깔면 되게 한다 — Gemini 담당자에게 anthropic 설치를 시키지 않는다"""
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError:
        raise SystemExit(
            f"{provider}: {pip_name} 패키지가 없다\n"
            f"  pip install {pip_name}   (또는 pip install -r requirements.txt)"
        )


# ── 스키마 변환 ──────────────────────────────────────────────────────────────


def to_anthropic_tool(schema: dict, name: str = "final_verdict") -> dict:
    """운영이 쓰는 형태 그대로. 도구 하나 + tool_choice 강제"""
    return {
        "name": name,
        "description": "최종 판정. evidence는 주어진 결과에서 그대로 옮긴 짧은 구절이어야 한다.",
        "input_schema": schema,
    }


# Gemini(OpenAPI 3.0 부분집합)가 모르는 키. 남기면 400
_GEMINI_DROP = {"additionalProperties", "$schema", "title", "default", "examples"}


def to_gemini_schema(schema: dict) -> dict:
    """JSON Schema → Gemini `response_schema`

    type 값은 대문자여야 한다(OBJECT·STRING). 소문자를 넣으면 SDK의 enum 검증에서 걸린다.
    propertyOrdering으로 필드 순서를 원본과 맞춘다 — 순서가 출력 품질에 영향을 준다는
    보고가 있고, 무엇보다 세 모델의 출력을 눈으로 비교할 때 같은 순서가 편하다
    """
    def conv(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in _GEMINI_DROP:
                continue
            if k == "type" and isinstance(v, str):
                out["type"] = v.upper()
            elif k == "properties" and isinstance(v, dict):
                out["properties"] = {pk: conv(pv) for pk, pv in v.items()}
                out["propertyOrdering"] = list(v)
            elif k == "items":
                out["items"] = conv(v)
            else:
                out[k] = v
        return out

    return conv(schema)


def to_openai_schema(schema: dict, name: str = "final_verdict") -> dict:
    """JSON Schema → OpenAI `response_format: json_schema` (strict)

    strict 모드는 모든 객체에 `additionalProperties:false`, 그리고 **모든 속성이
    required**일 것을 요구한다. 원본은 5개 전부 required라 그대로 맞지만,
    나중에 선택 필드가 붙어도 깨지지 않게 여기서 채운다
    """
    def conv(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        out = dict(node)
        if out.get("type") == "object":
            props = out.get("properties") or {}
            out["properties"] = {k: conv(v) for k, v in props.items()}
            out["required"] = list(props)
            out["additionalProperties"] = False
        if "items" in out:
            out["items"] = conv(out["items"])
        return out

    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": conv(schema)},
    }


# ── registry ─────────────────────────────────────────────────────────────────

PROVIDER_NAMES = ("claude", "gemini", "openai")


def get_provider(name: str) -> Provider:
    """--provider 이름 → 어댑터. SDK import는 **고른 것만** —
    Gemini 담당자의 PC에 anthropic 패키지가 없어도 돌아야 한다"""
    if name == "claude":
        from .claude import ClaudeProvider
        return ClaudeProvider()
    if name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider()
    if name == "openai":
        from .openai import OpenAIProvider
        return OpenAIProvider()
    raise SystemExit(f"모르는 provider: {name} (가능: {', '.join(PROVIDER_NAMES)})")
