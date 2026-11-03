"""Gemini 어댑터

구조화 출력은 `response_schema` + `application/json` mime type. Claude의 도구 강제와
달리 **본문이 JSON 문자열로** 오므로 여기서 파싱한다 — 파싱 실패도 스키마 준수 실패의
한 형태이므로 숨기지 않고 error로 남긴다.

system 프롬프트는 `system_instruction`으로 전달한다. user 메시지에 합치면 세 모델이
서로 다른 구조의 입력을 받게 되어 비교가 뒤틀린다 — 셋 다 "system 따로, user 따로"다.
"""

from __future__ import annotations

import json
import os

from .base import Request, Response, require_env, require_sdk, to_gemini_schema

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")


class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        genai = require_sdk("google.genai", "google-genai", "gemini")
        # GEMINI_API_KEY 우선, 없으면 GOOGLE_API_KEY — SDK 관례가 둘 다 쓴다
        key = os.environ.get("GEMINI_API_KEY") or require_env("GOOGLE_API_KEY", "gemini")
        self._genai = genai
        self._client = genai.Client(api_key=key)

    def default_model(self) -> str:
        return DEFAULT_MODEL

    def generate(self, req: Request) -> Response:
        from google.genai import types
        resp = self._client.models.generate_content(
            model=req.model,
            contents=req.user,
            config=types.GenerateContentConfig(
                system_instruction=req.system,
                # None이면 넘기지 않는다 — SDK가 None을 "미지정"으로 다룬다
                temperature=req.temperature,
                max_output_tokens=req.max_tokens,
                response_mime_type="application/json",
                response_schema=to_gemini_schema(req.schema),
            ),
        )
        u = getattr(resp, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(u, "prompt_token_count", None),
            "output_tokens": getattr(u, "candidates_token_count", None),
        }
        text = getattr(resp, "text", None)
        if not text:
            # 안전 필터·토큰 소진 등으로 본문이 없을 수 있다. 원인을 남겨야 나중에 세는 것이 가능
            reason = getattr(getattr(resp, "candidates", [None])[0], "finish_reason", None)
            return Response(usage=usage, error=f"본문 없음 (finish_reason={reason})")
        try:
            return Response(verdict=json.loads(text), usage=usage)
        except json.JSONDecodeError as e:
            return Response(usage=usage, error=f"JSON 파싱 실패: {e} — {text[:120]}")
