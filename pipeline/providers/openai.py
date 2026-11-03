"""OpenAI 어댑터

구조화 출력은 `response_format: json_schema` (strict). Gemini와 마찬가지로 본문이
JSON 문자열로 오므로 여기서 파싱한다.

## temperature 주의

기본은 **미지정**이다 (운영이 그러하므로). `--temperature 0` 같이 값을 준 경우에만 넘기는데,
일부 추론 모델(gpt-5 계열)은 1 이외의 값을 받지 않는다 — 거부당하면 파라미터를 빼고 한 번 더
부르고, 그 사실을 usage에 `temperature_rejected`로 남긴다. 그 모델만 표집 난수가 섞였다는
것을 결과에서 알 수 있어야 하기 때문이다
"""

from __future__ import annotations

import json
import os

from .base import Request, Response, require_env, require_sdk, to_openai_schema

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        openai = require_sdk("openai", "openai", "openai")
        self._client = openai.OpenAI(api_key=require_env("OPENAI_API_KEY", "openai"),
                                     max_retries=0)

    def default_model(self) -> str:
        return DEFAULT_MODEL

    def generate(self, req: Request) -> Response:
        kw = dict(
            model=req.model,
            max_completion_tokens=req.max_tokens,
            response_format=to_openai_schema(req.schema),
            messages=[
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
        )
        rejected = False
        if req.temperature is None:
            # 운영과 같은 조건 — 파라미터를 아예 넘기지 않는다
            resp = self._client.chat.completions.create(**kw)
        else:
            try:
                resp = self._client.chat.completions.create(temperature=req.temperature, **kw)
            except Exception as e:
                if "temperature" not in str(e):
                    raise
                rejected = True
                resp = self._client.chat.completions.create(**kw)

        u = getattr(resp, "usage", None)
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", None),
            "output_tokens": getattr(u, "completion_tokens", None),
        }
        if rejected:
            usage["temperature_rejected"] = True

        choice = resp.choices[0]
        text = choice.message.content
        if not text:
            refusal = getattr(choice.message, "refusal", None)
            return Response(usage=usage,
                            error=f"본문 없음 (finish_reason={choice.finish_reason}, refusal={refusal})")
        try:
            return Response(verdict=json.loads(text), usage=usage)
        except json.JSONDecodeError as e:
            return Response(usage=usage, error=f"JSON 파싱 실패: {e} — {text[:120]}")
