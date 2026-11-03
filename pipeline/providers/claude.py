"""Claude 어댑터 — 운영(server/judge_agent.py `_ask`)과 같은 방식

도구 하나(`final_verdict`)만 주고 `tool_choice`로 그것을 강제한다. 운영에서 실제로
쓰는 길이므로, 여기서 나온 결과는 "Claude를 다른 방식으로 불렀을 때"가 아니라
"서비스가 부르는 방식 그대로"의 결과다 — 세 모델 비교의 기준선으로 쓸 수 있다.

temperature는 **운영과 같이 지정하지 않는다** — server/ 어디에도 그 파라미터가 없으므로
API 기본값으로 간다. 임의로 0을 박으면 baseline이 서비스와 다른 설정으로 잰 것이 된다.
재현성이 필요한 실험만 `run_eval.py --temperature 0` 으로 그때 덮어쓴다.

운영과 다른 점은 재시도·스트리밍이 없다는 것뿐이다 (운영은 앱 12초 시한 때문에
하드 리밋과 저위험 조기 통보가 붙어 있다 — 판정 내용에는 영향이 없다).
"""

from __future__ import annotations

import os

from .base import Request, Response, require_env, require_sdk, to_anthropic_tool

# 운영 기본값과 동일(server/main.py:417 JUDGE_MODEL). 기준선을 서비스와 맞추기 위함 —
# 다른 Claude 모델과 비교하려면 --model 로 바꾼다
DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5")


class ClaudeProvider:
    name = "claude"

    def __init__(self) -> None:
        anthropic = require_sdk("anthropic", "anthropic", "claude")
        self._client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY", "claude"),
                                           max_retries=0)

    def default_model(self) -> str:
        return DEFAULT_MODEL

    def generate(self, req: Request) -> Response:
        tool = to_anthropic_tool(req.schema)
        # anthropic 1.0.0 은 messages.create() 시그니처에서 temperature 를 뺐다 (output_config 로 대체).
        # API 자체는 haiku-4-5 에 대해 아직 받으므로 extra_body 로 실어 보낸다 — 이렇게 하지 않으면
        # `run_eval.py --temperature 0` 이 전건 TypeError 로 죽는다. 운영은 temperature 를 지정하지
        # 않으므로(None) 기본 경로에는 이 키가 아예 실리지 않는다
        kw = {} if req.temperature is None else {"extra_body": {"temperature": req.temperature}}
        resp = self._client.messages.create(
            model=req.model,
            max_tokens=req.max_tokens,
            **kw,
            system=req.system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": req.user}],
        )
        u = getattr(resp, "usage", None)
        usage = {
            "input_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None),
        }
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use" and block.name == tool["name"]:
                return Response(verdict=dict(block.input), usage=usage)
        # tool_choice로 강제했는데도 도구를 안 불렀다 — 스키마 준수 실패로 기록
        stop = getattr(resp, "stop_reason", "?")
        return Response(usage=usage, error=f"tool_use 블록 없음 (stop_reason={stop})")
