import json
from dataclasses import dataclass

import httpx
from jsonschema import Draft202012Validator, ValidationError
from pydantic import ValidationError as PydanticValidationError

from wald_agent.config import Settings
from wald_agent.errors import InvalidProviderResponse
from wald_agent.schemas import Usage
from wald_agent.transport import JsonTransport, strict_json_loads


def closed_object(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


@dataclass
class ChatResult:
    data: dict
    model: str
    usage: Usage
    attempts: int = 1
    upstream_request_id: str | None = None


class ChatClient(JsonTransport):
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
        *,
        vision: bool = False,
    ):
        super().__init__(settings.api_timeout_seconds, http_client, settings=settings)
        self.settings = settings
        self.vision = vision

    async def structured(self, messages: list[dict], schema: dict, name: str) -> ChatResult:
        settings = self.settings
        provider = "vision" if self.vision else "llm"
        key = settings.key_for(provider)
        model = settings.vision_model if self.vision else settings.llm_model
        base_url = settings.llm_base_url
        if self.vision:
            base_url = settings.vision_base_url or base_url
        body = {
            "model": model,
            "messages": messages,
            settings.llm_token_limit_field: settings.llm_max_output_tokens,
        }
        if settings.llm_temperature is not None:
            body["temperature"] = settings.llm_temperature
        if settings.llm_response_format == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            }
        else:
            body["response_format"] = {"type": "json_object"}
            body["messages"] = [
                *messages,
                {"role": "user", "content": "Return JSON matching: " + json.dumps(schema)},
            ]
        raw = await self.post(f"{base_url}/chat/completions", key, body, provider)
        try:
            choices = raw["choices"]
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("invalid choices")
            choice = choices[0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise ValueError("invalid message")
            if message.get("refusal"):
                raise InvalidProviderResponse("LLM refused the decision request.")
            if choice.get("finish_reason") != "stop":
                raise InvalidProviderResponse("LLM output was incomplete or filtered.")
            content = message["content"]
            if not isinstance(content, str):
                raise InvalidProviderResponse("LLM response did not contain JSON text.")
            data = strict_json_loads(content)
            Draft202012Validator(schema).validate(data)
            # json.loads accepts NaN/Infinity; standard JSON does not.
            json.dumps(data, allow_nan=False)
            usage = raw.get("usage") or {}
            if not isinstance(usage, dict):
                raise ValueError("invalid usage")
            result = ChatResult(
                data=data,
                model=raw.get("model") or model,
                usage=Usage(
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                ),
                attempts=raw.attempts,
                upstream_request_id=raw.upstream_request_id,
            )
            if not isinstance(result.model, str):
                raise ValueError("invalid model identifier")
            return result
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            ValidationError,
            PydanticValidationError,
        ) as exc:
            raise InvalidProviderResponse("LLM returned an invalid structured response.") from exc
