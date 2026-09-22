from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from wald_agent.errors import ConfigurationError

LOCAL_CONFIG_FILE = Path("config/.env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=LOCAL_CONFIG_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
        populate_by_name=True,
    )

    llm_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY")
    )
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_response_format: Literal["json_schema", "json_object"] = "json_schema"
    llm_max_output_tokens: int = Field(default=2048, ge=64, le=100_000)
    llm_token_limit_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    llm_temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    typesafe_api_key: SecretStr | None = None
    jev_base_url: str = "https://api.typesafe.ai/v1"
    jev_model: str = "jev-1.13.0"
    vision_api_key: SecretStr | None = None
    vision_base_url: str | None = None
    vision_model: str = "gpt-4o-mini"
    api_timeout_seconds: float = Field(default=60, gt=0, le=600, allow_inf_nan=False)
    wald_api_key: SecretStr | None = None
    wald_env: Literal["development", "production"] = "development"
    wald_api_keys: dict[str, SecretStr] = Field(default_factory=dict)
    enabled_providers: list[Literal["llm", "jev"]] = Field(default_factory=lambda: ["llm"])
    database_path: Path = Path("data/wald.sqlite3")
    audit_store_inputs: bool = True
    retention_days: int = Field(default=30, ge=1, le=3650)
    max_concurrent_requests: int = Field(default=8, ge=1, le=256)
    max_queued_requests: int = Field(default=32, ge=0, le=4096)
    queue_timeout_seconds: float = Field(default=5, gt=0, le=60, allow_inf_nan=False)
    request_timeout_seconds: float = Field(default=150, gt=0, le=1800, allow_inf_nan=False)
    rate_limit_per_minute: int = Field(default=120, ge=1, le=1_000_000)
    max_body_bytes: int = Field(default=16 * 1024 * 1024, ge=1024, le=32 * 1024 * 1024)
    body_timeout_seconds: float = Field(default=15, gt=0, le=120, allow_inf_nan=False)
    max_state_bytes: int = Field(default=256_000, ge=1024, le=4 * 1024 * 1024)
    api_max_retries: int = Field(default=2, ge=0, le=5)
    api_retry_base_seconds: float = Field(default=0.5, ge=0, le=10, allow_inf_nan=False)
    api_retry_max_seconds: float = Field(default=10, gt=0, le=60, allow_inf_nan=False)
    api_connect_timeout_seconds: float = Field(default=5, gt=0, le=60, allow_inf_nan=False)
    api_max_response_bytes: int = Field(default=2 * 1024 * 1024, ge=1024, le=32 * 1024 * 1024)
    circuit_failure_threshold: int = Field(default=5, ge=1, le=100)
    circuit_cooldown_seconds: float = Field(default=30, gt=0, le=300, allow_inf_nan=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def validate_service(self) -> "Settings":
        if not self.enabled_providers or len(set(self.enabled_providers)) != len(
            self.enabled_providers
        ):
            raise ValueError("enabled_providers must be nonempty and unique")
        if len(self.wald_api_keys) > 64:
            raise ValueError("at most 64 service API keys are supported")
        if any(not name or len(name) > 64 or name == "anonymous" for name in self.wald_api_keys):
            raise ValueError("client names must have 1-64 characters and cannot be anonymous")
        keys = [secret.get_secret_value() for secret in self.wald_api_keys.values()]
        if self.wald_api_key:
            if "default" in self.wald_api_keys:
                raise ValueError("client name default is reserved when WALD_API_KEY is set")
            keys.append(self.wald_api_key.get_secret_value())
        if len(keys) != len(set(keys)) or any(not key.strip() for key in keys):
            raise ValueError("service API keys must be unique and nonempty")
        if self.wald_env == "production" and (not keys or any(len(key) < 32 for key in keys)):
            raise ValueError("production requires a service API key of at least 32 characters")
        return self

    def validate_startup(self) -> None:
        if self.wald_env == "production":
            for provider in self.enabled_providers:
                self.key_for(provider)

    def service_keys(self) -> dict[str, str]:
        keys = {name: secret.get_secret_value() for name, secret in self.wald_api_keys.items()}
        if self.wald_api_key:
            keys["default"] = self.wald_api_key.get_secret_value()
        return keys

    @field_validator("llm_base_url", "jev_base_url", "vision_base_url")
    @classmethod
    def valid_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base URL must be http(s) without credentials, query or fragment")
        return value.rstrip("/")

    @field_validator("llm_model", "jev_model", "vision_model")
    @classmethod
    def nonempty_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model name must not be empty")
        return value

    def key_for(self, provider: Literal["llm", "jev", "vision"]) -> str:
        value = {
            "llm": self.llm_api_key,
            "jev": self.typesafe_api_key,
            "vision": self.vision_api_key or self.llm_api_key,
        }[provider]
        if value is None or not value.get_secret_value().strip():
            name = {
                "llm": "LLM_API_KEY (or OPENAI_API_KEY)",
                "jev": "TYPESAFE_API_KEY",
                "vision": "VISION_API_KEY (or LLM_API_KEY / OPENAI_API_KEY)",
            }[provider]
            raise ConfigurationError(
                f"Missing {name}; configure it in config/.env or the environment."
            )
        return value.get_secret_value()
