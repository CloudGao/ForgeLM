from __future__ import annotations

from llm_platform.config import Settings


def test_api_bind_address_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PLATFORM_API_HOST", "0.0.0.0")
    monkeypatch.setenv("LLM_PLATFORM_API_PORT", "9123")

    settings = Settings(_env_file=None)

    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 9123
