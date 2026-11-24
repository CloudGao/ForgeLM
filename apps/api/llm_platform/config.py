from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="LLM_PLATFORM_", extra="ignore")

    app_name: str = "Local LLM Platform"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    database_url: str = "sqlite:///./.llm-platform/platform.db"
    global_assets_workspace_uri: str = ""
    storage_root: str = ""
    agent_token: str = "local-development-token"
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [value.strip() for value in self.cors_origins.split(",") if value.strip()]

    def ensure_directories(self) -> None:
        if self.database_url.startswith("sqlite:///"):
            raw = self.database_url.removeprefix("sqlite:///")
            if raw != ":memory:":
                Path(raw).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
