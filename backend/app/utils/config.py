from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """All runtime configuration. Override via environment or .env."""

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    face_model_path: str = "saved_models/face_best.pt"
    voice_model_path: str = "saved_models/voice_best.pt"
    fusion_model_path: str = "saved_models/fusion_best.pt"
    whisper_model_size: str = "base"

    target_cycle_ms: int = 500
    adaptive_cycle_max_ms: int = 1000
    transcription_interval_cycles: int = 2

    session_db_path: str = "data/sessions.db"
    session_ttl_hours: int = 24

    cors_origins: str = "*"
    rate_limit_per_min: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def cors_origins_list(self) -> List[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
