"""
Central configuration for the Agentic Chaser project.

Loads environment variables and provides strongly-typed settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

# Load .env if present
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)


# ---------- Settings Dataclasses ----------

@dataclass(frozen=True)
class OllamaSettings:
    """Local Ollama LLM configuration. If model not found (404), run: ollama pull <model> or set OLLAMA_MODEL to a model from 'ollama list'."""
    model: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature: float = 0.3
    timeout: int = 120  # seconds


@dataclass(frozen=True)
class DatabaseSettings:
    """PostgreSQL database configuration."""
    url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://nagesh:nagesh123@127.0.0.1:5433/agentic_chaser"
    )


@dataclass(frozen=True)
class RedisSettings:
    """Redis cache configuration."""
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")


@dataclass(frozen=True)
class CommunicationSettings:
    """Email, SMS, WhatsApp API settings."""
    # SendGrid
    sendgrid_api_key: str | None = os.getenv("SENDGRID_API_KEY")
    from_email: str = os.getenv("FROM_EMAIL", "advisor@example.com")
    
    # Twilio
    twilio_account_sid: str | None = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token: str | None = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_phone: str | None = os.getenv("TWILIO_PHONE_NUMBER")


@dataclass(frozen=True)
class PathsSettings:
    """Filesystem paths for data, models, evidence."""
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    test_data_dir: Path = BASE_DIR / "data" / "test"
    synthetic_data_dir: Path = BASE_DIR / "data" / "synthetic_data"
    trained_models_dir: Path = BASE_DIR / "data" / "trained_models"
    evidence_dir: Path = BASE_DIR / "data" / "evidence"


@dataclass(frozen=True)
class AppSettings:
    """General application settings."""
    environment: str = os.getenv("APP_ENV", "development")
    debug: bool = os.getenv("APP_DEBUG", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


@dataclass(frozen=True)
class MLSettings:
    """Machine learning model settings."""
    sentiment_model_path: Path = BASE_DIR / "data" / "trained_models" / "sentiment_model.pkl"
    priority_model_path: Path = BASE_DIR / "data" / "trained_models" / "priority_model.pkl"
    vectorizer_path: Path = BASE_DIR / "data" / "trained_models" / "vectorizer.pkl"


@dataclass(frozen=True)
class Settings:
    """Aggregate settings object."""
    ollama: OllamaSettings
    db: DatabaseSettings
    redis: RedisSettings
    comms: CommunicationSettings
    paths: PathsSettings
    app: AppSettings
    ml: MLSettings


# ---------- Factory ----------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load and cache settings from environment.
    Creates necessary directories if they don't exist.
    """
    # Ensure directories exist
    paths = PathsSettings()
    for path in [
        paths.data_dir,
        paths.test_data_dir,
        paths.synthetic_data_dir,
        paths.trained_models_dir,
        paths.evidence_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return Settings(
        ollama=OllamaSettings(),
        db=DatabaseSettings(),
        redis=RedisSettings(),
        comms=CommunicationSettings(),
        paths=paths,
        app=AppSettings(),
        ml=MLSettings(),
    )


# ---------- Global Settings Instance ----------

settings = get_settings()