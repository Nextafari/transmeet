"""Shared fixtures for the backend test suite."""

import os
import sys
from pathlib import Path

import pytest

# Required Settings fields so app imports never depend on a real .env.
BASE_ENV = {
    "APP_NAME": "Test Backend",
    "APP_VERSION": "0.0.0-test",
    "API_V1_PREFIX": "/api/v1",
    "CORS_ORIGINS": ["*"],
    "TRANSLATION_API_VERSION": "v1",
    "TRANSLATION_MODEL": "gemini-flash-latest",
    "LIVE_CAPTION_MODEL": "gemini-2.0-flash-lite",
    "LIVE_CAPTION_MIN_INTERVAL_SECONDS": "1",
    "LIVE_CAPTION_CACHE_TTL_SECONDS": "20",
    "LIVE_CAPTION_MIN_AUDIO_BYTES": "2500",
    "TRANSLATION_QUOTA_COOLDOWN_SECONDS": "60",
    "TRANSCRIPTION_CHUNK_SECONDS": "45",
    "MAX_TRANSCRIPTION_SECONDS": "900",
    "LIVE_CAPTION_CHUNK_SECONDS": "1",
}

for _key, _value in BASE_ENV.items():
    if isinstance(_value, str):
        os.environ.setdefault(_key, _value)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_settings(**overrides):
    """Fresh Settings with test defaults plus overrides."""
    from app.core.config import Settings

    data = dict(BASE_ENV)
    data.update(overrides)
    return Settings(**data)


@pytest.fixture()
def any_settings(monkeypatch):
    """Patch the global settings object with the given attributes."""
    from app.core.config import settings

    def _apply(**overrides):
        for key, value in overrides.items():
            monkeypatch.setattr(settings, key, value)
        return settings

    return _apply
