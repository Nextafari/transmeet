"""Baseten-powered transcription and translation (isolated from Meta/Whisper/Gemini).

Nothing here imports the Meta, Whisper, or Gemini backends. Shared,
provider-free helpers (local NLLB engine, TranslatorBackend ABC) are
imported from their own modules.
"""

from app.services.baseten.client import (
    BASETEN_BASE_URL,
    BasetenClient,
    BasetenError,
)
from app.services.baseten.file_backend import BasetenFileBackend

__all__ = [
    "BASETEN_BASE_URL",
    "BasetenClient",
    "BasetenError",
    "BasetenFileBackend",
]
