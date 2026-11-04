"""Baseten file backend: audio transcription + leg translation.

Implements the shared :class:`TranslatorBackend` interface so it can sit in
the live backend chain (and, if configured, the captions chain) next to the
Meta, Whisper, and Gemini backends. Transcription goes to an
audio-capable Baseten Model APIs slug; translation goes through the
configured legs.
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

from app.services.baseten.client import BasetenClient
from app.services.meta.translation import translate_with_legs
from app.services.translator_backends import TranslatorBackend

logger = logging.getLogger(__name__)


class BasetenFileBackend(TranslatorBackend):
    """Baseten audio transcription + leg translation (chunked HTTP path)."""

    def __init__(
        self,
        *,
        api_key: str | None,
        asr_model: str = "meta-models/muse-spark-open",
        chat_model: str = "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        base_url: str = "https://inference.baseten.co/v1",
        translation_primary: str = "nllb",
        translation_fallbacks: list[str] | None = None,
        local_engine: Any | None = None,
        baseten_client: Any | None = None,
        spark_client: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._asr_model = asr_model
        self._chat_model = chat_model
        self._base_url = base_url
        self._translation_primary = translation_primary
        self._translation_fallbacks = list(translation_fallbacks or [])
        self._local_engine = local_engine
        self._baseten_client = baseten_client
        self._spark_client = spark_client
        self._timeout = timeout

    def _client(self) -> BasetenClient:
        return BasetenClient(
            self._api_key,
            chat_model=self._chat_model,
            asr_model=self._asr_model,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    async def _translate(
        self,
        text: str,
        source_language: str | None,
        target_language: str,
    ) -> str:
        """Translate, degrading to the original text when all legs fail."""
        if not text.strip():
            return ""
        try:
            translated, leg = await translate_with_legs(
                text,
                source_language=source_language,
                target_language=target_language,
                primary=self._translation_primary,
                fallbacks=self._translation_fallbacks,
                local_engine=self._local_engine,
                spark_client=self._spark_client,
                baseten_client=self._baseten_client,
            )
        except Exception as exc:
            logger.warning(
                "Baseten translation legs failed, keeping source text: %s", exc
            )
            return text
        if not translated.strip():
            return text
        logger.info("Baseten file backend translated via leg '%s'", leg)
        return translated

    async def transcribe_and_translate_audio(
        self,
        audio_path: Path,
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> tuple[str, str]:
        started = time.monotonic()
        transcribed = await self._client().transcribe_audio(
            audio_path, source_language
        )
        if not transcribed.strip():
            logger.info(
                "Baseten file backend produced no transcription in %.3fs",
                time.monotonic() - started,
            )
            return "", ""
        translated = await self._translate(
            transcribed, source_language, target_language
        )
        logger.info(
            "Baseten file backend done in %.3fs (chars=%s)",
            time.monotonic() - started,
            len(translated),
        )
        return transcribed, translated

    async def transcribe_audio(
        self,
        audio_path: Path,
        source_language: Optional[str] = None,
    ) -> str:
        return await self._client().transcribe_audio(audio_path, source_language)

    async def transcribe_and_translate_audio_segments(
        self,
        audio_path: Path,
        target_language: str = "en",
        source_language: Optional[str] = None,
        translate: bool = True,
    ) -> list[dict[str, Any]]:
        started = time.monotonic()
        transcribed = await self._client().transcribe_audio(
            audio_path, source_language
        )
        if not transcribed.strip():
            return []

        # Baseten returns no timestamps: one segment, end estimated from
        # word count like the Meta backend's final-turn estimate.
        translated = transcribed
        if translate:
            translated = await self._translate(
                transcribed, source_language, target_language
            )
        segments = [
            {
                "start_sec": 0.0,
                "end_sec": min(8.0, max(1.0, len(transcribed.split()) * 0.4)),
                "source_text": transcribed,
                "translated_text": translated,
            }
        ]
        logger.info(
            "Baseten file backend segments done in %.3fs",
            time.monotonic() - started,
        )
        return segments
