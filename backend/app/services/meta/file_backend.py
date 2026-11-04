"""Meta file backend: one-shot transcription + leg translation.

Implements the shared :class:`TranslatorBackend` interface so it can sit in
the live backend chain (and, if configured, the captions chain) next to the
Whisper and Gemini backends. Transcription goes to Muse Voice Transcribe's
one-shot endpoint; translation goes through the configured legs.
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

from app.services.meta.translation import translate_with_legs
from app.services.meta.voice_client import MetaVoiceClient
from app.services.translator_backends import TranslatorBackend

logger = logging.getLogger(__name__)


class MetaFileBackend(TranslatorBackend):
    """Meta one-shot ASR + nllb/spark translation (chunked HTTP path)."""

    def __init__(
        self,
        *,
        api_key: str | None,
        asr_model: str = "muse-voice-transcribe-1.0",
        asr_mode: str = "ENDPOINTING",
        translation_primary: str = "nllb",
        translation_fallbacks: list[str] | None = None,
        local_engine: Any | None = None,
        spark_client: Any | None = None,
        baseten_client: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._asr_model = asr_model
        self._asr_mode = asr_mode
        self._translation_primary = translation_primary
        self._translation_fallbacks = list(translation_fallbacks or [])
        self._local_engine = local_engine
        self._spark_client = spark_client
        self._baseten_client = baseten_client
        self._timeout = timeout

    def _voice_client(self, source_language: str | None) -> MetaVoiceClient:
        language_bias = [source_language] if source_language else None
        return MetaVoiceClient(
            self._api_key,
            model=self._asr_model,
            mode=self._asr_mode,
            language_bias=language_bias,
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
            logger.warning("Meta translation legs failed, keeping source text: %s", exc)
            return text
        if not translated.strip():
            return text
        logger.info("Meta file backend translated via leg '%s'", leg)
        return translated

    async def transcribe_and_translate_audio(
        self,
        audio_path: Path,
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> tuple[str, str]:
        started = time.monotonic()
        result = await self._voice_client(source_language).transcribe_file(audio_path)
        transcribed = result["transcript"]
        if not transcribed.strip():
            logger.info(
                "Meta file backend produced no transcription in %.3fs",
                time.monotonic() - started,
            )
            return "", ""
        translated = await self._translate(
            transcribed, source_language, target_language
        )
        logger.info(
            "Meta file backend done in %.3fs (chars=%s)",
            time.monotonic() - started,
            len(translated),
        )
        return transcribed, translated

    async def transcribe_audio(
        self,
        audio_path: Path,
        source_language: Optional[str] = None,
    ) -> str:
        result = await self._voice_client(source_language).transcribe_file(audio_path)
        return result["transcript"]

    async def transcribe_and_translate_audio_segments(
        self,
        audio_path: Path,
        target_language: str = "en",
        source_language: Optional[str] = None,
        translate: bool = True,
    ) -> list[dict[str, Any]]:
        started = time.monotonic()
        result = await self._voice_client(source_language).transcribe_file(audio_path)
        turns = result["turns"]
        if not turns:
            transcript = result["transcript"]
            if not transcript.strip():
                return []
            turns = [{"turn_id": 0, "start_sec": 0.0, "speaker": None, "text": transcript}]

        segments: list[dict[str, Any]] = []
        for index, turn in enumerate(turns):
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            try:
                start_sec = max(0.0, float(turn.get("start_sec") or 0.0))
            except (TypeError, ValueError):
                start_sec = 0.0
            # Turn-level timestamps only: end at the next turn, or estimate
            # from word count for the final turn.
            if index + 1 < len(turns):
                try:
                    end_sec = max(
                        start_sec + 0.2,
                        float(turns[index + 1].get("start_sec") or 0.0),
                    )
                except (TypeError, ValueError):
                    end_sec = start_sec + 4.0
            else:
                end_sec = start_sec + min(8.0, max(1.0, len(text.split()) * 0.4))
            segments.append(
                {
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "source_text": text,
                    "translated_text": text,
                }
            )

        if translate:
            for segment in segments:
                segment["translated_text"] = await self._translate(
                    segment["source_text"], source_language, target_language
                )

        logger.info(
            "Meta file backend segments done in %.3fs (segments=%s)",
            time.monotonic() - started,
            len(segments),
        )
        return segments
