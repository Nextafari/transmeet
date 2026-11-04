"""
Translation and transcription service.

Supports multiple backends:
- Gemini: High-quality post-processing (transcription + translation)
- Whisper + Google Translate: Fast, low-cost live captions (language auto-detect)
- Whisper + local translation: fully local live captions and subtitles

Live captions use LIVE_CAPTION_BACKEND setting.
Post-recording captions use CAPTIONS_BACKEND.
"""
import asyncio
import base64
import ast
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings
from app.services.baseten.client import BasetenClient
from app.services.baseten.file_backend import BasetenFileBackend
from app.services.local_translation import LocalTranslationEngine
from app.services.meta.file_backend import MetaFileBackend
from app.services.meta.spark_client import MetaSparkClient
from app.services.translator_backends import (
    GeminiTranslatorBackend,
    WhisperLocalTranslateBackend,
    WhisperGoogleTranslateBackend,
)

logger = logging.getLogger(__name__)

# Backends that may appear in the ordered live caption chain.
VALID_LIVE_RUNGS = frozenset(
    {"whisper-google", "whisper-local", "gemini", "meta", "baseten"}
)


def _looks_like_quota(exc: BaseException) -> bool:
    if bool(getattr(exc, "quota", False)):
        return True
    lowered = str(exc).lower()
    return (
        "429" in lowered
        or "quota" in lowered
        or "rate limit" in lowered
        or "retry-after" in lowered
    )


@dataclass
class TranscriptSegment:
    start_sec: float
    end_sec: float
    source_text: str
    translated_text: str


class TranslatorService:
    """High-level translation service with backend abstraction."""

    def __init__(self) -> None:
        self._quota_cooldown_until = 0.0
        self._live_caption_backend: Any = None
        self._captions_backend: Any = None
        self._rung_backends: dict[tuple[str, str, str], Any] = {}
        self._rung_cooldown_until: dict[str, float] = {}
        self._shared_local_engine: Any = None
        self._shared_local_engine_ready = False
        self._spark_client: Any = None
        self._spark_client_ready = False
        self._baseten_client: Any = None
        self._baseten_client_ready = False
        self._backend_lock = threading.RLock()

    @staticmethod
    def _model_size(value: str | None, fallback: str) -> str:
        return (value or fallback or "base").strip() or "base"

    def _live_whisper_model_size(self) -> str:
        return self._model_size(
            settings.live_whisper_model_size,
            settings.whisper_model_size,
        )

    def _captions_whisper_model_size(self) -> str:
        return self._model_size(
            settings.captions_whisper_model_size,
            settings.whisper_model_size,
        )

    def _get_shared_local_engine(self) -> Any | None:
        """One shared NLLB/Marian engine for whisper-local and Meta legs."""
        with self._backend_lock:
            if not self._shared_local_engine_ready:
                self._shared_local_engine_ready = True
                try:
                    self._shared_local_engine = LocalTranslationEngine(
                        provider=settings.local_translation_provider,
                        model_name=settings.local_translation_model,
                    )
                except Exception as exc:
                    logger.warning("Local translation engine unavailable: %s", exc)
                    self._shared_local_engine = None
            return self._shared_local_engine

    def _get_spark_client(self) -> Any | None:
        """One shared Spark client, or None when the Meta key is missing."""
        with self._backend_lock:
            if not self._spark_client_ready:
                self._spark_client_ready = True
                if not (settings.meta_model_api_key or "").strip():
                    self._spark_client = None
                else:
                    try:
                        self._spark_client = MetaSparkClient(
                            api_key=settings.meta_model_api_key,
                            model=settings.meta_spark_model,
                        )
                    except Exception as exc:
                        logger.warning("Spark client unavailable: %s", exc)
                        self._spark_client = None
            return self._spark_client

    def _get_baseten_client(self) -> Any | None:
        """One shared Baseten client, or None when the key is missing."""
        with self._backend_lock:
            if not self._baseten_client_ready:
                self._baseten_client_ready = True
                if not (settings.baseten_api_key or "").strip():
                    self._baseten_client = None
                else:
                    try:
                        self._baseten_client = BasetenClient(
                            settings.baseten_api_key,
                            chat_model=settings.baseten_chat_model,
                            asr_model=settings.baseten_asr_model,
                            base_url=settings.baseten_base_url,
                            timeout=settings.baseten_timeout_seconds,
                        )
                    except Exception as exc:
                        logger.warning("Baseten client unavailable: %s", exc)
                        self._baseten_client = None
            return self._baseten_client

    def _build_backend(
        self,
        backend_type: str,
        *,
        whisper_model_size: str,
        gemini_model: str,
    ) -> Any:
        logger.info("Initializing backend: %s", backend_type)
        if backend_type == "whisper-google":
            return WhisperGoogleTranslateBackend(
                google_api_key=settings.google_api_key,
                google_project_id=settings.google_translate_project_id,
                whisper_model_size=whisper_model_size,
            )
        if backend_type == "whisper-local":
            return WhisperLocalTranslateBackend(
                whisper_model_size=whisper_model_size,
                provider=settings.local_translation_provider,
                model_name=settings.local_translation_model,
                translation_engine=self._get_shared_local_engine(),
            )
        if backend_type == "gemini":
            return GeminiTranslatorBackend(
                api_key=settings.google_api_key,
                model=gemini_model,
            )
        if backend_type == "meta":
            return MetaFileBackend(
                api_key=settings.meta_model_api_key,
                asr_model=settings.meta_asr_model,
                asr_mode=settings.meta_asr_mode,
                translation_primary=settings.meta_translation_engine,
                translation_fallbacks=settings.stream_translation_fallbacks,
                local_engine=self._get_shared_local_engine(),
                spark_client=self._get_spark_client(),
                baseten_client=self._get_baseten_client(),
                timeout=settings.live_stream_op_timeout_seconds,
            )
        if backend_type == "baseten":
            return BasetenFileBackend(
                api_key=settings.baseten_api_key,
                asr_model=settings.baseten_asr_model,
                chat_model=settings.baseten_chat_model,
                base_url=settings.baseten_base_url,
                translation_primary=settings.meta_translation_engine,
                translation_fallbacks=settings.stream_translation_fallbacks,
                local_engine=self._get_shared_local_engine(),
                baseten_client=self._get_baseten_client(),
                spark_client=self._get_spark_client(),
                timeout=settings.baseten_timeout_seconds,
            )
        raise ValueError(f"Unknown backend: {backend_type}")

    def _get_cached_backend(
        self,
        backend_type: str,
        *,
        whisper_model_size: str,
        gemini_model: str,
    ) -> Any:
        # Shared cache keyed by everything that changes construction, so
        # identical live/captions configurations reuse one instance.
        key = (backend_type, whisper_model_size, gemini_model)
        with self._backend_lock:
            backend = self._rung_backends.get(key)
            if backend is None:
                backend = self._build_backend(
                    backend_type,
                    whisper_model_size=whisper_model_size,
                    gemini_model=gemini_model,
                )
                self._rung_backends[key] = backend
            return backend

    def _get_live_rung_backend(self, backend_type: str) -> Any:
        """Build (or reuse) one rung of the live chain with live settings."""
        return self._get_cached_backend(
            backend_type,
            whisper_model_size=self._live_whisper_model_size(),
            gemini_model=settings.live_caption_model,
        )

    def _get_live_caption_backend(self) -> Any:
        """Get or create the legacy single live caption backend."""
        with self._backend_lock:
            if self._live_caption_backend is None:
                self._live_caption_backend = self._get_live_rung_backend(
                    settings.live_caption_backend
                )
            return self._live_caption_backend

    def _get_captions_backend(self) -> Any:
        """Get or create the captions backend."""
        with self._backend_lock:
            if self._captions_backend is None:
                self._captions_backend = self._get_cached_backend(
                    settings.captions_backend,
                    whisper_model_size=self._captions_whisper_model_size(),
                    gemini_model=settings.translation_model,
                )
            return self._captions_backend

    async def transcribe_and_translate_live_chain(
        self,
        audio_path: Path,
        *,
        source_language: str | None = None,
        target_language: str,
        translate: bool = True,
    ) -> tuple[str, str]:
        """Walk the configured live chain; return (caption_text, rung_used).

        Rungs in cooldown are skipped. The ``meta`` and ``baseten`` rungs
        are skipped when their keys are not configured. Quota-like
        failures start a per-rung cooldown; anything else just falls
        through to the next rung.
        """
        chain = settings.effective_live_chain()
        unknown = [rung for rung in chain if rung not in VALID_LIVE_RUNGS]
        if unknown:
            raise ValueError(
                f"Unknown live caption backend(s): {unknown}. "
                f"Use entries from {sorted(VALID_LIVE_RUNGS)}."
            )
        if not chain:
            raise RuntimeError("No live caption backend configured")

        last_error: Exception | None = None
        for rung in chain:
            if self._rung_cooldown_until.get(rung, 0.0) > time.monotonic():
                logger.info("Live caption rung '%s' cooling down, skipping", rung)
                continue
            if rung == "meta" and not (settings.meta_model_api_key or "").strip():
                logger.info(
                    "Live caption rung 'meta' skipped: META_MODEL_API_KEY not set"
                )
                continue
            if rung == "baseten" and not (settings.baseten_api_key or "").strip():
                logger.info(
                    "Live caption rung 'baseten' skipped: BASETEN_API_KEY not set"
                )
                continue
            try:
                backend = await asyncio.to_thread(self._get_live_rung_backend, rung)
                if not translate:
                    transcribed = await backend.transcribe_audio(
                        audio_path,
                        source_language=source_language,
                    )
                    return transcribed.strip(), rung

                transcribed, translated = (
                    await backend.transcribe_and_translate_audio(
                        audio_path,
                        source_language=source_language,
                        target_language=target_language,
                    )
                )
                caption_text = translated.strip() or transcribed.strip()
                if transcribed and not caption_text:
                    logger.warning(
                        "Live caption output empty; using transcription."
                    )
                    caption_text = transcribed
                logger.info(
                    "Live caption served by rung '%s' (chars=%s)",
                    rung,
                    len(caption_text),
                )
                return caption_text, rung
            except Exception as exc:
                last_error = exc
                logger.warning("Live caption rung '%s' failed: %s", rung, exc)
                if _looks_like_quota(exc):
                    self._rung_cooldown_until[rung] = (
                        time.monotonic()
                        + settings.translation_quota_cooldown_seconds
                    )

        if last_error is not None:
            raise last_error
        raise RuntimeError(
            "All live caption backends unavailable (cooling down or unconfigured)"
        )

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        name = model_name.strip()
        if name.startswith("models/"):
            return name.split("/", 1)[1]
        return name

    @staticmethod
    def _candidate_models(model_name: str) -> list[str]:
        base_name = TranslatorService._normalize_model_name(model_name)
        candidates = [base_name]

        if "flash-lite" in base_name:
            candidates.extend(
                [
                    "gemini-2.0-flash-lite",
                    "gemini-2.0-flash-lite-001",
                    "gemini-flash-lite-latest",
                    "gemini-2.0-flash",
                ]
            )

        if base_name in {
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-002",
        }:
            candidates.extend(
                [
                    "gemini-flash-latest",
                    "gemini-flash-lite-latest",
                    "gemini-2.5-flash",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                ]
            )
        if base_name in {
            "gemini-1.5-pro",
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro-002",
        }:
            candidates.extend(
                [
                    "gemini-pro-latest",
                    "gemini-2.5-pro",
                ]
            )
        if base_name == "gemini-flash-latest":
            candidates.extend(["gemini-flash-lite-latest", "gemini-2.5-flash"])
        # De-duplicate while preserving order
        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            if candidate in seen:
                continue
            ordered.append(candidate)
            seen.add(candidate)
        return ordered

    @staticmethod
    def _candidate_versions(version: str) -> list[str]:
        candidates = [version]
        if version != "v1":
            candidates.append("v1")
        # De-duplicate while preserving order
        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            if candidate in seen:
                continue
            ordered.append(candidate)
            seen.add(candidate)
        return ordered

    def _require_api_key(self) -> str:
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not configured. "
                "Set it in backend/.env before using translation features."
            )
        return settings.google_api_key

    def _remaining_quota_cooldown(self) -> float:
        now = time.monotonic()
        return max(0.0, self._quota_cooldown_until - now)

    def _set_quota_cooldown(self, retry_after_header: str | None) -> None:
        cooldown_seconds = settings.translation_quota_cooldown_seconds
        if retry_after_header:
            try:
                parsed = int(retry_after_header)
                if parsed > 0:
                    cooldown_seconds = max(cooldown_seconds, parsed)
            except ValueError:
                pass
        self._quota_cooldown_until = time.monotonic() + cooldown_seconds

    async def _generate_content(
        self,
        parts: list[dict[str, Any]],
        *,
        preferred_model: str | None = None,
    ) -> str:
        api_key = self._require_api_key()
        cooldown = self._remaining_quota_cooldown()
        if cooldown > 0:
            raise RuntimeError(
                "Translation quota cooldown is active. "
                f"Retry in about {int(cooldown)}s"
            )

        async with httpx.AsyncClient(timeout=120.0) as client:
            last_error: tuple[int, str, str] | None = None
            for version in self._candidate_versions(
                settings.translation_api_version
            ):
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": parts,
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.1,
                    },
                }

                endpoint_base = (
                    f"https://generativelanguage.googleapis.com/{version}/models/"
                )
                for model_name in self._candidate_models(
                    preferred_model or settings.translation_model
                ):
                    response = await client.post(
                        f"{endpoint_base}{model_name}:generateContent",
                        params={"key": api_key},
                        json=payload,
                    )

                    if response.status_code >= 400:
                        body = response.text[:300]
                        last_error = (response.status_code, body, model_name)
                        if response.status_code == 429:
                            self._set_quota_cooldown(
                                response.headers.get("Retry-After")
                            )
                            break
                        if response.status_code != 404:
                            break
                        continue

                    data = response.json()
                    candidates = data.get("candidates") or []
                    if not candidates:
                        raise RuntimeError("Gemini returned no candidates")

                    parts = candidates[0].get("content", {}).get("parts") or []
                    for part in parts:
                        text = part.get("text")
                        if text:
                            if model_name != settings.translation_model:
                                settings.translation_model = model_name
                            if version != settings.translation_api_version:
                                settings.translation_api_version = version
                            return text

                    raise RuntimeError(
                        "Gemini response did not include text output"
                    )

                if self._remaining_quota_cooldown() > 0:
                    break

        if last_error:
            status, body, model_name = last_error
            raise RuntimeError(
                f"Gemini request failed ({status}) for {model_name}: {body}"
            )

        raise RuntimeError("Gemini request failed for unknown reasons")

    def _extract_json_payload(self, raw_text: str) -> Any:
        text = raw_text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Gemini sometimes wraps JSON in fenced code blocks.
        fenced = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.DOTALL,
        )
        if fenced:
            try:
                return json.loads(fenced.group(1).strip())
            except json.JSONDecodeError:
                pass

        bracketed = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if bracketed:
            snippet = bracketed.group(1)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass

            # Fallback for Python-like dict/list output with single quotes.
            try:
                return ast.literal_eval(snippet)
            except (ValueError, SyntaxError):
                pass

        raise RuntimeError("Could not parse JSON payload from model output")

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _normalize_segments(
        self,
        raw_segments: list[dict[str, Any]],
        *,
        chunk_start_sec: float,
        chunk_duration_sec: float,
    ) -> list[TranscriptSegment]:
        normalized: list[TranscriptSegment] = []
        for idx, raw in enumerate(raw_segments):
            if not isinstance(raw, dict):
                continue

            source_text = str(
                raw.get("source_text") or raw.get("text") or ""
            ).strip()
            translated_text = str(raw.get("translated_text") or "").strip()
            if not source_text:
                continue

            start_rel = self._to_float(
                raw.get(
                    "start_sec",
                    raw.get("start", raw.get("start_seconds")),
                ),
                default=max(0.0, idx * 3.0),
            )
            end_rel = self._to_float(
                raw.get("end_sec", raw.get("end", raw.get("end_seconds"))),
                default=min(chunk_duration_sec, start_rel + 3.0),
            )

            start_rel = max(0.0, min(start_rel, chunk_duration_sec))
            end_rel = max(start_rel + 0.2, min(end_rel, chunk_duration_sec))

            if not translated_text:
                translated_text = source_text

            normalized.append(
                TranscriptSegment(
                    start_sec=chunk_start_sec + start_rel,
                    end_sec=chunk_start_sec + end_rel,
                    source_text=source_text,
                    translated_text=translated_text,
                )
            )

        return normalized

    async def translate_text(
        self,
        text: str,
        target_language: str,
    ) -> tuple[str, str]:
        prompt = (
            "Translate the following text to the target language. "
            "Return strict JSON with this exact shape: "
            '{"translated_text": "..."}.\n'
            f"Target language: {target_language}\n"
            f"Text: {text}"
        )

        raw = await self._generate_content([{"text": prompt}])
        payload = self._extract_json_payload(raw)
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected translation payload from model")

        translated = str(payload.get("translated_text") or "").strip()
        if not translated:
            raise RuntimeError("Model returned empty translation")

        return translated, settings.translation_model

    async def transcribe_and_translate_audio_chunk(
        self,
        audio_path: Path,
        *,
        source_language: str | None = None,
        target_language: str,
        chunk_start_sec: float,
        chunk_duration_sec: float,
        translate: bool = True,
        model_name: str | None = None,
    ) -> tuple[list[TranscriptSegment], str]:
        """
        Transcribe (and optionally translate) an audio chunk for subtitles.

        Returns: (list of segments, model name used)
        """
        if settings.captions_backend != "gemini":
            captions_backend = await asyncio.to_thread(
                self._get_captions_backend
            )
            segment_method = getattr(
                captions_backend,
                "transcribe_and_translate_audio_segments",
                None,
            )
            if segment_method is None:
                raise RuntimeError(
                    f"Captions backend {settings.captions_backend} does not "
                    "support timestamped segments"
                )

            raw_segments = await segment_method(
                audio_path,
                source_language=source_language,
                target_language=target_language,
                translate=translate,
            )
            if not raw_segments:
                return [], settings.captions_backend

            segments = self._normalize_segments(
                raw_segments,
                chunk_start_sec=chunk_start_sec,
                chunk_duration_sec=chunk_duration_sec,
            )
            model_used = getattr(
                captions_backend,
                "model_name",
                settings.captions_backend,
            )
            return segments, str(model_used)

        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        if translate:
            source_instruction = (
                f"The spoken source language is {source_language}.\n"
                if source_language
                else ""
            )
            prompt = (
                source_instruction +
                "Transcribe all spoken content in this audio and translate "
                "each segment "
                f"to {target_language}.\n"
                "Return strict JSON array only. Each item must be:\n"
                "{\"start_sec\": number, \"end_sec\": number, "
                "\"source_text\": string, \"translated_text\": string}.\n"
                "Rules:\n"
                "- start_sec/end_sec are relative to this chunk and "
                f"between 0 and {chunk_duration_sec:.2f}.\n"
                "- keep segment duration between 1 and 8 seconds when possible.\n"
                "- include all spoken words.\n"
                "- no markdown, no explanation."
            )
        else:
            source_instruction = (
                f"The spoken source language is {source_language}.\n"
                if source_language
                else ""
            )
            prompt = (
                source_instruction +
                "Transcribe all spoken content in this audio.\n"
                "Return strict JSON array only. Each item must be:\n"
                "{\"start_sec\": number, \"end_sec\": number, "
                "\"source_text\": string}.\n"
                "Rules:\n"
                "- start_sec/end_sec are relative to this chunk and "
                f"between 0 and {chunk_duration_sec:.2f}.\n"
                "- keep segment duration between 1 and 8 seconds when possible.\n"
                "- include all spoken words.\n"
                "- no markdown, no explanation."
            )

        chosen_model = model_name or settings.translation_model
        raw = await self._generate_content(
            [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": audio_b64,
                    }
                },
            ],
            preferred_model=chosen_model,
        )

        payload = self._extract_json_payload(raw)
        if isinstance(payload, dict):
            raw_segments = payload.get("segments", [])
        elif isinstance(payload, list):
            raw_segments = payload
        else:
            raw_segments = []

        if not raw_segments:
            return [], chosen_model

        segments = self._normalize_segments(
            raw_segments,
            chunk_start_sec=chunk_start_sec,
            chunk_duration_sec=chunk_duration_sec,
        )
        return segments, chosen_model

    async def transcribe_and_translate_live(
        self,
        audio_path: Path,
        *,
        source_language: str | None = None,
        target_language: str,
        translate: bool = True,
    ) -> str:
        """
        Transcribe and translate audio for LIVE captions using the configured live caption backend.

        This is a simplified flow for live captions:
        - Input: audio file
        - Output: translated text (no segments with timing)
        - Uses appropriate backend (whisper-google or gemini)

        Returns: translated text
        """
        backend = await asyncio.to_thread(self._get_live_caption_backend)

        try:
            if not translate:
                transcribed = await backend.transcribe_audio(
                    audio_path,
                    source_language=source_language,
                )
                caption_text = transcribed.strip()
                return caption_text

            transcribed, translated = (
                await backend.transcribe_and_translate_audio(
                    audio_path,
                    source_language=source_language,
                    target_language=target_language,
                )
            )

            caption_text = translated.strip() or transcribed.strip()

            if transcribed and not caption_text:
                logger.warning(
                    "Live caption output empty; using transcription instead."
                )
                caption_text = transcribed

            logger.info(
                f"Live caption: {settings.live_caption_backend} backend. "
                f"Transcribed: {transcribed[:50]}... Translated: {translated[:50]}..."
            )

            return caption_text
        except Exception as e:
            logger.exception(
                f"Live caption failed with {settings.live_caption_backend}: {e}"
            )
            raise


translator_service = TranslatorService()
