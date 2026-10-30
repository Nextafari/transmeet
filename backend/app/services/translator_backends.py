"""
Translator backends: support multiple transcription + translation services.
- GeminiTranslatorBackend: for post-recording captions (high quality)
- WhisperGoogleTranslateBackend: for live captions (fast, cheap, language auto-detect)
- WhisperLocalTranslateBackend: fully local translation (no billing)
"""
import asyncio
import base64
import json
import logging
import os
import re
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import httpx

from app.services.local_translation import LocalTranslationEngine

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

logger = logging.getLogger(__name__)


def _whisper_transcribe_options(
    source_language: str | None,
    *,
    live: bool,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "beam_size": 1 if live else 5,
        "condition_on_previous_text": False,
        "vad_filter": True,
    }
    if source_language:
        options["language"] = source_language
    return options


class TranslatorBackend(ABC):
    """Abstract base for transcription + translation services."""

    @abstractmethod
    async def transcribe_and_translate_audio(
        self,
        audio_path: Path,
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Transcribe and translate audio.

        Returns: (transcribed_text, translated_text)
        """
        pass

    async def transcribe_audio(
        self,
        audio_path: Path,
        source_language: Optional[str] = None,
    ) -> str:
        transcribed, _ = await self.transcribe_and_translate_audio(
            audio_path,
            target_language="en",
            source_language=source_language,
        )
        return transcribed


class GeminiTranslatorBackend(TranslatorBackend):
    """Gemini API backend for high-quality post-processing."""

    def __init__(self, api_key: str, model: str = "gemini-flash-latest"):
        import httpx

        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(timeout=120.0)

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        name = model_name.strip()
        if name.startswith("models/"):
            return name.split("/", 1)[1]
        return name

    @staticmethod
    def _candidate_models(model_name: str) -> list[str]:
        base_name = GeminiTranslatorBackend._normalize_model_name(model_name)
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

    async def _generate_content(
        self,
        parts: list[dict[str, Any]],
        *,
        api_version: str = "v1",
        preferred_model: str | None = None,
    ) -> str:
        """Call Gemini API with fallback logic."""
        last_error: tuple[int, str, str] | None = None

        for version in self._candidate_versions(api_version):
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
            for model_name in self._candidate_models(preferred_model or self.model):
                response = await self.client.post(
                    f"{endpoint_base}{model_name}:generateContent",
                    params={"key": self.api_key},
                    json=payload,
                )

                if response.status_code >= 400:
                    body = response.text[:300]
                    last_error = (response.status_code, body, model_name)
                    if response.status_code == 429:
                        logger.warning(
                            "Gemini quota exceeded (429). Consider using Whisper + Google Translate."
                        )
                        break
                    if response.status_code != 404:
                        break
                    continue

                data = response.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    raise RuntimeError("Gemini returned no candidates")

                parts_list = candidates[0].get("content", {}).get("parts") or []
                for part in parts_list:
                    text = part.get("text")
                    if text:
                        return text

                raise RuntimeError("Gemini response did not include text output")

        if last_error:
            status, body, model_name = last_error
            raise RuntimeError(
                f"Gemini request failed ({status}) for {model_name}: {body}"
            )

        raise RuntimeError("Gemini request failed for unknown reasons")

    @staticmethod
    def _extract_json_payload(raw_text: str) -> Any:
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
            return json.loads(bracketed.group(1))

        raise RuntimeError("Could not parse JSON payload from model output")

    async def transcribe_and_translate_audio(
        self,
        audio_path: Path,
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> tuple[str, str]:
        """Use Gemini to transcribe and translate audio."""
        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        prompt = (
            (
                f"The spoken source language is {source_language}.\n"
                if source_language
                else ""
            ) +
            "Transcribe all spoken content in this audio and translate "
            f"to {target_language}.\n"
            "Return strict JSON array only. Each item must be:\n"
            "{\"start_sec\": number, \"end_sec\": number, "
            "\"source_text\": string, \"translated_text\": string}.\n"
            "Rules:\n"
            "- keep segment duration between 1 and 8 seconds when possible.\n"
            "- include all spoken words.\n"
            "- no markdown, no explanation."
        )

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
            preferred_model=self.model,
        )

        payload = self._extract_json_payload(raw)
        if not isinstance(payload, list):
            raise RuntimeError("Unexpected transcription payload from Gemini")

        # Extract source and translated text from segments
        source_texts = []
        translated_texts = []
        for segment in payload:
            if isinstance(segment, dict):
                source_texts.append(segment.get("source_text", ""))
                translated_texts.append(segment.get("translated_text", ""))

        transcribed = " ".join(t for t in source_texts if t).strip()
        translated = " ".join(t for t in translated_texts if t).strip()

        return transcribed, translated


class WhisperGoogleTranslateBackend(TranslatorBackend):
    """Local faster-whisper + Google Translate API backend (RECOMMENDED for live captions)."""

    def __init__(
        self,
        google_api_key: str,
        google_project_id: str | None,
        whisper_model_size: str = "base",
    ):
        """
        Args:
            google_api_key: Google Cloud API key
            google_project_id: Optional for v2 REST API key flow
            whisper_model_size: 'tiny', 'small', 'base', 'medium'
        """
        from faster_whisper import WhisperModel

        if not google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is required for whisper-google backend"
            )
        self.google_api_key = google_api_key
        self.google_project_id = google_project_id

        # Initialize Whisper model (downloads on first use)
        device = "cuda" if self._has_gpu() else "cpu"
        compute_type = "float16" if self._has_gpu() else "float32"
        logger.info(
            f"Initializing Whisper model '{whisper_model_size}' on device '{device}'"
        )
        self.whisper_model = WhisperModel(
            whisper_model_size, device=device, compute_type=compute_type
        )

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    async def transcribe_and_translate_audio(
        self,
        audio_path: Path,
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        1. Transcribe audio using faster-whisper (local, language auto-detected)
        2. Translate text using Google Translate API (fast, cheap)

        Returns: (transcribed_text, translated_text)
        """
        loop = asyncio.get_event_loop()

        # Transcribe (local, runs in executor to avoid blocking)
        transcribed_text = await loop.run_in_executor(
            None, self._transcribe_local, audio_path, source_language
        )

        if not transcribed_text.strip():
            return "", ""

        # Translate using Google Translate API
        translated_text = await self._translate_text(
            transcribed_text,
            target_language,
            source_language=source_language,
        )

        return transcribed_text, translated_text

    async def transcribe_audio(
        self,
        audio_path: Path,
        source_language: Optional[str] = None,
    ) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._transcribe_local,
            audio_path,
            source_language,
        )

    async def transcribe_and_translate_audio_segments(
        self,
        audio_path: Path,
        target_language: str = "en",
        source_language: Optional[str] = None,
        translate: bool = True,
    ) -> list[dict[str, Any]]:
        started = time.monotonic()
        loop = asyncio.get_event_loop()
        raw_segments, detected_lang = await loop.run_in_executor(
            None,
            self._transcribe_segments_local,
            audio_path,
            source_language,
        )
        transcription_elapsed = time.monotonic() - started
        if not raw_segments:
            logger.info(
                "Whisper captions produced no segments in %.3fs",
                transcription_elapsed,
            )
            return []

        if not translate:
            for segment in raw_segments:
                segment["translated_text"] = segment["source_text"]
            return raw_segments

        translation_started = time.monotonic()
        for segment in raw_segments:
            segment["translated_text"] = await self._translate_text(
                segment["source_text"],
                target_language,
                source_language=source_language,
            )

        logger.info(
            (
                "Whisper-Google caption timings: transcription=%.3fs "
                "translation=%.3fs total=%.3fs segments=%s detected=%s"
            ),
            transcription_elapsed,
            time.monotonic() - translation_started,
            time.monotonic() - started,
            len(raw_segments),
            source_language or detected_lang or "unknown",
        )
        return raw_segments

    def _transcribe_local(
        self,
        audio_path: Path,
        source_language: Optional[str] = None,
    ) -> str:
        """Transcribe audio using faster-whisper (runs in executor)."""
        try:
            options = _whisper_transcribe_options(source_language, live=True)
            segments, info = self.whisper_model.transcribe(
                str(audio_path),
                **options,
            )
            transcribed = " ".join([segment.text for segment in segments])
            detected_lang = source_language or info.language or "unknown"
            logger.info(
                "Whisper source language: %s, Text length: %s",
                detected_lang,
                len(transcribed),
            )
            return transcribed
        except Exception as e:
            logger.exception("Whisper transcription failed: %s", e)
            raise

    def _transcribe_segments_local(
        self,
        audio_path: Path,
        source_language: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], str]:
        options = _whisper_transcribe_options(source_language, live=False)
        segments, info = self.whisper_model.transcribe(str(audio_path), **options)
        raw_segments: list[dict[str, Any]] = []
        for segment in segments:
            text = str(segment.text or "").strip()
            if not text:
                continue
            raw_segments.append(
                {
                    "start_sec": float(segment.start),
                    "end_sec": float(segment.end),
                    "source_text": text,
                    "translated_text": text,
                }
            )

        detected_lang = (source_language or info.language or "").strip().lower()
        logger.info(
            "Whisper source language for captions: %s, segments: %s",
            detected_lang or "unknown",
            len(raw_segments),
        )
        return raw_segments, detected_lang

    async def _translate_text(
        self,
        text: str,
        target_language: str,
        *,
        source_language: str | None = None,
    ) -> str:
        """Call Google Translate v2 REST API with API key (no OAuth token flow)."""
        url = "https://translation.googleapis.com/language/translate/v2"
        payload = {
            "q": text,
            "target": target_language,
            "format": "text",
        }
        if source_language:
            payload["source"] = source_language
        params = {"key": self.google_api_key}

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, params=params, data=payload)
                response.raise_for_status()
                data = response.json()
                translations = (
                    data.get("data", {}).get("translations", [])
                )
                translated = (
                    str(translations[0].get("translatedText", ""))
                    if translations
                    else ""
                )
                logger.info(
                    "Translated to %s: %s...",
                    target_language,
                    translated[:100],
                )
                return translated
            except Exception as e:
                logger.exception("Google Translate REST call failed: %s", e)
                raise


class WhisperLocalTranslateBackend(TranslatorBackend):
    """Local faster-whisper + local translation model (NLLB/Marian)."""

    def __init__(
        self,
        whisper_model_size: str = "base",
        provider: str = "nllb",
        model_name: str = "facebook/nllb-200-distilled-600M",
        translation_engine: LocalTranslationEngine | None = None,
    ):
        from faster_whisper import WhisperModel
        import torch

        # Shared engine: pass one in to reuse a loaded NLLB/Marian model
        # (e.g. with the Meta backends) instead of loading it twice.
        self.translation_engine = translation_engine or LocalTranslationEngine(
            provider=provider,
            model_name=model_name,
        )
        self.provider = self.translation_engine.provider
        self.model_name = self.translation_engine.model_name

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "float32"
        logger.info(
            "Initializing Whisper model '%s' on device '%s'",
            whisper_model_size,
            device,
        )
        self.whisper_model = WhisperModel(
            whisper_model_size, device=device, compute_type=compute_type
        )

    def _transcribe_local(
        self,
        audio_path: Path,
        source_language: str | None = None,
    ) -> tuple[str, str]:
        """Transcribe audio using faster-whisper (runs in executor)."""
        options = _whisper_transcribe_options(source_language, live=True)
        segments, info = self.whisper_model.transcribe(str(audio_path), **options)
        transcribed = " ".join([segment.text for segment in segments])
        detected_lang = (source_language or info.language or "").strip().lower()
        logger.info(
            "Whisper source language: %s, Text length: %s",
            detected_lang or "unknown",
            len(transcribed),
        )
        return transcribed, detected_lang

    def _transcribe_segments_local(
        self,
        audio_path: Path,
        source_language: str | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        options = _whisper_transcribe_options(source_language, live=False)
        segments, info = self.whisper_model.transcribe(str(audio_path), **options)
        raw_segments: list[dict[str, Any]] = []
        for segment in segments:
            text = str(segment.text or "").strip()
            if not text:
                continue
            raw_segments.append(
                {
                    "start_sec": float(segment.start),
                    "end_sec": float(segment.end),
                    "source_text": text,
                    "translated_text": text,
                }
            )

        detected_lang = (source_language or info.language or "").strip().lower()
        logger.info(
            "Whisper source language for local captions: %s, segments: %s",
            detected_lang or "unknown",
            len(raw_segments),
        )
        return raw_segments, detected_lang

    async def transcribe_and_translate_audio(
        self,
        audio_path: Path,
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> tuple[str, str]:
        loop = asyncio.get_event_loop()
        started = time.monotonic()
        transcribed_text, detected_lang = await loop.run_in_executor(
            None,
            self._transcribe_local,
            audio_path,
            source_language,
        )
        transcription_elapsed = time.monotonic() - started

        if not transcribed_text.strip():
            logger.info(
                "Local live caption produced no transcription in %.3fs",
                transcription_elapsed,
            )
            return "", ""

        lang_for_source = source_language or detected_lang
        if (
            self.translation_engine.provider != "marian"
            and self.translation_engine.same_language(
                lang_for_source, target_language
            )
        ):
            logger.info(
                "Local live caption skipped translation: source and target are both %s",
                target_language,
            )
            return transcribed_text, transcribed_text

        if self.translation_engine.provider != "marian" and not self.translation_engine.map_language(
            lang_for_source
        ):
            logger.warning(
                (
                    "Local live caption skipped unsafe translation: "
                    "unsupported or unknown source language '%s'"
                ),
                lang_for_source or "unknown",
            )
            return transcribed_text, transcribed_text

        translation_started = time.monotonic()
        try:
            translated_text = await loop.run_in_executor(
                None,
                self.translation_engine.translate,
                transcribed_text,
                lang_for_source,
                target_language,
            )
        except Exception as exc:
            logger.exception(
                "Local translation failed after %.3fs: %s",
                time.monotonic() - translation_started,
                exc,
            )
            raise RuntimeError(f"Local translation failed: {exc}") from exc

        if not translated_text.strip():
            raise RuntimeError("Local translation returned empty text")

        logger.info(
            (
                "Local live caption timings: transcription=%.3fs "
                "translation=%.3fs total=%.3fs source_len=%s translated_len=%s"
            ),
            transcription_elapsed,
            time.monotonic() - translation_started,
            time.monotonic() - started,
            len(transcribed_text),
            len(translated_text),
        )

        return transcribed_text, translated_text

    async def transcribe_and_translate_audio_segments(
        self,
        audio_path: Path,
        target_language: str = "en",
        source_language: Optional[str] = None,
        translate: bool = True,
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        started = time.monotonic()
        raw_segments, detected_lang = await loop.run_in_executor(
            None,
            self._transcribe_segments_local,
            audio_path,
            source_language,
        )
        transcription_elapsed = time.monotonic() - started
        if not raw_segments:
            logger.info(
                "Local captions produced no segments in %.3fs",
                transcription_elapsed,
            )
            return []

        lang_for_source = source_language or detected_lang
        same_language = (
            self.translation_engine.provider != "marian"
            and self.translation_engine.same_language(
                lang_for_source, target_language
            )
        )
        unsupported_source = (
            self.translation_engine.provider != "marian"
            and translate
            and not self.translation_engine.map_language(lang_for_source)
        )
        if not translate or same_language or unsupported_source:
            for segment in raw_segments:
                segment["translated_text"] = segment["source_text"]
            logger.info(
                (
                    "Local captions skipped translation: translate=%s "
                    "same_language=%s unsupported_source=%s segments=%s"
                ),
                translate,
                same_language,
                unsupported_source,
                len(raw_segments),
            )
            return raw_segments

        translation_started = time.monotonic()
        try:
            for segment in raw_segments:
                translated_text = await loop.run_in_executor(
                    None,
                    self.translation_engine.translate,
                    segment["source_text"],
                    lang_for_source,
                    target_language,
                )
                if not translated_text.strip():
                    raise RuntimeError("Local segment translation returned empty text")
                segment["translated_text"] = translated_text
        except Exception as exc:
            logger.exception(
                "Local caption segment translation failed after %.3fs: %s",
                time.monotonic() - translation_started,
                exc,
            )
            raise RuntimeError(f"Local caption translation failed: {exc}") from exc

        logger.info(
            (
                "Local caption timings: transcription=%.3fs translation=%.3fs "
                "total=%.3fs segments=%s"
            ),
            transcription_elapsed,
            time.monotonic() - translation_started,
            time.monotonic() - started,
            len(raw_segments),
        )
        return raw_segments

    async def transcribe_audio(
        self,
        audio_path: Path,
        source_language: Optional[str] = None,
    ) -> str:
        loop = asyncio.get_event_loop()
        transcribed_text, _ = await loop.run_in_executor(
            None,
            self._transcribe_local,
            audio_path,
            source_language,
        )
        return transcribed_text


def get_translator_backend(
    backend_type: str,
    **kwargs,
) -> TranslatorBackend:
    """Factory function to create translator backend."""
    if backend_type == "gemini":
        return GeminiTranslatorBackend(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "gemini-flash-latest"),
        )
    elif backend_type == "whisper-google":
        return WhisperGoogleTranslateBackend(
            google_api_key=kwargs.get("google_api_key"),
            google_project_id=kwargs.get("google_project_id"),
            whisper_model_size=kwargs.get("whisper_model_size", "base"),
        )
    elif backend_type == "whisper-local":
        return WhisperLocalTranslateBackend(
            whisper_model_size=kwargs.get("whisper_model_size", "base"),
            provider=kwargs.get("local_translation_provider", "nllb"),
            model_name=kwargs.get(
                "local_translation_model",
                "facebook/nllb-200-distilled-600M",
            ),
        )
    else:
        raise ValueError(f"Unknown backend: {backend_type}")
