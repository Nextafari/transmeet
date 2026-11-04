"""Baseten Model APIs client (OpenAI-compatible chat completions).

Covers both Baseten uses: text translation on a chat slug and speech
transcription via ``audio_url`` on an audio-capable slug.
Docs: https://docs.baseten.co/inference/model-apis/overview and the Chat
Completions reference; base URL ``https://inference.baseten.co/v1``.
"""

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASETEN_BASE_URL = "https://inference.baseten.co/v1"
DEFAULT_CHAT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
DEFAULT_ASR_MODEL = "meta-models/muse-spark-open"
# Per the audio docs: 25 MB and 5 minutes per file on Muse Spark Open.
MAX_AUDIO_BYTES = 25 * 1024 * 1024

_AUDIO_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
}


class BasetenError(RuntimeError):
    """A Baseten failure. ``quota`` marks rate-limit/backoff."""

    def __init__(self, message: str, *, quota: bool = False) -> None:
        super().__init__(message)
        self.quota = quota


def _extract_json_field(raw_text: str, field: str) -> str:
    text = (raw_text or "").strip()

    def _from_json(candidate: str) -> str | None:
        try:
            payload = json.loads(candidate)
        except (ValueError, TypeError):
            return None
        if isinstance(payload, dict):
            value = str(payload.get(field) or "").strip()
            return value or None
        return None

    direct = _from_json(text)
    if direct:
        return direct
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        nested = _from_json(fenced.group(1).strip())
        if nested:
            return nested
    bracketed = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if bracketed:
        nested = _from_json(bracketed.group(1))
        if nested:
            return nested
    raise BasetenError(f"Baseten returned no JSON payload with {field!r}")


class BasetenClient:
    """Minimal chat-completions client for translation and transcription."""

    def __init__(
        self,
        api_key: str | None,
        *,
        chat_model: str = DEFAULT_CHAT_MODEL,
        asr_model: str = DEFAULT_ASR_MODEL,
        base_url: str = BASETEN_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        if not (api_key or "").strip():
            raise BasetenError(
                "BASETEN_API_KEY is not configured; Baseten translation "
                "and transcription cannot run without it."
            )
        self._api_key = api_key.strip()
        self._chat_model = (chat_model or "").strip() or DEFAULT_CHAT_MODEL
        self._asr_model = (asr_model or "").strip() or DEFAULT_ASR_MODEL
        self._base_url = (base_url or "").rstrip("/") or BASETEN_BASE_URL
        self._timeout = timeout

    @property
    def chat_model(self) -> str:
        return self._chat_model

    @property
    def asr_model(self) -> str:
        return self._asr_model

    async def _chat(
        self,
        payload: dict[str, Any],
        *,
        operation: str,
        timeout: float | None = None,
    ) -> str:
        request_timeout = self._timeout if timeout is None else timeout
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                    timeout=request_timeout,
                )
        except BasetenError:
            raise
        except Exception as exc:
            raise BasetenError(
                f"Baseten {operation} request failed: {exc}"
            ) from exc

        if response.status_code == 429:
            raise BasetenError(
                f"Baseten {operation} rate limited (429): "
                f"{response.text[:300]}",
                quota=True,
            )
        if response.status_code == 401:
            raise BasetenError(
                f"Baseten {operation} rejected the credential (401). "
                "Check BASETEN_API_KEY."
            )
        if response.status_code != 200:
            raise BasetenError(
                f"Baseten {operation} failed ({response.status_code}): "
                f"{response.text[:300]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise BasetenError(
                f"Baseten {operation} returned invalid JSON: {exc}"
            ) from exc
        choices = data.get("choices") if isinstance(data, dict) else None
        content: Any = None
        if choices:
            message = (choices[0] or {}).get("message") or {}
            content = message.get("content")
        if isinstance(content, list):
            # Some models return content blocks; join their text parts.
            content = " ".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise BasetenError(
                f"Baseten {operation} returned empty content"
            )
        return content

    async def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
        timeout: float | None = None,
    ) -> str:
        if not (text or "").strip():
            return ""
        source_hint = (
            f"The source language is {source_language}.\n" if source_language else ""
        )
        content = await self._chat(
            {
                "model": self._chat_model,
                "temperature": 0.1,
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Translate the user text to the target language. "
                            'Return strict JSON only: {"translated_text": "..."}. '
                            "No markdown, no explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{source_hint}Target language: {target_language}\n"
                            f"Text: {text}"
                        ),
                    },
                ],
            },
            operation="translation",
            timeout=timeout,
        )
        translated = _extract_json_field(content, "translated_text")
        logger.info(
            "Baseten translated to %s: %s...", target_language, translated[:80]
        )
        return translated

    async def transcribe_audio(
        self,
        audio_path: Path,
        source_language: str | None = None,
    ) -> str:
        raw = audio_path.read_bytes()
        if len(raw) > MAX_AUDIO_BYTES:
            raise BasetenError(
                f"Audio is {len(raw) / 1024 / 1024:.1f} MB; Baseten caps "
                "audio files at 25 MB."
            )
        mime = _AUDIO_MIME_TYPES.get(audio_path.suffix.lower(), "audio/wav")
        audio_url = (
            f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        )
        source_hint = (
            f"The spoken language is {source_language}.\n" if source_language else ""
        )
        content = await self._chat(
            {
                "model": self._asr_model,
                "temperature": 0.1,
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"{source_hint}Transcribe the speech in "
                                    "this recording exactly. Return strict "
                                    'JSON only: {"transcript": "..."}. No '
                                    "markdown, no explanation."
                                ),
                            },
                            {"type": "audio_url", "audio_url": {"url": audio_url}},
                        ],
                    }
                ],
            },
            operation="transcription",
        )
        transcript = _extract_json_field(content, "transcript")
        logger.info("Baseten transcribed %s chars", len(transcript))
        return transcript
