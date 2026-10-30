"""Muse Spark translation leg (Meta Model API, OpenAI-compatible chat).

Used as one translation leg for Meta-powered backends when configured.
Docs: https://dev.meta.ai/docs/models (Spark section) and the Chat
Completions protocol; base URL ``https://api.meta.ai/v1``.
"""

import json
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SPARK_BASE_URL = "https://api.meta.ai/v1"


class MetaSparkError(RuntimeError):
    """A Muse Spark failure. ``quota`` marks rate-limit/backoff."""

    def __init__(self, message: str, *, quota: bool = False) -> None:
        super().__init__(message)
        self.quota = quota


def _extract_translated_text(raw_text: str) -> str:
    text = (raw_text or "").strip()

    def _from_json(candidate: str) -> str | None:
        try:
            payload = json.loads(candidate)
        except (ValueError, TypeError):
            return None
        if isinstance(payload, dict):
            translated = str(payload.get("translated_text") or "").strip()
            return translated or None
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
    raise MetaSparkError("Spark returned no translatable JSON payload")


class MetaSparkClient:
    """Minimal Chat Completions client for text translation."""

    def __init__(
        self,
        api_key: str | None,
        model: str = "muse-spark-1.3",
        base_url: str = SPARK_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        if not (api_key or "").strip():
            raise MetaSparkError(
                "META_MODEL_API_KEY is not configured; the 'spark' "
                "translation leg cannot run without it."
            )
        self._api_key = api_key.strip()
        self._model = (model or "").strip() or "muse-spark-1.3"
        self._base_url = (base_url or "").rstrip("/") or SPARK_BASE_URL
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    async def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> str:
        if not (text or "").strip():
            return ""
        source_hint = (
            f"The source language is {source_language}.\n" if source_language else ""
        )
        payload = {
            "model": self._model,
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
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
        except MetaSparkError:
            raise
        except Exception as exc:
            raise MetaSparkError(f"Spark translation request failed: {exc}") from exc

        if response.status_code == 429:
            raise MetaSparkError(
                f"Spark translation rate limited (429): {response.text[:300]}",
                quota=True,
            )
        if response.status_code == 401:
            raise MetaSparkError(
                "Spark translation rejected the credential (401). "
                "Check META_MODEL_API_KEY."
            )
        if response.status_code != 200:
            raise MetaSparkError(
                f"Spark translation failed ({response.status_code}): "
                f"{response.text[:300]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise MetaSparkError(
                f"Spark translation returned invalid JSON: {exc}"
            ) from exc
        choices = data.get("choices") if isinstance(data, dict) else None
        content: Any = None
        if choices:
            message = (choices[0] or {}).get("message") or {}
            content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise MetaSparkError("Spark translation returned empty content")
        translated = _extract_translated_text(content)
        logger.info("Spark translated to %s: %s...", target_language, translated[:80])
        return translated
