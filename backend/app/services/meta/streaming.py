"""Realtime live-stream orchestration: Meta events -> translated captions.

Consumes raw Muse Voice Transcribe realtime events and emits caption
messages (``partial`` / ``final``) through a caller-provided ``send``
callable. Translation failures degrade to the original transcript text
rather than dropping the caption.
"""

import logging
import time
from typing import Any, Awaitable, Callable

from app.services.meta.voice_client import MetaVoiceError

logger = logging.getLogger(__name__)


class LiveStreamSession:
    """One recording's streaming session (frontend WS <-> Meta WS)."""

    def __init__(
        self,
        *,
        translate_text: Callable[[str, str | None], Awaitable[str]],
        source_language: str | None,
        target_language: str = "en",
        translate: bool = True,
        debounce_ms: int = 600,
    ) -> None:
        self._translate_text = translate_text
        self._source_language = source_language
        self._target_language = target_language
        self._translate = translate
        self._debounce_sec = max(0.0, float(debounce_ms) / 1000.0)
        self._pending_partial: str | None = None
        self._last_sent_partial = ""
        self._last_partial_sent_at = 0.0
        self._turn_id: int | None = None

    async def _render(self, text: str) -> str:
        if not self._translate:
            return text
        if not text.strip():
            return ""
        try:
            rendered = await self._translate_text(text, self._source_language)
        except Exception as exc:
            logger.warning("Stream translation failed, keeping source text: %s", exc)
            return text
        return rendered.strip() or text

    async def handle_event(
        self,
        event: dict[str, Any],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Handle one Meta realtime event, sending caption messages."""
        kind = event.get("type")
        if kind == "error":
            detail = str(event.get("message", "unknown transcription error"))
            raise MetaVoiceError(f"Meta realtime error: {detail[:300]}")

        if kind == "transcript":
            text = str(event.get("transcript") or "")
            turn_id = event.get("turnId")
            if turn_id is not None:
                try:
                    self._turn_id = int(turn_id)
                except (TypeError, ValueError):
                    pass
            if event.get("final"):
                await self._emit_final(text, send)
            elif text and text != self._last_sent_partial:
                await self._emit_partial(text, send)
            return

        if kind == "speechComplete":
            text = str(event.get("transcript") or "")
            turn_id = event.get("turnId")
            if turn_id is not None:
                try:
                    self._turn_id = int(turn_id)
                except (TypeError, ValueError):
                    pass
            if text:
                await self._emit_final(text, send)
            return

        # speechStart / speechEnd / speaker / audioProgress / session:
        # no caption output on their own.

    async def _emit_partial(
        self,
        text: str,
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        now = time.monotonic()
        if self._translate and now - self._last_partial_sent_at < self._debounce_sec:
            self._pending_partial = text
            return
        rendered = await self._render(text)
        self._pending_partial = None
        self._last_sent_partial = text
        self._last_partial_sent_at = now
        await send({"type": "partial", "text": rendered, "turn_id": self._turn_id})

    async def _emit_final(
        self,
        text: str,
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if not text.strip():
            return
        rendered = await self._render(text)
        self._pending_partial = None
        self._last_sent_partial = text
        self._last_partial_sent_at = time.monotonic()
        await send({"type": "final", "text": rendered, "turn_id": self._turn_id})

    async def flush(
        self,
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Emit any debounced partial that never went out (forced)."""
        if self._pending_partial and self._pending_partial != self._last_sent_partial:
            pending = self._pending_partial
            self._pending_partial = None
            rendered = await self._render(pending)
            self._last_sent_partial = pending
            self._last_partial_sent_at = time.monotonic()
            await send(
                {"type": "partial", "text": rendered, "turn_id": self._turn_id}
            )
