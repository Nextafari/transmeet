"""Realtime live-stream orchestration: Meta events -> translated captions.

Consumes raw Muse Voice Transcribe realtime events and emits caption
messages (``partial`` / ``final``) through a caller-provided ``send``
callable.

Latency design: ``handle_event`` never waits for translation, so the Meta
pump keeps draining while a translation is in flight and captions cannot
fall further behind on long turns. When translation is enabled, partials
are translated in a throttled background worker (latest-wins, at most one
translation per ``partial_translate_ms``) while finals are translated in
order by a separate worker that drops backlog (newer turns queued) and
stale turns before translating. Overlong inputs are truncated to their
tail so the overlay tracks recent speech. When translation is disabled,
partials go out raw and throttled only for display. Translation failures
degrade to the original transcript text rather than dropping the caption.
"""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

from app.services.meta.voice_client import MetaVoiceError

logger = logging.getLogger(__name__)

# Upper bound for flush() waiting on in-flight final translations.
FLUSH_TIMEOUT_SEC = 15.0
# Upper bound for flush() waiting on a pending partial translation and for
# the partial worker to settle. Keeps clean close from hanging on a slow leg.
PARTIAL_FLUSH_TIMEOUT_SEC = 5.0
# Overlong cumulative partials/finals are truncated to this tail before
# translation so the overlay tracks recent speech instead of re-translating
# stale prefixes at growing cost.
TRANSLATION_TAIL_CHARS = 600
# A queued final with this many newer finals behind it is dropped without
# translating so a slow leg cannot trail minutes behind the speaker.
FINALS_BACKLOG_DROP_QSIZE = 2


def _translation_tail(text: str) -> str:
    if len(text) <= TRANSLATION_TAIL_CHARS:
        return text
    tail = text[-TRANSLATION_TAIL_CHARS:]
    _head, sep, rest = tail.partition(" ")
    return rest if sep else tail


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
        partial_translate_ms: int = 1000,
    ) -> None:
        self._translate_text = translate_text
        self._source_language = source_language
        self._target_language = target_language
        self._translate = translate
        self._debounce_sec = max(0.0, float(debounce_ms) / 1000.0)
        self._partial_translate_sec = max(
            0.0, float(partial_translate_ms) / 1000.0
        )
        self._pending_partial: str | None = None
        self._last_sent_partial = ""
        self._last_partial: tuple[int | None, str] = (None, "")
        self._last_final: tuple[int | None, str] | None = None
        self._last_partial_sent_at = 0.0
        self._turn_id: int | None = None
        self._max_sent_turn: int | None = None
        self._max_final_turn: int | None = None
        self._finals: asyncio.Queue[
            tuple[str, int | None, Callable[[dict[str, Any]], Awaitable[None]]]
        ] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._partial_pending: (
            tuple[int, str, int | None, Callable[[dict[str, Any]], Awaitable[None]]]
            | None
        ) = None
        self._partial_worker_task: asyncio.Task[None] | None = None
        self._partial_seq = 0
        self._last_sent_partial_seq = 0
        self._last_partial_translate_at = 0.0

    async def _render(self, text: str) -> str:
        if not self._translate:
            return text
        if not text.strip():
            return ""
        text = _translation_tail(text)
        try:
            rendered = await self._translate_text(text, self._source_language)
        except Exception as exc:
            logger.warning("Stream translation failed, keeping source text: %s", exc)
            return text
        return rendered.strip() or text

    def _note_turn(self, event: dict[str, Any]) -> None:
        turn_id = event.get("turnId")
        if turn_id is not None:
            try:
                self._turn_id = int(turn_id)
            except (TypeError, ValueError):
                pass

    def _mark_sent_turn(self, turn_id: int | None) -> None:
        if turn_id is None:
            return
        if self._max_sent_turn is None or turn_id > self._max_sent_turn:
            self._max_sent_turn = turn_id

    def _mark_final_turn(self, turn_id: int | None) -> None:
        if turn_id is None:
            return
        if self._max_final_turn is None or turn_id > self._max_final_turn:
            self._max_final_turn = turn_id

    def _is_stale(self, turn_id: int | None) -> bool:
        return (
            turn_id is not None
            and self._max_sent_turn is not None
            and turn_id < self._max_sent_turn
        )

    def _is_partial_stale(self, turn_id: int | None, seq: int) -> bool:
        if seq <= self._last_sent_partial_seq:
            return True
        if (
            turn_id is not None
            and self._max_final_turn is not None
            and turn_id <= self._max_final_turn
        ):
            return True
        return self._is_stale(turn_id)

    def _start_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._final_worker())

    def _start_partial_worker(self) -> None:
        if (
            self._partial_worker_task is None
            or self._partial_worker_task.done()
        ):
            self._partial_worker_task = asyncio.create_task(self._partial_loop())

    async def _final_worker(self) -> None:
        """Translate queued finals in order; never dies on send errors."""
        while True:
            text, turn_id, send = await self._finals.get()
            try:
                if self._finals.qsize() >= FINALS_BACKLOG_DROP_QSIZE:
                    logger.debug(
                        "Dropping backlog turn %s final; newer turns queued",
                        turn_id,
                    )
                    continue
                if self._is_stale(turn_id):
                    logger.debug(
                        "Dropping stale turn %s final before translating",
                        turn_id,
                    )
                    continue
                rendered = await self._render(text)
                if self._is_stale(turn_id):
                    logger.debug(
                        "Dropping stale turn %s final; turn %s already shown",
                        turn_id,
                        self._max_sent_turn,
                    )
                    continue
                self._mark_sent_turn(turn_id)
                self._mark_final_turn(turn_id)
                try:
                    await send(
                        {"type": "final", "text": rendered, "turn_id": turn_id}
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("Stream final send failed: %s", exc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Stream final worker failed: %s", exc)
            finally:
                self._finals.task_done()

    async def _partial_loop(self) -> None:
        """Translate latest pending partials; coalesces rapid updates."""
        while self._partial_pending is not None:
            now = time.monotonic()
            wait = self._partial_translate_sec - (
                now - self._last_partial_translate_at
            )
            if wait > 0:
                try:
                    await asyncio.sleep(wait)
                except asyncio.CancelledError:
                    raise
                if self._partial_pending is None:
                    break
            pending = self._partial_pending
            self._partial_pending = None
            if pending is None:
                break
            seq, text, turn_id, send = pending
            if self._is_partial_stale(turn_id, seq):
                continue
            self._last_partial_translate_at = time.monotonic()
            try:
                rendered = await self._render(text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Stream partial translation failed: %s", exc)
                rendered = text
            if self._is_partial_stale(turn_id, seq):
                continue
            try:
                await send(
                    {"type": "partial", "text": rendered, "turn_id": turn_id}
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Stream partial send failed: %s", exc)
                continue
            self._last_sent_partial_seq = seq
            self._last_sent_partial = text
            self._last_partial_sent_at = time.monotonic()
            self._mark_sent_turn(turn_id)

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
            self._note_turn(event)
            if event.get("final"):
                await self._emit_final(text, send)
            elif text:
                if self._translate or text != self._last_sent_partial:
                    await self._emit_partial(text, send)
            return

        if kind == "speechComplete":
            text = str(event.get("transcript") or "")
            self._note_turn(event)
            await self._emit_final(text, send)
            return

        # speechStart / speechEnd / speaker / audioProgress / session:
        # no caption output on their own.

    async def _emit_partial(
        self,
        text: str,
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if self._translate:
            turn_id = self._turn_id
            if text == self._last_partial[1] and turn_id == self._last_partial[0]:
                return
            self._last_partial = (turn_id, text)
            self._partial_seq += 1
            self._partial_pending = (self._partial_seq, text, turn_id, send)
            self._start_partial_worker()
            return
        # Raw path (translate=False): track speech with display throttle only.
        self._last_partial = (self._turn_id, text)
        now = time.monotonic()
        if now - self._last_partial_sent_at < self._debounce_sec:
            self._pending_partial = text
            return
        self._pending_partial = None
        self._last_sent_partial = text
        self._last_partial_sent_at = now
        self._mark_sent_turn(self._turn_id)
        await send({"type": "partial", "text": text, "turn_id": self._turn_id})

    def _fallback_text(self) -> str:
        """Latest partial text usable when Meta finalizes a turn empty."""
        if self._partial_pending and self._partial_pending[1].strip():
            return self._partial_pending[1]
        if self._pending_partial and self._pending_partial.strip():
            return self._pending_partial
        partial_turn, partial_text = self._last_partial
        if not partial_text.strip():
            return ""
        if (
            self._turn_id is not None
            and partial_turn is not None
            and partial_turn != self._turn_id
        ):
            return ""
        return partial_text

    async def _emit_final(
        self,
        text: str,
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        raw = text.strip()
        if not raw:
            # Endpointing can finalize a turn (e.g. a short switched-language
            # fragment) with no transcript; keep the turn's latest partial
            # instead of dropping that slice of speech.
            raw = self._fallback_text().strip()
            if not raw:
                return
        turn_id = self._turn_id
        if self._last_final == (turn_id, raw):
            # Meta may finalize a turn twice (final transcript event plus
            # speechComplete); the second copy adds nothing.
            self._pending_partial = None
            self._partial_pending = None
            return
        self._last_final = (turn_id, raw)
        self._pending_partial = None
        self._partial_pending = None
        self._last_sent_partial = raw
        if not self._translate:
            self._mark_sent_turn(turn_id)
            self._mark_final_turn(turn_id)
            await send({"type": "final", "text": raw, "turn_id": turn_id})
            return
        self._start_worker()
        self._finals.put_nowait((raw, turn_id, send))

    async def _flush_partial(
        self,
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        pending = self._partial_pending
        self._partial_pending = None
        if pending is not None:
            seq, text, turn_id, _stored_send = pending
            if not self._is_partial_stale(turn_id, seq):
                try:
                    rendered = await asyncio.wait_for(
                        self._render(text),
                        timeout=PARTIAL_FLUSH_TIMEOUT_SEC,
                    )
                except (TimeoutError, asyncio.TimeoutError):
                    logger.warning(
                        "Stream partial flush timed out; keeping source text"
                    )
                    rendered = text
                if not self._is_partial_stale(turn_id, seq):
                    try:
                        await send(
                            {
                                "type": "partial",
                                "text": rendered,
                                "turn_id": turn_id,
                            }
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning("Stream partial flush send failed: %s", exc)
                    else:
                        self._last_sent_partial_seq = seq
                        self._last_sent_partial = text
                        self._last_partial_sent_at = time.monotonic()
                        self._mark_sent_turn(turn_id)
        worker = self._partial_worker_task
        if worker is not None and not worker.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(worker),
                    timeout=PARTIAL_FLUSH_TIMEOUT_SEC,
                )
            except (TimeoutError, asyncio.TimeoutError):
                logger.warning(
                    "Stream partial worker still busy at flush; closing anyway"
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Stream partial worker flush failed: %s", exc)

    async def flush(
        self,
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Emit any throttled partial, then wait for queued finals."""
        if self._translate:
            try:
                await self._flush_partial(send)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Stream partial flush failed: %s", exc)
        elif (
            self._pending_partial
            and self._pending_partial != self._last_sent_partial
        ):
            pending = self._pending_partial
            self._pending_partial = None
            self._last_sent_partial = pending
            self._last_partial_sent_at = time.monotonic()
            self._mark_sent_turn(self._turn_id)
            await send(
                {"type": "partial", "text": pending, "turn_id": self._turn_id}
            )
        if self._finals.empty():
            return
        try:
            await asyncio.wait_for(self._finals.join(), timeout=FLUSH_TIMEOUT_SEC)
        except TimeoutError:
            logger.warning(
                "Stream flush timed out with finals still queued; closing anyway"
            )

    async def aclose(self) -> None:
        """Stop translation workers, discarding anything queued."""
        worker, self._worker_task = self._worker_task, None
        if worker is not None:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        partial_worker, self._partial_worker_task = (
            self._partial_worker_task,
            None,
        )
        self._partial_pending = None
        if partial_worker is not None:
            partial_worker.cancel()
            try:
                await partial_worker
            except asyncio.CancelledError:
                pass
