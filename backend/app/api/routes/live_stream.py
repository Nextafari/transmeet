"""Realtime live-stream route: frontend PCM -> Meta -> translated captions.

Transport fields only come from the query string (source/target languages,
translate flag). Model and engine selection is server-side, from settings;
any ``engine``/``backend``/``model`` query keys are ignored.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.services.meta.events import StreamServerMessage, parse_stream_params
from app.services.meta.streaming import LiveStreamSession
from app.services.meta.translation import translate_with_legs
from app.services.meta.voice_client import MetaVoiceClient, MetaVoiceError
from app.services.translator import translator_service

router = APIRouter()

logger = logging.getLogger(__name__)


class _ClientClosed(Exception):
    pass


class _ClientDone(Exception):
    pass


@router.websocket("/live-stream")
async def live_stream(websocket: WebSocket) -> None:
    params = parse_stream_params(websocket.query_params)
    await websocket.accept()

    if not settings.live_stream_enabled or not (
        settings.meta_model_api_key or ""
    ).strip():
        await websocket.send_json(
            StreamServerMessage(
                type="downgrade", reason="live streaming is not enabled"
            ).model_dump()
        )
        await websocket.close()
        return

    language_bias = [params.source_language] if params.source_language else None
    meta = MetaVoiceClient(
        settings.meta_model_api_key,
        model=settings.meta_asr_model,
        mode=settings.meta_asr_mode,
        language_bias=language_bias,
        timeout=settings.live_stream_op_timeout_seconds,
    )
    try:
        await meta.connect()
    except MetaVoiceError as exc:
        logger.warning("Live stream Meta connect failed: %s", exc)
        await _send_downgrade(websocket, str(exc))
        await websocket.close()
        await meta.aclose()
        return

    # Resolve translation legs once per session (first call may load NLLB).
    local_engine = await asyncio.to_thread(
        translator_service._get_shared_local_engine
    )
    spark_client = await asyncio.to_thread(translator_service._get_spark_client)

    async def translate_turn(text: str, source: str | None) -> str:
        translated, _leg = await translate_with_legs(
            text,
            source_language=source,
            target_language=params.target_language,
            primary=settings.meta_translation_engine,
            fallbacks=settings.stream_translation_fallbacks,
            local_engine=local_engine,
            spark_client=spark_client,
        )
        return translated

    session = LiveStreamSession(
        translate_text=translate_turn,
        source_language=params.source_language,
        target_language=params.target_language,
        translate=params.translate,
        debounce_ms=settings.live_stream_partial_debounce_ms,
    )

    async def send_to_client(message: dict) -> None:
        await websocket.send_json(message)

    async def pump_client_to_meta() -> None:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise _ClientClosed()
            data = message.get("bytes")
            if data:
                await meta.send_pcm(bytes(data))
                continue
            text = message.get("text")
            if not text:
                continue
            try:
                frame = json.loads(text)
            except ValueError:
                continue
            if isinstance(frame, dict) and frame.get("type") == "end":
                await meta.end_stream()
                raise _ClientDone()

    async def pump_meta_to_client() -> None:
        while True:
            event = await meta.recv_event()
            await session.handle_event(event, send_to_client)

    client_task = asyncio.create_task(pump_client_to_meta())
    meta_task = asyncio.create_task(pump_meta_to_client())
    try:
        done, pending = await asyncio.wait(
            {client_task, meta_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        failure: BaseException | None = None
        for task in done:
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                continue
            if isinstance(exc, _ClientClosed):
                return
            if isinstance(exc, _ClientDone):
                try:
                    await session.flush(send_to_client)
                except Exception:
                    pass
                return
            if exc is not None:
                failure = exc

        if isinstance(failure, MetaVoiceError):
            logger.warning("Live stream Meta failed: %s", failure)
            await _send_downgrade(websocket, str(failure))
        elif failure is not None:
            logger.warning("Live stream failed: %s", failure)
            await _send_downgrade(websocket, str(failure))
    except WebSocketDisconnect:
        pass
    finally:
        await meta.aclose()
        try:
            await websocket.close()
        except Exception:
            pass


async def _send_downgrade(websocket: WebSocket, reason: str) -> None:
    try:
        await websocket.send_json(
            StreamServerMessage(
                type="downgrade", reason=reason[:180]
            ).model_dump()
        )
    except Exception:
        pass
