"""WS live-stream route: downgrade rules and the streaming happy path."""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.routes.live_stream as live_stream_module
from app.api.routes.live_stream import router as live_stream_router


def _client():
    app = FastAPI()
    app.include_router(live_stream_router)
    return TestClient(app)


def _settings(monkeypatch, enabled=True, key=None):
    from app.core.config import settings

    monkeypatch.setattr(settings, "live_stream_enabled", enabled)
    monkeypatch.setattr(settings, "meta_model_api_key", key)
    monkeypatch.setattr(settings, "meta_translation_engine", "nllb")
    monkeypatch.setattr(settings, "stream_translation_fallbacks", [])
    return settings


class FakeSparkTranslate:
    """Instant Spark stand-in so captions never touch the local engine."""

    async def translate_text(
        self, text, target, source_language=None, timeout=None
    ):
        return f"{text}|en"


def test_downgrade_when_key_missing(monkeypatch):
    _settings(monkeypatch, enabled=True, key=None)
    with _client().websocket_connect(
        "/live-stream?target_language=es&engine=gemini&model=x"
    ) as websocket:
        message = websocket.receive_json()
    assert message["type"] == "downgrade"


def test_downgrade_when_streaming_disabled(monkeypatch):
    _settings(monkeypatch, enabled=False, key="meta-key")
    with _client().websocket_connect("/live-stream?target_language=es") as websocket:
        message = websocket.receive_json()
    assert message["type"] == "downgrade"


def test_client_engine_params_ignored_on_connect_failure(monkeypatch):
    from app.services.meta.voice_client import MetaVoiceError

    _settings(monkeypatch, enabled=True, key="meta-key")

    class FailingMeta:
        def __init__(self, *args, **kwargs):
            pass

        async def connect(self):
            raise MetaVoiceError("boom")

        async def aclose(self):
            pass

    monkeypatch.setattr(live_stream_module, "MetaVoiceClient", FailingMeta)
    with _client().websocket_connect(
        "/live-stream?target_language=es&engine=whisper-local&backend=gemini"
    ) as websocket:
        message = websocket.receive_json()
    assert message["type"] == "downgrade"
    assert "boom" in (message.get("reason") or "")


class ScriptedMeta:
    """Fake Meta session: one partial, one final, then silence."""

    def __init__(self, *args, **kwargs):
        self.sent_pcm = []
        self._step = 0

    async def connect(self):
        return "test-session"

    async def send_pcm(self, data):
        self.sent_pcm.append(bytes(data))

    async def recv_event(self):
        self._step += 1
        if self._step == 1:
            return {"type": "transcript", "transcript": "hola", "final": False}
        if self._step == 2:
            return {"type": "speechComplete", "turnId": 1, "transcript": "Hola."}
        await asyncio.sleep(3600)

    async def end_stream(self):
        pass

    async def aclose(self):
        pass


def test_streaming_happy_path_passthrough(monkeypatch):
    _settings(monkeypatch, enabled=True, key="meta-key")
    monkeypatch.setattr(live_stream_module, "MetaVoiceClient", ScriptedMeta)

    with _client().websocket_connect(
        "/live-stream?source_language=es&target_language=en&translate=false"
    ) as websocket:
        websocket.send_bytes(b"\x00\x01" * 160)
        first = websocket.receive_json()
        second = websocket.receive_json()

    assert first == {"type": "partial", "text": "hola", "turn_id": None}
    assert second == {"type": "final", "text": "Hola.", "turn_id": 1}


def test_streaming_starts_before_local_engine_ready(monkeypatch):
    """Session setup must not block on the fallback engine's model load.

    The local engine may download and load gigabytes on first use; audio
    pumps and Spark translation must flow while it is still loading.
    """
    import threading

    from app.services.translator import translator_service

    _settings(monkeypatch, enabled=True, key="meta-key")
    monkeypatch.setattr(live_stream_module, "MetaVoiceClient", ScriptedMeta)
    from app.core.config import settings

    monkeypatch.setattr(settings, "meta_translation_engine", "spark")
    monkeypatch.setattr(settings, "stream_translation_fallbacks", [])

    release = threading.Event()

    def slow_engine():
        assert release.wait(timeout=30)
        return None

    monkeypatch.setattr(
        translator_service, "_get_shared_local_engine", slow_engine
    )
    monkeypatch.setattr(
        translator_service, "_get_spark_client", lambda: FakeSparkTranslate()
    )
    monkeypatch.setattr(translator_service, "_get_baseten_client", lambda: None)

    # Backstop: unblock the loader so a regression fails instead of hanging.
    backstop = threading.Timer(10.0, release.set)
    backstop.daemon = True
    backstop.start()
    try:
        with _client().websocket_connect(
            "/live-stream?source_language=es&target_language=en&translate=true"
        ) as websocket:
            websocket.send_bytes(b"\x00\x01" * 160)
            # The back-to-back final supersedes the pending partial before
            # its worker ticks, so only the translated final arrives.
            first = websocket.receive_json()
            engine_still_loading = not release.is_set()
            # Release the loader thread before exit: portal shutdown joins
            # executor threads, so holding it would stall disconnect.
            release.set()
    finally:
        backstop.cancel()
        release.set()

    # The caption arrived while the local engine was still loading.
    assert engine_still_loading
    assert first == {"type": "final", "text": "Hola.|en", "turn_id": 1}


def test_clean_end_flushes_and_closes(monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    _settings(monkeypatch, enabled=True, key="meta-key")
    monkeypatch.setattr(live_stream_module, "MetaVoiceClient", ScriptedMeta)
    monkeypatch.setattr(live_stream_module, "TRAILING_EVENT_GRACE_SEC", 0.05)

    with _client().websocket_connect(
        "/live-stream?source_language=es&target_language=en&translate=false"
    ) as websocket:
        websocket.send_bytes(b"\x00\x01" * 160)
        first = websocket.receive_json()
        second = websocket.receive_json()
        websocket.send_text('{"type": "end"}')
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()

    assert first == {"type": "partial", "text": "hola", "turn_id": None}
    assert second == {"type": "final", "text": "Hola.", "turn_id": 1}


@pytest.mark.asyncio()
async def test_config_endpoint_advertises_streaming(monkeypatch):
    from app.api.routes.media import get_live_caption_config

    _settings(monkeypatch, enabled=True, key="meta-key")
    config = await get_live_caption_config()
    assert config["streaming"]["enabled"] is True
    assert config["streaming"]["ws_path"] == "/api/v1/media/live-stream"
    assert config["streaming"]["encoding"] == "PCM_16KHZ"
    assert isinstance(config["backends"], list) and config["backends"]

    _settings(monkeypatch, enabled=True, key=None)
    assert (await get_live_caption_config())["streaming"]["enabled"] is False


@pytest.mark.asyncio()
async def test_config_chunk_pacing(monkeypatch):
    from app.api.routes.media import get_live_caption_config
    from app.core.config import settings

    _settings(monkeypatch, enabled=True, key="meta-key")
    monkeypatch.setattr(settings, "live_caption_chunk_seconds", 1.0)
    monkeypatch.setattr(settings, "live_caption_min_interval_seconds", 1)
    config = await get_live_caption_config()
    assert config["chunk_ms"] == 1000
    assert config["min_interval_ms"] == 1000

    # Sub-second chunks are allowed (floored at 500 ms) and the throttle
    # is capped at the chunk period so it cannot skip whole chunks.
    monkeypatch.setattr(settings, "live_caption_chunk_seconds", 0.5)
    monkeypatch.setattr(settings, "live_caption_min_interval_seconds", 5)
    config = await get_live_caption_config()
    assert config["chunk_ms"] == 500
    assert config["min_interval_ms"] == 500
