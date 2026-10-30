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
