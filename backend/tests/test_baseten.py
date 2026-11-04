"""Baseten client, file backend, and chain registration."""

import httpx
import pytest

from app.services.baseten.client import (
    BASETEN_BASE_URL,
    BasetenClient,
    BasetenError,
)
from app.services.baseten.file_backend import BasetenFileBackend


def _chat_payload(content):
    return {"choices": [{"message": {"content": content}}]}


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeAsyncClient:
    """Captures requests; replays one canned response."""

    last_instance = None

    def __init__(self, response, *args, **kwargs):
        self._response = response
        self.requests = []
        FakeAsyncClient.last_instance = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return self._response


def _patch_http(monkeypatch, response):
    def factory(*args, **kwargs):
        return FakeAsyncClient(response, *args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


def test_client_requires_key():
    with pytest.raises(BasetenError, match="BASETEN_API_KEY"):
        BasetenClient(None)
    with pytest.raises(BasetenError, match="BASETEN_API_KEY"):
        BasetenClient("  ")


@pytest.mark.asyncio()
async def test_translate_text_posts_chat_completions(monkeypatch):
    _patch_http(
        monkeypatch, FakeResponse(200, _chat_payload('{"translated_text": "bonjour"}'))
    )
    client = BasetenClient("test-key", chat_model="some/slug")
    assert await client.translate_text("hello", "fr", source_language="en") == "bonjour"

    request = FakeAsyncClient.last_instance.requests[0]
    assert request["url"] == f"{BASETEN_BASE_URL}/chat/completions"
    assert request["headers"] == {"Authorization": "Bearer test-key"}
    assert request["json"]["model"] == "some/slug"
    user_content = request["json"]["messages"][1]["content"]
    assert "Target language: fr" in user_content
    assert "Text: hello" in user_content


@pytest.mark.asyncio()
async def test_translate_text_extracts_fenced_json(monkeypatch):
    _patch_http(
        monkeypatch,
        FakeResponse(200, _chat_payload('```json\n{"translated_text": "hola"}\n```')),
    )
    client = BasetenClient("test-key")
    assert await client.translate_text("hello", "es") == "hola"


@pytest.mark.asyncio()
async def test_translate_empty_text_short_circuits(monkeypatch):
    FakeAsyncClient.last_instance = None
    _patch_http(monkeypatch, FakeResponse(200, _chat_payload("{}")))
    client = BasetenClient("test-key")
    assert await client.translate_text("  ", "es") == ""
    assert FakeAsyncClient.last_instance is None


@pytest.mark.asyncio()
async def test_translate_rate_limit_marks_quota(monkeypatch):
    _patch_http(monkeypatch, FakeResponse(429, None, text="slow down"))
    client = BasetenClient("test-key")
    with pytest.raises(BasetenError) as excinfo:
        await client.translate_text("hello", "es")
    assert excinfo.value.quota is True


@pytest.mark.asyncio()
async def test_translate_auth_failure_names_key(monkeypatch):
    _patch_http(monkeypatch, FakeResponse(401, None, text="bad key"))
    client = BasetenClient("test-key")
    with pytest.raises(BasetenError, match="BASETEN_API_KEY"):
        await client.translate_text("hello", "es")


@pytest.mark.asyncio()
async def test_translate_garbage_content_raises(monkeypatch):
    _patch_http(monkeypatch, FakeResponse(200, _chat_payload("not json at all")))
    client = BasetenClient("test-key")
    with pytest.raises(BasetenError, match="translated_text"):
        await client.translate_text("hello", "es")


@pytest.mark.asyncio()
async def test_transcribe_audio_sends_audio_url(monkeypatch, tmp_path):
    _patch_http(
        monkeypatch, FakeResponse(200, _chat_payload('{"transcript": "hello world"}'))
    )
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 100)
    client = BasetenClient("test-key", asr_model="audio/slug")
    assert await client.transcribe_audio(audio, "en") == "hello world"

    request = FakeAsyncClient.last_instance.requests[0]
    assert request["json"]["model"] == "audio/slug"
    parts = request["json"]["messages"][0]["content"]
    audio_part = next(p for p in parts if p.get("type") == "audio_url")
    assert audio_part["audio_url"]["url"].startswith("data:audio/wav;base64,")
    assert "Transcribe" in parts[0]["text"]


@pytest.mark.asyncio()
async def test_transcribe_audio_rejects_oversize(monkeypatch, tmp_path):
    FakeAsyncClient.last_instance = None
    _patch_http(monkeypatch, FakeResponse(200, _chat_payload("{}")))
    audio = tmp_path / "big.wav"
    audio.write_bytes(b"\x00" * (25 * 1024 * 1024 + 1))
    client = BasetenClient("test-key")
    with pytest.raises(BasetenError, match="25 MB"):
        await client.transcribe_audio(audio)
    assert FakeAsyncClient.last_instance is None


class FakeBasetenClient:
    def __init__(self, transcript="hola", translated="hello"):
        self._transcript = transcript
        self._translated = translated
        self.transcribe_calls = []
        self.translate_calls = []

    async def transcribe_audio(self, audio_path, source_language=None):
        self.transcribe_calls.append((audio_path, source_language))
        return self._transcript

    async def translate_text(
        self, text, target, source_language=None, timeout=None
    ):
        self.translate_calls.append((text, target, source_language))
        return self._translated


@pytest.mark.asyncio()
async def test_file_backend_transcribes_and_translates(monkeypatch, tmp_path):
    fake = FakeBasetenClient()
    monkeypatch.setattr(BasetenFileBackend, "_client", lambda self: fake)
    backend = BasetenFileBackend(
        api_key="test-key",
        translation_primary="baseten",
        baseten_client=fake,
    )
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    transcribed, translated = await backend.transcribe_and_translate_audio(
        audio, target_language="en", source_language="es"
    )
    assert (transcribed, translated) == ("hola", "hello")
    assert fake.transcribe_calls == [(audio, "es")]
    assert fake.translate_calls == [("hola", "en", "es")]


@pytest.mark.asyncio()
async def test_file_backend_segments_shape(monkeypatch, tmp_path):
    fake = FakeBasetenClient(transcript="one two three", translated="un deux trois")
    monkeypatch.setattr(BasetenFileBackend, "_client", lambda self: fake)
    backend = BasetenFileBackend(
        api_key="test-key",
        translation_primary="baseten",
        baseten_client=fake,
    )
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    segments = await backend.transcribe_and_translate_audio_segments(
        audio, target_language="fr", source_language="en"
    )
    assert len(segments) == 1
    assert segments[0]["source_text"] == "one two three"
    assert segments[0]["translated_text"] == "un deux trois"
    assert segments[0]["start_sec"] == 0.0
    assert segments[0]["end_sec"] > 0.0
