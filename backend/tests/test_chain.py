"""Live chain resolution: order, fallback, cooldown, skips."""

import pytest

from app.services.translator import TranslatorService


class FakeBackend:
    """Scripted stand-in: script entries are ("ok", text) or ("raise", exc)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def transcribe_and_translate_audio(
        self, audio_path, target_language="en", source_language=None
    ):
        self.calls.append(("full", target_language, source_language))
        action = self.script.pop(0)
        if action[0] == "raise":
            raise action[1]
        return action[1], action[2]

    async def transcribe_audio(self, audio_path, source_language=None):
        self.calls.append(("audio", source_language))
        return "  hello  "


def _service_with_chain(monkeypatch, chain, backends, meta_key="meta-key"):
    from app.core.config import settings

    monkeypatch.setattr(settings, "live_caption_backends", list(chain))
    monkeypatch.setattr(settings, "meta_model_api_key", meta_key)
    service = TranslatorService()
    monkeypatch.setattr(
        service, "_get_live_rung_backend", lambda rung: backends[rung]
    )
    return service


@pytest.mark.asyncio()
async def test_primary_rung_serves(monkeypatch, tmp_path):
    ok = FakeBackend([("ok", "hola", "hello")])
    service = _service_with_chain(monkeypatch, ["whisper-local", "gemini"], {"whisper-local": ok})
    text, rung = await service.transcribe_and_translate_live_chain(
        tmp_path / "a.wav", target_language="en"
    )
    assert (text, rung) == ("hello", "whisper-local")
    assert ok.calls == [("full", "en", None)]


@pytest.mark.asyncio()
async def test_failure_falls_through_to_next_rung(monkeypatch, tmp_path):
    bad = FakeBackend([("raise", RuntimeError("boom"))])
    good = FakeBackend([("ok", "hola", "hello")])
    service = _service_with_chain(
        monkeypatch, ["meta", "whisper-local"], {"meta": bad, "whisper-local": good}
    )
    text, rung = await service.transcribe_and_translate_live_chain(
        tmp_path / "a.wav", target_language="en", source_language="es"
    )
    assert (text, rung) == ("hello", "whisper-local")
    assert good.calls == [("full", "en", "es")]


@pytest.mark.asyncio()
async def test_quota_failure_cools_rung_down(monkeypatch, tmp_path):
    from app.services.meta.voice_client import MetaVoiceError

    calls = {"meta": 0}

    class FlakyMeta(FakeBackend):
        async def transcribe_and_translate_audio(
            self, audio_path, target_language="en", source_language=None
        ):
            calls["meta"] += 1
            raise MetaVoiceError("rate limited (429)", quota=True)

    good = FakeBackend([("ok", "hola", "hello"), ("ok", "hola", "hello")])
    service = _service_with_chain(
        monkeypatch, ["meta", "whisper-local"], {"meta": FlakyMeta([]), "whisper-local": good}
    )
    await service.transcribe_and_translate_live_chain(tmp_path / "a.wav", target_language="en")
    await service.transcribe_and_translate_live_chain(tmp_path / "a.wav", target_language="en")
    assert calls["meta"] == 1
    assert "meta" in service._rung_cooldown_until


@pytest.mark.asyncio()
async def test_meta_rung_skipped_without_key(monkeypatch, tmp_path):
    meta = FakeBackend([("ok", "hola", "hello")])
    local = FakeBackend([("ok", "bonjour", "hello")])
    service = _service_with_chain(
        monkeypatch,
        ["meta", "whisper-local"],
        {"meta": meta, "whisper-local": local},
        meta_key=None,
    )
    text, rung = await service.transcribe_and_translate_live_chain(
        tmp_path / "a.wav", target_language="en"
    )
    assert (text, rung) == ("hello", "whisper-local")
    assert meta.calls == []


@pytest.mark.asyncio()
async def test_unknown_rung_rejected(monkeypatch, tmp_path):
    service = _service_with_chain(monkeypatch, ["nope"], {})
    with pytest.raises(ValueError, match="Unknown live caption backend"):
        await service.transcribe_and_translate_live_chain(
            tmp_path / "a.wav", target_language="en"
        )


@pytest.mark.asyncio()
async def test_baseten_rung_serves(monkeypatch, tmp_path):
    from app.core.config import settings

    baseten = FakeBackend([("ok", "hola", "hello")])
    service = _service_with_chain(
        monkeypatch, ["baseten", "whisper-local"], {"baseten": baseten}
    )
    monkeypatch.setattr(settings, "baseten_api_key", "test-key")
    text, rung = await service.transcribe_and_translate_live_chain(
        tmp_path / "a.wav", target_language="en"
    )
    assert (text, rung) == ("hello", "baseten")


@pytest.mark.asyncio()
async def test_baseten_rung_skipped_without_key(monkeypatch, tmp_path):
    from app.core.config import settings

    baseten = FakeBackend([("ok", "hola", "hello")])
    local = FakeBackend([("ok", "bonjour", "hello")])
    service = _service_with_chain(
        monkeypatch,
        ["baseten", "whisper-local"],
        {"baseten": baseten, "whisper-local": local},
    )
    monkeypatch.setattr(settings, "baseten_api_key", None)
    text, rung = await service.transcribe_and_translate_live_chain(
        tmp_path / "a.wav", target_language="en"
    )
    assert (text, rung) == ("hello", "whisper-local")
    assert baseten.calls == []


@pytest.mark.asyncio()
async def test_translate_false_uses_transcribe_only(monkeypatch, tmp_path):
    backend = FakeBackend([])
    service = _service_with_chain(monkeypatch, ["gemini"], {"gemini": backend})
    text, rung = await service.transcribe_and_translate_live_chain(
        tmp_path / "a.wav", target_language="en", translate=False
    )
    assert (text, rung) == ("hello", "gemini")
    assert backend.calls == [("audio", None)]
