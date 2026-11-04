"""Parity: the rewired whisper-local backend keeps its old behavior."""

import sys
import types

import pytest


def _install_faster_whisper_stub():
    torch_stub = types.ModuleType("torch")
    torch_cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.cuda = torch_cuda
    sys.modules["torch"] = torch_stub

    stub = types.ModuleType("faster_whisper")

    class FakeInfo:
        language = "es"

    class FakeSegment:
        def __init__(self, text, start, end):
            self.text = text
            self.start = start
            self.end = end

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.last_options = kwargs

        def transcribe(self, path, **options):
            self.last_options = options
            return ([FakeSegment("hola mundo", 0.0, 1.5)], FakeInfo())

    stub.WhisperModel = FakeModel
    sys.modules["faster_whisper"] = stub
    return FakeModel


class FakeEngine:
    provider = "nllb"
    model_name = "fake-nllb"

    MAP = {"es": "spa_Latn", "en": "eng_Latn"}

    def __init__(self):
        self.calls = []

    def map_language(self, lang):
        return self.MAP.get((lang or "").strip().lower())

    def same_language(self, source, target):
        source_mapped = self.map_language(source)
        target_mapped = self.map_language(target)
        return bool(source_mapped and target_mapped and source_mapped == target_mapped)

    def is_same_language(self, source, target):
        return self.same_language(source, target)

    def translate(self, text, source, target):
        self.calls.append((text, source, target))
        return f"{text} [{target}]"


def _make_backend():
    _install_faster_whisper_stub()
    from app.services.translator_backends import WhisperLocalTranslateBackend

    engine = FakeEngine()
    backend = WhisperLocalTranslateBackend(
        whisper_model_size="tiny",
        provider="nllb",
        model_name="fake-nllb",
        translation_engine=engine,
    )
    return backend, engine


def test_backend_mirrors_engine_identity():
    backend, _engine = _make_backend()
    assert backend.provider == "nllb"
    assert backend.model_name == "fake-nllb"


@pytest.mark.asyncio()
async def test_segments_translate_each_segment(tmp_path):
    backend, engine = _make_backend()
    segments = await backend.transcribe_and_translate_audio_segments(
        tmp_path / "a.wav", target_language="en", source_language="es"
    )
    assert len(segments) == 1
    segment = segments[0]
    assert segment["source_text"] == "hola mundo"
    assert segment["translated_text"] == "hola mundo [en]"
    assert segment["start_sec"] == 0.0
    assert segment["end_sec"] == 1.5
    assert engine.calls == [("hola mundo", "es", "en")]


@pytest.mark.asyncio()
async def test_segments_same_language_skips_translation(tmp_path):
    backend, engine = _make_backend()
    segments = await backend.transcribe_and_translate_audio_segments(
        tmp_path / "a.wav", target_language="en", source_language="en"
    )
    assert segments[0]["translated_text"] == "hola mundo"
    assert engine.calls == []


@pytest.mark.asyncio()
async def test_segments_unsupported_source_skips_translation(tmp_path):
    backend, engine = _make_backend()
    segments = await backend.transcribe_and_translate_audio_segments(
        tmp_path / "a.wav", target_language="en", source_language="xx"
    )
    assert segments[0]["translated_text"] == "hola mundo"
    assert engine.calls == []


@pytest.mark.asyncio()
async def test_segments_translate_false_skips_translation(tmp_path):
    backend, engine = _make_backend()
    segments = await backend.transcribe_and_translate_audio_segments(
        tmp_path / "a.wav", target_language="en", source_language="es",
        translate=False,
    )
    assert segments[0]["translated_text"] == "hola mundo"
    assert engine.calls == []


@pytest.mark.asyncio()
async def test_full_audio_translates(tmp_path):
    backend, engine = _make_backend()
    transcribed, translated = await backend.transcribe_and_translate_audio(
        tmp_path / "a.wav", target_language="en", source_language="es"
    )
    assert transcribed == "hola mundo"
    assert translated == "hola mundo [en]"
    assert engine.calls == [("hola mundo", "es", "en")]
