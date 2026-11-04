"""Auto-detect wiring: no forced language, detected language translated.

Uses stub Whisper models (no ML dependencies) to prove the repository's
own wiring: ``source=None`` ("auto") reaches faster-whisper without a
``language`` option, and the detected language is handed to translation.
"""

import pytest

from app.services.translator_backends import (
    WhisperGoogleTranslateBackend,
    WhisperLocalTranslateBackend,
)


class _Segment:
    def __init__(self, text):
        self.text = text


class _Info:
    def __init__(self, language):
        self.language = language


class StubWhisperModel:
    """Records transcribe kwargs; replays canned segments + detection."""

    def __init__(self, texts, language):
        self._texts = texts
        self._language = language
        self.calls = []

    def transcribe(self, audio_path, **options):
        self.calls.append({"audio_path": audio_path, "options": options})
        segments = [_Segment(text) for text in self._texts]
        return segments, _Info(self._language)


class FakeEngine:
    provider = "nllb"

    def __init__(self, translated="hello"):
        self._translated = translated
        self.calls = []

    def same_language(self, source, target):
        return False

    def is_same_language(self, source, target):
        return self.same_language(source, target)

    def map_language(self, lang):
        return f"mapped:{lang}"

    def translate(self, text, source, target):
        self.calls.append((text, source, target))
        return self._translated


def _local_backend(stub, engine):
    backend = WhisperLocalTranslateBackend.__new__(
        WhisperLocalTranslateBackend
    )
    backend.whisper_model = stub
    backend.translation_engine = engine
    return backend


@pytest.mark.asyncio()
async def test_local_auto_detects_and_translates_detected(tmp_path):
    stub = StubWhisperModel(["bonjour"], "fr")
    engine = FakeEngine(translated="hello")
    backend = _local_backend(stub, engine)

    transcribed, translated = await backend.transcribe_and_translate_audio(
        tmp_path / "clip.wav", target_language="en", source_language=None
    )

    assert transcribed == "bonjour"
    assert translated == "hello"
    # No forced language: faster-whisper detects it.
    assert "language" not in stub.calls[0]["options"]
    # The detected language drives translation.
    assert engine.calls == [("bonjour", "fr", "en")]


@pytest.mark.asyncio()
async def test_local_explicit_source_still_forced(tmp_path):
    stub = StubWhisperModel(["bonjour"], "fr")
    engine = FakeEngine(translated="hello")
    backend = _local_backend(stub, engine)

    await backend.transcribe_and_translate_audio(
        tmp_path / "clip.wav", target_language="en", source_language="fr"
    )

    assert stub.calls[0]["options"]["language"] == "fr"
    assert engine.calls == [("bonjour", "fr", "en")]


@pytest.mark.asyncio()
async def test_google_auto_detection_omits_language(tmp_path):
    stub = StubWhisperModel(["hola"], "es")
    backend = WhisperGoogleTranslateBackend.__new__(
        WhisperGoogleTranslateBackend
    )
    backend.whisper_model = stub

    transcribed = await backend.transcribe_audio(
        tmp_path / "clip.wav", source_language=None
    )

    assert transcribed == "hola"
    assert "language" not in stub.calls[0]["options"]
