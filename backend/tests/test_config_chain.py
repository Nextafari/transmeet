"""Chain and list parsing in Settings."""

from tests.conftest import make_settings


def test_comma_separated_chain_parses():
    settings = make_settings(LIVE_CAPTION_BACKENDS="meta,whisper-local,gemini")
    assert settings.live_caption_backends == ["meta", "whisper-local", "gemini"]
    assert settings.effective_live_chain() == ["meta", "whisper-local", "gemini"]


def test_json_list_chain_parses():
    settings = make_settings(LIVE_CAPTION_BACKENDS='["meta", "gemini"]')
    assert settings.effective_live_chain() == ["meta", "gemini"]


def test_empty_chain_falls_back_to_single_backend():
    settings = make_settings(
        LIVE_CAPTION_BACKENDS="", LIVE_CAPTION_BACKEND="whisper-local"
    )
    assert settings.effective_live_chain() == ["whisper-local"]


def test_translation_fallbacks_parse():
    settings = make_settings(STREAM_TRANSLATION_FALLBACKS="spark,nllb")
    assert settings.stream_translation_fallbacks == ["spark", "nllb"]


def test_meta_and_stream_defaults():
    settings = make_settings()
    assert settings.meta_asr_model == "muse-voice-transcribe-1.0"
    assert settings.meta_asr_mode == "ENDPOINTING"
    assert settings.meta_translation_engine == "nllb"
    assert settings.live_stream_enabled is True
    assert settings.live_stream_partial_debounce_ms == 600
