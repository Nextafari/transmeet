"""parse_stream_params: transport only, never model selection."""

from app.services.meta.events import PROTOCOL_VERSION, WS_PATH, parse_stream_params


def test_defaults():
    params = parse_stream_params({})
    assert params.source_language is None
    assert params.target_language == "en"
    assert params.translate is True


def test_languages_and_flag():
    params = parse_stream_params(
        {"source_language": "ES", "target_language": " Fr ", "translate": "false"}
    )
    assert params.source_language == "es"
    assert params.target_language == "fr"
    assert params.translate is False


def test_auto_source_normalizes_to_none():
    for value in ("auto", "AUTO", "detect", "auto-detect", ""):
        assert parse_stream_params({"source_language": value}).source_language is None


def test_engine_backend_model_keys_are_ignored():
    params = parse_stream_params(
        {
            "engine": "gemini",
            "backend": "whisper-local",
            "model": "some-model",
            "provider": "spark",
            "target_language": "es",
        }
    )
    assert params.target_language == "es"
    assert params.translate is True
    assert "engine" not in params.model_dump()
    assert PROTOCOL_VERSION == "1"
    assert WS_PATH == "/api/v1/media/live-stream"
