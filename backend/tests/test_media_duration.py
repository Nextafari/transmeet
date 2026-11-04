"""ffprobe duration parsing without ffmpeg installed."""

from app.services.media import probe_duration_seconds


def test_format_duration_wins():
    probe = {
        "format": {"duration": "12.5"},
        "streams": [{"duration": "3.0"}],
    }
    assert probe_duration_seconds(probe) == 12.5


def test_stream_duration_fallback():
    probe = {
        "format": {},
        "streams": [{"duration": "7.25"}, {"duration": "6.0"}],
    }
    assert probe_duration_seconds(probe) == 7.25


def test_timestamp_duration_fallback():
    probe = {
        "format": {},
        "streams": [{"duration_ts": 48000, "time_base": "1/16000"}],
    }
    assert probe_duration_seconds(probe) == 3.0


def test_missing_or_invalid_durations_return_none():
    assert probe_duration_seconds({"format": {}, "streams": []}) is None
    assert probe_duration_seconds({"format": {"duration": "0"}}) is None
    assert (
        probe_duration_seconds({"format": {}, "streams": [{"duration": "nan"}]})
        is None
    )
