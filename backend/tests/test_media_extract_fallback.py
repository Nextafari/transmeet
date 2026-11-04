"""Caption extraction must survive ffprobe failures with readable errors."""

import ffmpeg
import pytest

from app.services import media as media_module
from app.services.media import MediaService, _stderr_tail

_BANNER = (
    b"ffprobe version 7.1.5-0+deb13u1 Copyright (c) 2007-2026 the FFmpeg developers\n"
    b"  built with gcc 14 (Debian 14.2.0-19)\n"
    b"  configuration: --prefix=/usr\n"
    b"recording.webm: Invalid data found when processing input\n"
)


def test_stderr_tail_shows_reason_not_banner():
    exc = ffmpeg.Error("ffprobe", stdout=b"", stderr=_BANNER)
    tail = _stderr_tail(exc)
    assert "Invalid data found when processing input" in tail
    assert "ffprobe version" not in tail
    assert "Copyright" not in tail


class _FakeRun:
    def run(self, capture_stdout=True, capture_stderr=True):
        return b"", b""


class _FakeOutput:
    def overwrite_output(self):
        return _FakeRun()


class _FakeInput:
    def output(self, *args, **kwargs):
        return _FakeOutput()


def _patch_ffmpeg(monkeypatch, probe_impl):
    monkeypatch.setattr(media_module.ffmpeg, "probe", probe_impl)
    monkeypatch.setattr(
        media_module.ffmpeg, "input", lambda *a, **k: _FakeInput()
    )


def test_probe_failure_falls_back_to_wav(tmp_path, monkeypatch):
    source = tmp_path / "recording.webm"
    source.write_bytes(b"fake-webm-bytes")

    def probe(path):
        if str(path).endswith(".wav"):
            return {"format": {"duration": "12.5"}, "streams": []}
        raise ffmpeg.Error("ffprobe", stdout=b"", stderr=_BANNER)

    _patch_ffmpeg(monkeypatch, probe)
    chunks, _out_dir, effective = MediaService().extract_audio_chunks(
        source, chunk_seconds=45, max_seconds=900
    )
    assert effective == 12.5
    assert len(chunks) == 1
    assert chunks[0].duration_sec == 12.5


def test_probe_and_wav_failure_reports_reason(tmp_path, monkeypatch):
    source = tmp_path / "recording.webm"
    source.write_bytes(b"fake-webm-bytes")

    def probe(path):
        raise ffmpeg.Error("ffprobe", stdout=b"", stderr=_BANNER)

    _patch_ffmpeg(monkeypatch, probe)
    with pytest.raises(RuntimeError, match="Invalid data found"):
        MediaService().extract_audio_chunks(
            source, chunk_seconds=45, max_seconds=900
        )


def test_empty_recording_rejected(tmp_path, monkeypatch):
    source = tmp_path / "recording.webm"
    source.write_bytes(b"")
    with pytest.raises(RuntimeError, match="empty"):
        MediaService().extract_audio_chunks(
            source, chunk_seconds=45, max_seconds=900
        )
