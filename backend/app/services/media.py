import shutil
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import ffmpeg
import srt


@dataclass
class AudioChunk:
    path: Path
    start_sec: float
    duration_sec: float


def _stderr_text(exc: BaseException) -> str:
    raw = getattr(exc, "stderr", None)
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="ignore")
    return str(raw or "")


def _stderr_tail(exc: BaseException, limit: int = 300) -> str:
    """Last meaningful ffmpeg/ffprobe lines (the banner comes first)."""
    lines = [
        line.strip()
        for line in _stderr_text(exc).splitlines()
        if line.strip()
    ]
    detail = " | ".join(lines[-3:]) if lines else ""
    return detail[:limit] or "no ffmpeg output"


def probe_duration_seconds(probe: dict[str, Any]) -> float | None:
    """Best-effort duration from an ffprobe dict, else None.

    Chrome MediaRecorder webm files often lack ``format.duration``, while
    individual streams may still carry ``duration`` or timestamp-derived
    durations.
    """
    try:
        raw = (probe.get("format") or {}).get("duration")
        if raw is not None and float(raw) > 0:
            return float(raw)
    except (TypeError, ValueError):
        pass

    best = 0.0
    streams = probe.get("streams") or []
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        try:
            value = float(stream.get("duration"))
        except (TypeError, ValueError):
            pass
        else:
            best = max(best, value)
        try:
            duration_ts = stream.get("duration_ts")
            time_base = str(stream.get("time_base") or "")
            numerator, denominator = time_base.split("/")
            value = float(duration_ts) * float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        else:
            best = max(best, value)

    return best if best > 0 else None


class MediaService:
    def convert_video(
        self,
        source_path: Path,
        output_ext: str = "mp4",
    ) -> Path:
        output_dir = Path(tempfile.mkdtemp(prefix="screenrec-convert-"))
        output_path = output_dir / f"converted.{output_ext}"

        # Basic transcode settings; tune per use case later.
        (
            ffmpeg.input(str(source_path))
            .output(
                str(output_path),
                vcodec="libx264",
                acodec="aac",
                movflags="+faststart",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        return output_path

    def extract_audio_chunks(
        self,
        source_path: Path,
        *,
        chunk_seconds: int,
        max_seconds: int,
    ) -> tuple[list[AudioChunk], Path, float]:
        if source_path.stat().st_size == 0:
            raise RuntimeError("Recording file is empty (0 bytes)")

        # ffprobe may fail outright on header-less or slightly damaged
        # MediaRecorder output even though ffmpeg can still decode it, so
        # a probe failure falls through to the WAV-extraction fallback
        # below instead of failing captions immediately.
        probe_error: ffmpeg.Error | None = None
        probe: dict[str, Any] = {}
        try:
            probe = ffmpeg.probe(str(source_path))
        except ffmpeg.Error as exc:
            probe_error = exc

        output_dir = Path(tempfile.mkdtemp(prefix="screenrec-audio-"))
        chunk_source = source_path
        total_duration = probe_duration_seconds(probe)
        if total_duration is None:
            # Chrome MediaRecorder webm often carries no duration metadata
            # at all: extract the full audio track once (wav always probes
            # cleanly) and chunk from that instead.
            full_audio_path = output_dir / "full.wav"
            (
                ffmpeg.input(str(source_path))
                .output(
                    str(full_audio_path),
                    format="wav",
                    ac=1,
                    ar=16000,
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            try:
                fallback_probe = ffmpeg.probe(str(full_audio_path))
            except ffmpeg.Error as exc:
                if probe_error is not None:
                    raise RuntimeError(
                        "Unable to read recording: "
                        f"probe failed ({_stderr_tail(probe_error)}); "
                        f"wav probe failed ({_stderr_tail(exc)})"
                    ) from exc
                raise RuntimeError(
                    f"Unable to read extracted audio: {_stderr_tail(exc)}"
                ) from exc
            total_duration = probe_duration_seconds(fallback_probe)
            chunk_source = full_audio_path

        if total_duration is None or total_duration <= 0:
            if probe_error is not None:
                raise RuntimeError(
                    "Unable to determine media duration: "
                    f"{_stderr_tail(probe_error)}"
                )
            raise RuntimeError("Unable to determine media duration")

        effective_duration = min(total_duration, float(max_seconds))
        chunks: list[AudioChunk] = []

        cursor = 0.0
        index = 0
        step = max(5, int(chunk_seconds))
        while cursor < effective_duration:
            clip_duration = min(float(step), effective_duration - cursor)
            if clip_duration <= 0:
                break

            chunk_path = output_dir / f"chunk-{index:04d}.wav"
            (
                ffmpeg.input(str(chunk_source), ss=cursor, t=clip_duration)
                .output(
                    str(chunk_path),
                    format="wav",
                    ac=1,
                    ar=16000,
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            chunks.append(
                AudioChunk(
                    path=chunk_path,
                    start_sec=cursor,
                    duration_sec=clip_duration,
                )
            )
            cursor += clip_duration
            index += 1

        return chunks, output_dir, effective_duration

    def extract_audio_clip(
        self,
        source_path: Path,
    ) -> tuple[Path, Path]:
        output_dir = Path(tempfile.mkdtemp(prefix="screenrec-audio-"))
        output_path = output_dir / "clip.wav"

        try:
            (
                ffmpeg.input(str(source_path))
                .output(
                    str(output_path),
                    format="wav",
                    ac=1,
                    ar=16000,
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as exc:
            raise RuntimeError(
                f"ffmpeg error: {_stderr_tail(exc)}"
            ) from exc

        return output_path, output_dir

    def build_subtitle_content(
        self,
        segments: list[Any],
        *,
        subtitle_format: str = "srt",
    ) -> str:
        subtitle_format = subtitle_format.lower()
        sorted_segments = sorted(
            segments,
            key=lambda segment: float(getattr(segment, "start_sec", 0.0)),
        )

        if subtitle_format == "srt":
            subtitles: list[srt.Subtitle] = []
            for idx, segment in enumerate(sorted_segments, start=1):
                start_sec = max(0.0, float(getattr(segment, "start_sec", 0.0)))
                end_sec = max(
                    start_sec + 0.2,
                    float(getattr(segment, "end_sec", start_sec + 2.0)),
                )
                content = str(getattr(segment, "translated_text", "")).strip()
                if not content:
                    continue

                subtitles.append(
                    srt.Subtitle(
                        index=idx,
                        start=timedelta(seconds=start_sec),
                        end=timedelta(seconds=end_sec),
                        content=content,
                    )
                )
            return srt.compose(subtitles)

        if subtitle_format == "vtt":
            lines = ["WEBVTT", ""]
            for segment in sorted_segments:
                start_sec = max(0.0, float(getattr(segment, "start_sec", 0.0)))
                end_sec = max(
                    start_sec + 0.2,
                    float(getattr(segment, "end_sec", start_sec + 2.0)),
                )
                content = str(getattr(segment, "translated_text", "")).strip()
                if not content:
                    continue

                start_ts = self._format_vtt_timestamp(start_sec)
                end_ts = self._format_vtt_timestamp(end_sec)
                lines.append(
                    f"{start_ts} --> {end_ts}"
                )
                lines.append(content)
                lines.append("")
            return "\n".join(lines).strip() + "\n"

        raise ValueError("Unsupported subtitle format. Use 'srt' or 'vtt'.")

    def write_subtitle_file(
        self,
        subtitle_content: str,
        *,
        subtitle_format: str,
    ) -> Path:
        subtitle_format = subtitle_format.lower()
        output_dir = Path(tempfile.mkdtemp(prefix="screenrec-subtitles-"))
        subtitle_path = output_dir / f"captions.{subtitle_format}"
        subtitle_path.write_text(subtitle_content, encoding="utf-8")
        return subtitle_path

    def burn_subtitles(
        self,
        source_path: Path,
        subtitle_path: Path,
        *,
        output_ext: str = "mp4",
    ) -> Path:
        output_dir = Path(tempfile.mkdtemp(prefix="screenrec-burned-"))
        output_path = output_dir / f"captioned.{output_ext}"

        escaped_path = self._escape_filter_path(subtitle_path)
        try:
            (
                ffmpeg.input(str(source_path))
                .output(
                    str(output_path),
                    vf=f"subtitles='{escaped_path}'",
                    vcodec="libx264",
                    acodec="aac",
                    movflags="+faststart",
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as exc:
            raise RuntimeError(
                f"Subtitle burn failed: {_stderr_tail(exc)}"
            ) from exc

        return output_path

    def _format_vtt_timestamp(self, seconds: float) -> str:
        total_ms = max(0, int(round(seconds * 1000)))
        hours = total_ms // 3600000
        minutes = (total_ms % 3600000) // 60000
        secs = (total_ms % 60000) // 1000
        millis = total_ms % 1000
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

    def _escape_filter_path(self, path: Path) -> str:
        return (
            str(path)
            .replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
        )

    def cleanup_temp_file(self, path: Path) -> None:
        try:
            shutil.rmtree(path.parent, ignore_errors=True)
        except Exception:
            pass

    def cleanup_temp_dir(self, path: Path) -> None:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass


media_service = MediaService()
