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
        try:
            probe = ffmpeg.probe(str(source_path))
        except ffmpeg.Error as exc:
            stderr = ""
            if isinstance(exc.stderr, (bytes, bytearray)):
                stderr = exc.stderr.decode("utf-8", errors="ignore")
            else:
                stderr = str(exc.stderr)
            raise RuntimeError(
                f"Unable to determine media duration: {stderr[:300].strip()}"
            ) from exc

        duration_raw = probe.get("format", {}).get("duration")
        if duration_raw is None:
            raise RuntimeError("Unable to determine media duration")

        total_duration = float(duration_raw)
        if total_duration <= 0:
            raise RuntimeError("Media duration is zero")

        effective_duration = min(total_duration, float(max_seconds))
        output_dir = Path(tempfile.mkdtemp(prefix="screenrec-audio-"))
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
                ffmpeg.input(str(source_path), ss=cursor, t=clip_duration)
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
            stderr = ""
            if isinstance(exc.stderr, (bytes, bytearray)):
                stderr = exc.stderr.decode("utf-8", errors="ignore")
            else:
                stderr = str(exc.stderr)
            raise RuntimeError(
                f"ffmpeg error: {stderr[:300].strip()}"
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
