import tempfile
import time
from hashlib import sha1
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

import logging

from app.core.config import settings
from app.schemas.media import (
    CaptionSegment,
    LiveCaptionResponse,
    MediaCaptionResponse,
)
from app.services.media import media_service
from app.services.translator import translator_service

router = APIRouter()

logger = logging.getLogger(__name__)

_live_caption_last_hash: str | None = None
_live_caption_last_text: str = ""
_live_caption_last_model: str = settings.live_caption_backend
_live_caption_last_ts: float = 0.0
_live_caption_last_translate: bool = True
_live_caption_last_source_language: str = "auto"
_live_caption_last_target_language: str = "en"


@router.post("/convert")
async def convert_video(
    file: UploadFile = File(...),
    output_ext: str = "mp4",
) -> FileResponse:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename",
        )

    temp_input_path = Path(f"/tmp/{file.filename}")

    try:
        content = await file.read()
        temp_input_path.write_bytes(content)

        output_path = media_service.convert_video(
            temp_input_path,
            output_ext=output_ext,
        )

        return FileResponse(
            path=output_path,
            media_type="application/octet-stream",
            filename=f"{Path(file.filename).stem}.{output_ext}",
            background=None,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Conversion failed: {exc}",
        ) from exc
    finally:
        if temp_input_path.exists():
            temp_input_path.unlink(missing_ok=True)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_source_language(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    if normalized in {"", "auto", "detect", "auto-detect", "autodetect"}:
        return None
    return normalized


@router.get("/live-caption/config")
async def get_live_caption_config() -> dict[str, Any]:
    streaming_available = bool(
        settings.live_stream_enabled
        and (settings.meta_model_api_key or "").strip()
    )
    return {
        "backend": settings.live_caption_backend,
        "backends": settings.effective_live_chain(),
        "chunk_ms": max(1000, int(settings.live_caption_chunk_seconds) * 1000),
        "min_interval_ms": max(
            0,
            int(settings.live_caption_min_interval_seconds) * 1000,
        ),
        "min_audio_bytes": max(0, int(settings.live_caption_min_audio_bytes)),
        "cache_ttl_ms": max(
            0,
            int(settings.live_caption_cache_ttl_seconds) * 1000,
        ),
        "streaming": {
            "enabled": streaming_available,
            "ws_path": "/api/v1/media/live-stream",
            "protocol": "1",
            "encoding": "PCM_16KHZ",
            "sample_rate_hz": 16000,
            "frame_ms": max(20, int(settings.live_stream_frame_ms)),
            "partial_debounce_ms": max(
                0, int(settings.live_stream_partial_debounce_ms)
            ),
        },
    }


@router.post("/captions", response_model=MediaCaptionResponse)
async def create_translated_captions(
    file: UploadFile = File(...),
    source_language: str = Form("auto"),
    target_language: str = Form("en"),
    subtitle_format: str = Form("srt"),
    translate: str | bool = Form("true"),
) -> MediaCaptionResponse:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename",
        )

    subtitle_format = subtitle_format.lower().strip()
    translate = _parse_bool(translate)
    source_language_value = _normalize_source_language(source_language)
    source_language_label = source_language_value or "auto"
    target_language = target_language.strip().lower() or "en"
    if subtitle_format not in {"srt", "vtt"}:
        raise HTTPException(
            status_code=400,
            detail="subtitle_format must be 'srt' or 'vtt'",
        )

    input_dir = Path(tempfile.mkdtemp(prefix="screenrec-input-"))
    temp_input_path = input_dir / file.filename
    audio_dir: Path | None = None

    try:
        content = await file.read()
        temp_input_path.write_bytes(content)

        audio_chunks, audio_dir, _ = media_service.extract_audio_chunks(
            temp_input_path,
            chunk_seconds=settings.transcription_chunk_seconds,
            max_seconds=settings.max_transcription_seconds,
        )

        all_segments = []
        model_name = settings.translation_model
        for chunk in audio_chunks:
            try:
                (
                    chunk_segments,
                    model_name,
                ) = await translator_service.transcribe_and_translate_audio_chunk(
                    chunk.path,
                    source_language=source_language_value,
                    target_language=target_language,
                    chunk_start_sec=chunk.start_sec,
                    chunk_duration_sec=chunk.duration_sec,
                    translate=translate,
                )
                all_segments.extend(chunk_segments)
            except Exception as exc:
                logger.exception(
                    "Caption generation failed for chunk %s (%s): %s",
                    getattr(chunk, "path", str(chunk)),
                    getattr(chunk, "start_sec", "?"),
                    exc,
                )
                raise RuntimeError(
                    "Caption generation failed for audio around "
                    f"{float(getattr(chunk, 'start_sec', 0.0)):.1f}s: {exc}"
                ) from exc

        if not all_segments:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No spoken segments were detected. "
                    "Try a recording with clearer speech or longer duration."
                ),
            )

        subtitle_content = media_service.build_subtitle_content(
            all_segments,
            subtitle_format=subtitle_format,
        )

        transcript_text = " ".join(
            segment.source_text for segment in all_segments
        )
        translated_text = " ".join(
            segment.translated_text for segment in all_segments
        )

        return MediaCaptionResponse(
            model=model_name,
            source_language=source_language_label,
            target_language=target_language,
            subtitle_format=subtitle_format,
            subtitle_content=subtitle_content,
            transcript_text=transcript_text,
            translated_text=translated_text,
            segments=[
                CaptionSegment(
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                    source_text=segment.source_text,
                    translated_text=segment.translated_text,
                )
                for segment in all_segments
            ],
        )
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Caption generation failed: {exc}",
        ) from exc
    finally:
        if temp_input_path.exists():
            temp_input_path.unlink(missing_ok=True)
        media_service.cleanup_temp_dir(input_dir)
        if audio_dir:
            media_service.cleanup_temp_dir(audio_dir)


@router.post("/live-caption", response_model=LiveCaptionResponse)
async def create_live_caption(
    file: UploadFile = File(...),
    source_language: str = Form("auto"),
    target_language: str = Form("en"),
    translate: str | bool = Form("true"),
) -> LiveCaptionResponse:
    global _live_caption_last_hash
    global _live_caption_last_text
    global _live_caption_last_model
    global _live_caption_last_ts
    global _live_caption_last_translate
    global _live_caption_last_source_language
    global _live_caption_last_target_language

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename",
        )

    translate = _parse_bool(translate)
    source_language_value = _normalize_source_language(source_language)
    source_language_label = source_language_value or "auto"
    target_language = target_language.strip().lower() or "en"

    input_dir = Path(tempfile.mkdtemp(prefix="screenrec-live-input-"))
    temp_input_path = input_dir / file.filename
    audio_dir: Path | None = None

    try:
        request_started = time.monotonic()
        now = time.monotonic()
        content = await file.read()
        read_elapsed = time.monotonic() - request_started

        if len(content) < settings.live_caption_min_audio_bytes:
            logger.info(
                "Live caption skipped: audio blob too small (%s bytes, %.3fs read)",
                len(content),
                read_elapsed,
            )
            return LiveCaptionResponse(
                model=_live_caption_last_model,
                source_language=source_language_label,
                target_language=target_language,
                caption_text="",
            )

        content_hash = sha1(content).hexdigest()
        if (
            _live_caption_last_hash == content_hash
            and _live_caption_last_ts > 0
            and now - _live_caption_last_ts
            < float(settings.live_caption_cache_ttl_seconds)
            and _live_caption_last_translate == translate
            and _live_caption_last_source_language == source_language_label
            and _live_caption_last_target_language == target_language
        ):
            logger.info(
                (
                    "Live caption cache hit: %s bytes, translate=%s, "
                    "source=%s, target=%s"
                ),
                len(content),
                translate,
                source_language_label,
                target_language,
            )
            return LiveCaptionResponse(
                model=_live_caption_last_model,
                source_language=source_language_label,
                target_language=target_language,
                caption_text=_live_caption_last_text,
            )

        temp_input_path.write_bytes(content)

        extract_started = time.monotonic()
        audio_clip_path, audio_dir = media_service.extract_audio_clip(temp_input_path)
        extract_elapsed = time.monotonic() - extract_started

        translate_started = time.monotonic()
        caption_text, model_name = (
            await translator_service.transcribe_and_translate_live_chain(
                audio_clip_path,
                source_language=source_language_value,
                target_language=target_language,
                translate=translate,
            )
        )
        translate_elapsed = time.monotonic() - translate_started
        total_elapsed = time.monotonic() - request_started

        _live_caption_last_hash = content_hash
        _live_caption_last_text = caption_text
        _live_caption_last_model = model_name
        _live_caption_last_ts = time.monotonic()
        _live_caption_last_translate = translate
        _live_caption_last_source_language = source_language_label
        _live_caption_last_target_language = target_language
        logger.info(
            (
                "Live caption processed: bytes=%s read=%.3fs extract=%.3fs "
                "backend=%.3fs total=%.3fs text_len=%s translate=%s "
                "source=%s target=%s"
            ),
            len(content),
            read_elapsed,
            extract_elapsed,
            translate_elapsed,
            total_elapsed,
            len(caption_text),
            translate,
            source_language_label,
            target_language,
        )
        return LiveCaptionResponse(
            model=model_name,
            source_language=source_language_label,
            target_language=target_language,
            caption_text=caption_text,
        )
    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.exception("Live caption processing failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Live caption processing failed: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Live caption failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Live caption failed: {exc}",
        ) from exc
    finally:
        if temp_input_path.exists():
            temp_input_path.unlink(missing_ok=True)
        media_service.cleanup_temp_dir(input_dir)
        if audio_dir:
            media_service.cleanup_temp_dir(audio_dir)


@router.post("/burn-subtitles")
async def burn_translated_subtitles(
    file: UploadFile = File(...),
    subtitle_content: str = Form(...),
    subtitle_format: str = Form("srt"),
    output_ext: str = Form("mp4"),
    keep_original_audio: str | bool = Form(True),
) -> FileResponse:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename",
        )

    subtitle_format = subtitle_format.lower().strip()
    if subtitle_format not in {"srt", "vtt"}:
        raise HTTPException(
            status_code=400,
            detail="subtitle_format must be 'srt' or 'vtt'",
        )

    output_ext = output_ext.lower().strip()
    if output_ext not in {"mp4", "webm"}:
        raise HTTPException(
            status_code=400,
            detail="output_ext must be 'mp4' or 'webm'",
        )

    # keep_original_audio is accepted for future behavior tuning.
    _parse_bool(keep_original_audio)

    input_dir = Path(tempfile.mkdtemp(prefix="screenrec-burn-input-"))
    temp_input_path = input_dir / file.filename
    subtitle_path: Path | None = None

    try:
        content = await file.read()
        temp_input_path.write_bytes(content)
        subtitle_path = media_service.write_subtitle_file(
            subtitle_content,
            subtitle_format=subtitle_format,
        )
        output_path = media_service.burn_subtitles(
            temp_input_path,
            subtitle_path,
            output_ext=output_ext,
        )
        response_name = f"{Path(file.filename).stem}-captioned.{output_ext}"
        return FileResponse(
            path=output_path,
            media_type="video/mp4" if output_ext == "mp4" else "video/webm",
            filename=response_name,
            background=BackgroundTask(
                media_service.cleanup_temp_dir,
                output_path.parent,
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Burn subtitles failed: {exc}",
        ) from exc
    finally:
        if temp_input_path.exists():
            temp_input_path.unlink(missing_ok=True)
        media_service.cleanup_temp_dir(input_dir)
        if subtitle_path:
            media_service.cleanup_temp_dir(subtitle_path.parent)
