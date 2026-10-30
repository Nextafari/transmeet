from pydantic import BaseModel, Field


class CaptionSegment(BaseModel):
    start_sec: float = Field(..., ge=0)
    end_sec: float = Field(..., gt=0)
    source_text: str = Field(..., min_length=1)
    translated_text: str = Field(..., min_length=1)


class MediaCaptionResponse(BaseModel):
    model: str
    source_language: str
    target_language: str
    subtitle_format: str
    subtitle_content: str
    transcript_text: str
    translated_text: str
    segments: list[CaptionSegment]


class LiveCaptionResponse(BaseModel):
    model: str
    source_language: str
    target_language: str
    caption_text: str
