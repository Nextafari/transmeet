from pydantic import BaseModel, Field


class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1)
    target_language: str = Field(default="en", min_length=2, max_length=8)


class TranslationResponse(BaseModel):
    translated_text: str
    model: str
    target_language: str
