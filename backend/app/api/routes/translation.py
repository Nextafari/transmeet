from fastapi import APIRouter, HTTPException

from app.schemas.translation import TranslationRequest, TranslationResponse
from app.services.translator import translator_service

router = APIRouter()


@router.post("/text", response_model=TranslationResponse)
async def translate_text(payload: TranslationRequest) -> TranslationResponse:
    try:
        translated_text, model_name = await translator_service.translate_text(
            text=payload.text,
            target_language=payload.target_language,
        )
        return TranslationResponse(
            translated_text=translated_text,
            model=model_name,
            target_language=payload.target_language,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
