from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.live_stream import router as live_stream_router
from app.api.routes.media import router as media_router
from app.api.routes.translation import router as translation_router

api_router = APIRouter()
api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(
    translation_router,
    prefix="/translation",
    tags=["translation"],
)
api_router.include_router(media_router, prefix="/media", tags=["media"])
api_router.include_router(live_stream_router, prefix="/media", tags=["media"])
