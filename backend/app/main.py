import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.services.translator import translator_service

import logging

# Ensure basic logging is configured so our exception logs appear in stdout
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


async def _warm_live_caption_backend() -> None:
    for rung in settings.effective_live_chain():
        try:
            await asyncio.to_thread(
                translator_service._get_live_rung_backend, rung
            )
            logger.info("Live caption rung warmed: %s", rung)
        except Exception as exc:
            logger.exception("Live caption rung warmup failed (%s): %s", rung, exc)


@app.on_event("startup")
async def warm_live_caption_backend() -> None:
    """Preload the live caption backend without blocking API startup."""
    asyncio.create_task(_warm_live_caption_backend())


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    return {"message": "Screen Recorder backend is running"}
