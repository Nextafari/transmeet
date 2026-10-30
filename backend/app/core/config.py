import json

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(...)
    app_version: str = Field(...)
    api_v1_prefix: str = Field(...)
    cors_origins: list[str] = Field(...)

    # AI provider settings
    google_api_key: str | None = None
    
    # Backend selection: 'gemini', 'whisper-google', or 'whisper-local'
    live_caption_backend: str = Field(default="whisper-google", description="Backend for live captions")
    captions_backend: str = Field(default="gemini", description="Backend for post-recording captions")

    # Ordered live caption backend chain: first entry is the primary, the
    # rest are fallbacks tried in order. Entries: 'meta',
    # 'whisper-google', 'whisper-local', 'gemini'. When empty, the single
    # live_caption_backend above is used. Model/engine selection is
    # server-side only; clients never choose backends.
    live_caption_backends: list[str] = Field(
        default=["meta", "whisper-local"],
        description="Ordered live caption backends; first entry is primary",
    )
    # Translation legs for Meta-powered backends: 'nllb' (local, free) or
    # 'spark' (Meta Model API, billed). First is primary, rest are fallbacks.
    meta_translation_engine: str = Field(
        default="nllb",
        description="Primary translation leg for Meta backends",
    )
    stream_translation_fallbacks: list[str] = Field(
        default=["spark"],
        description="Fallback translation legs for Meta backends",
    )

    # Meta Model API settings (https://dev.meta.ai). META_MODEL_API_KEY is
    # required for the 'meta' live backend and the 'spark' translation leg.
    meta_model_api_key: str | None = Field(
        default=None,
        description="Meta Model API key",
    )
    meta_asr_model: str = Field(
        default="muse-voice-transcribe-1.0",
        description="Meta speech-to-text model id",
    )
    meta_asr_mode: str = Field(
        default="ENDPOINTING",
        description="Meta realtime turn mode: PUSH_TO_TALK, ENDPOINTING, DIARIZATION",
    )
    meta_spark_model: str = Field(
        default="muse-spark-1.3",
        description="Meta text model id for the spark translation leg",
    )

    # Realtime live-stream transport (frontend PCM -> backend WS -> Meta).
    live_stream_enabled: bool = Field(
        default=True,
        description="Advertise and accept the live-stream WebSocket",
    )
    live_stream_partial_debounce_ms: int = Field(
        default=600,
        description="Debounce for translating streaming partial transcripts",
    )
    live_stream_frame_ms: int = Field(
        default=80,
        description="PCM frame duration the frontend streams per message",
    )
    live_stream_op_timeout_seconds: int = Field(
        default=10,
        description="Timeout for a single Meta stream operation",
    )
    
    # Gemini settings (for post-recording captions and fallback)
    translation_api_version: str = Field(...)
    translation_model: str = Field(...)
    live_caption_model: str = Field(...)
    
    # Whisper settings. Keep live captions fast, but allow exported captions to use a better model.
    whisper_model_size: str = Field(
        default="base",
        description="Fallback Whisper model: 'tiny', 'base', 'small', 'medium'",
    )
    live_whisper_model_size: str | None = Field(
        default=None,
        description="Whisper model for live overlay captions",
    )
    captions_whisper_model_size: str | None = Field(
        default=None,
        description="Whisper model for exported subtitles",
    )
    google_translate_project_id: str | None = Field(default=None, description="Google Cloud Project ID for Translate API")
    local_translation_provider: str = Field(default="nllb", description="Local translation provider: 'nllb' or 'marian'")
    local_translation_model: str = Field(
        default="facebook/nllb-200-distilled-600M",
        description="Local translation model name (transformers hub id)",
    )
    
    # Live caption common settings
    live_caption_min_interval_seconds: int = Field(...)
    live_caption_cache_ttl_seconds: int = Field(...)
    live_caption_min_audio_bytes: int = Field(...)
    translation_quota_cooldown_seconds: int = Field(...)
    transcription_chunk_seconds: int = Field(...)
    max_transcription_seconds: int = Field(...)
    live_caption_chunk_seconds: int = Field(...)

    @field_validator(
        "live_caption_backends", "stream_translation_fallbacks", mode="before"
    )
    @classmethod
    def _split_env_list(cls, value: object) -> object:
        # Accept comma-separated env values ("meta,whisper-local") as well
        # as JSON lists (["meta", "whisper-local"]) and native lists.
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                return [str(item).strip() for item in json.loads(text)]
            return [part.strip() for part in text.split(",") if part.strip()]
        return value

    def effective_live_chain(self) -> list[str]:
        """Ordered live backends: explicit chain, else the single backend."""
        chain = [str(item).strip() for item in (self.live_caption_backends or [])]
        chain = [item for item in chain if item]
        if chain:
            return chain
        return [self.live_caption_backend]


settings = Settings()
