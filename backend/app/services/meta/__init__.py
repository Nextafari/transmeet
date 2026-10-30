"""Meta-powered live transcription services (isolated from Whisper/Gemini).

Nothing here imports the Whisper or Gemini backends. Shared, provider-free
helpers (local NLLB engine, TranslatorBackend ABC) are imported from their
own modules.
"""

from app.services.meta.events import (
    PROTOCOL_VERSION,
    WS_PATH,
    StreamParams,
    StreamServerMessage,
    parse_stream_params,
)
from app.services.meta.file_backend import MetaFileBackend
from app.services.meta.spark_client import MetaSparkClient, MetaSparkError
from app.services.meta.streaming import LiveStreamSession
from app.services.meta.translation import (
    KNOWN_LEGS,
    ordered_legs,
    translate_with_legs,
)
from app.services.meta.voice_client import (
    ASR_MODES,
    MetaVoiceClient,
    MetaVoiceError,
)

__all__ = [
    "ASR_MODES",
    "KNOWN_LEGS",
    "PROTOCOL_VERSION",
    "WS_PATH",
    "LiveStreamSession",
    "MetaFileBackend",
    "MetaSparkClient",
    "MetaSparkError",
    "MetaVoiceClient",
    "MetaVoiceError",
    "StreamParams",
    "StreamServerMessage",
    "ordered_legs",
    "parse_stream_params",
    "translate_with_legs",
]
