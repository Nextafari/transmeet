"""Backend <-> frontend live-stream protocol (version 1).

The frontend sends raw PCM frames over the WebSocket and receives caption
messages. The frontend never names a model or engine: only transport fields
(source/target languages, translate flag, protocol version) are read from the
connection, and anything else is ignored.
"""

from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field

PROTOCOL_VERSION = "1"
WS_PATH = "/api/v1/media/live-stream"
# Query keys a client may send that must never influence model selection.
IGNORED_CLIENT_KEYS = frozenset({"engine", "backend", "model", "provider"})


class StreamServerMessage(BaseModel):
    """One message from the backend to the frontend."""

    type: Literal["partial", "final", "downgrade", "error"]
    text: str = ""
    turn_id: int | None = None
    reason: str | None = None


class StreamParams(BaseModel):
    """Transport-only parameters for a live-stream session."""

    source_language: str | None = None
    target_language: str = "en"
    translate: bool = True


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_source_language(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "auto", "detect", "auto-detect", "autodetect"}:
        return None
    return normalized


def parse_stream_params(raw: Mapping[str, Any] | Any) -> StreamParams:
    """Extract transport params, ignoring model/engine selection keys."""
    get = raw.get if hasattr(raw, "get") else (lambda key, default=None: default)
    target = str(get("target_language") or "").strip().lower() or "en"
    return StreamParams(
        source_language=_normalize_source_language(get("source_language")),
        target_language=target,
        translate=_parse_bool(get("translate", True)),
    )


class ClientEndSignal(BaseModel):
    """Optional JSON text frame the frontend may send to end cleanly."""

    type: str = Field(default="audio")
