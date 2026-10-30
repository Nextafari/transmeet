"""Muse Voice Transcribe client (Meta Model API).

Speaks the realtime WebSocket protocol and the one-shot file endpoint
documented at https://dev.meta.ai/docs/speech-to-text, following Meta's own
client recipe (meta-models/meta-model-cookbook, 06_muse_voice).

Wire notes that matter here:
- Realtime: ``wss://api.meta.ai/v1/asr/realtime``. The credential rides in
  the first JSON handshake frame (``authorization.accessToken``); the
  ``Authorization`` header is ignored on this endpoint. Audio goes as raw
  PCM16 mono binary frames. ``{"type": "endStream"}`` ends the session.
  The handshake ack is a bare ``{"sessionId": ...}`` struct.
- One-shot: multipart ``POST https://api.meta.ai/v1/asr/transcribe`` with a
  ``request`` JSON part plus an ``audio`` file part, Bearer auth header.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

REALTIME_URL = "wss://api.meta.ai/v1/asr/realtime"
TRANSCRIBE_URL = "https://api.meta.ai/v1/asr/transcribe"
ASR_MODEL = "muse-voice-transcribe-1.0"
ASR_MODES = ("PUSH_TO_TALK", "ENDPOINTING", "DIARIZATION")
MAX_FILE_BYTES = 32 * 1024 * 1024


class MetaVoiceError(RuntimeError):
    """A Meta speech-to-text failure. ``quota`` marks rate-limit/backoff."""

    def __init__(self, message: str, *, quota: bool = False) -> None:
        super().__init__(message)
        self.quota = quota


def _looks_like_quota(message: str) -> bool:
    lowered = message.lower()
    return "429" in lowered or "quota" in lowered or "rate limit" in lowered


class MetaVoiceClient:
    """One Meta realtime session (or one-shot file request)."""

    def __init__(
        self,
        api_key: str | None,
        model: str = ASR_MODEL,
        mode: str = "ENDPOINTING",
        *,
        language_bias: Optional[list[str]] = None,
        keywords: Optional[list[str]] = None,
        audio_encoding: str = "PCM_16KHZ",
        realtime_url: str = REALTIME_URL,
        transcribe_url: str = TRANSCRIBE_URL,
        timeout: float = 30.0,
    ) -> None:
        if not (api_key or "").strip():
            raise MetaVoiceError(
                "META_MODEL_API_KEY is not configured; the 'meta' backend "
                "cannot run without it."
            )
        if mode not in ASR_MODES:
            raise MetaVoiceError(
                f"Unsupported Meta ASR mode: {mode}. Use one of {ASR_MODES}."
            )
        self._api_key = api_key.strip()
        self._model = model
        self._mode = mode
        self._language_bias = [str(item) for item in (language_bias or []) if str(item).strip()]
        self._keywords = [str(item) for item in (keywords or []) if str(item).strip()]
        self._audio_encoding = audio_encoding
        self._realtime_url = realtime_url
        self._transcribe_url = transcribe_url
        self._timeout = timeout
        self._ws: Any = None
        self.session_id: str | None = None

    def _handshake(self) -> dict[str, Any]:
        # Per Meta's client recipe, the raw key goes in the handshake JSON
        # (no "Bearer " prefix, no HTTP header on this endpoint).
        handshake: dict[str, Any] = {
            "mode": self._mode,
            "authorization": {"accessToken": self._api_key},
            "audioEncoding": self._audio_encoding,
            "model": self._model,
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": False,
        }
        if self._keywords:
            handshake["keywords"] = self._keywords
        if self._language_bias:
            handshake["languageBias"] = self._language_bias
        return handshake

    async def connect(self) -> str:
        """Open the session. Returns the server's session id."""
        import websockets

        session_id = f"transmeet-{uuid.uuid4()}"
        url = f"{self._realtime_url}?sessionId={session_id}"
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(url, max_size=None),
                timeout=self._timeout,
            )
            await self._ws.send(json.dumps(self._handshake()))
            raw = await asyncio.wait_for(self._ws.recv(), timeout=self._timeout)
        except MetaVoiceError:
            raise
        except Exception as exc:
            raise MetaVoiceError(f"Meta realtime connect failed: {exc}") from exc

        try:
            ack = json.loads(raw) if isinstance(raw, str) else {}
        except (ValueError, TypeError):
            ack = {}
        if not isinstance(ack, dict):
            ack = {}
        # The handshake ack is a bare {"sessionId": ...} struct.
        if "sessionId" not in ack:
            message = str(ack.get("message", raw))[:300]
            await self.aclose()
            raise MetaVoiceError(
                f"Meta realtime handshake rejected: {message}",
                quota=_looks_like_quota(message),
            )
        self.session_id = str(ack["sessionId"])
        logger.info("Meta realtime session started: %s", self.session_id)
        return self.session_id

    async def send_pcm(self, data: bytes) -> None:
        if not self._ws:
            raise MetaVoiceError("Meta realtime session is not connected")
        try:
            await self._ws.send(bytes(data))
        except Exception as exc:
            raise MetaVoiceError(f"Meta realtime send failed: {exc}") from exc

    async def recv_event(self) -> dict[str, Any]:
        """Return the next normalized server event.

        Normalizes the bare handshake ack to ``{"type": "session", ...}``.
        Raises MetaVoiceError on error frames; ignores binary frames.
        """
        if not self._ws:
            raise MetaVoiceError("Meta realtime session is not connected")
        while True:
            try:
                message = await self._ws.recv()
            except Exception as exc:
                raise MetaVoiceError(
                    f"Meta realtime session closed: {exc}"
                ) from exc
            if isinstance(message, bytes):
                continue
            try:
                event = json.loads(message)
            except (ValueError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            if "sessionId" in event and "type" not in event:
                return {"type": "session", "sessionId": event["sessionId"]}
            if event.get("type") == "error":
                detail = str(event.get("message", "unknown transcription error"))
                raise MetaVoiceError(
                    f"Meta realtime error: {detail[:300]}",
                    quota=_looks_like_quota(detail),
                )
            return event

    async def end_stream(self) -> None:
        if not self._ws:
            return
        try:
            await self._ws.send(json.dumps({"type": "endStream"}))
        except Exception:
            pass

    async def aclose(self) -> None:
        ws, self._ws = self._ws, None
        self.session_id = None
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            pass

    async def transcribe_file(
        self,
        audio_path: Path,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """One-shot transcription. Returns ``{"transcript", "turns"}``.

        Turns carry turn-level ``start_sec`` timings (no word timestamps;
        Meta returns turn-level timestamps only).
        """
        size = audio_path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise MetaVoiceError(
                f"Audio is {size / 1024 / 1024:.1f} MB; Meta caps file "
                "uploads at 32 MB."
            )

        request_part: dict[str, Any] = {
            "mode": self._mode,
            "model": self._model,
            "audioEncoding": "WAV",
        }
        if self._keywords:
            request_part["keywords"] = self._keywords
        if self._language_bias:
            request_part["languageBias"] = self._language_bias

        params = {"sessionId": session_id or f"transmeet-{uuid.uuid4()}"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                with audio_path.open("rb") as audio_file:
                    response = await client.post(
                        self._transcribe_url,
                        params=params,
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        files={
                            "request": (
                                None,
                                json.dumps(request_part),
                                "application/json",
                            ),
                            "audio": (audio_path.name, audio_file, "audio/wav"),
                        },
                    )
        except MetaVoiceError:
            raise
        except Exception as exc:
            raise MetaVoiceError(f"Meta transcribe request failed: {exc}") from exc

        if response.status_code == 429:
            raise MetaVoiceError(
                f"Meta transcribe rate limited (429): {response.text[:300]}",
                quota=True,
            )
        if response.status_code == 401:
            raise MetaVoiceError(
                "Meta transcribe rejected the credential (401). "
                "Check META_MODEL_API_KEY."
            )
        if response.status_code != 200:
            raise MetaVoiceError(
                f"Meta transcribe failed ({response.status_code}): "
                f"{response.text[:300]}",
                quota=_looks_like_quota(response.text),
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise MetaVoiceError(
                f"Meta transcribe returned invalid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise MetaVoiceError("Meta transcribe returned an unexpected payload")

        turns: list[dict[str, Any]] = []
        raw_turns = payload.get("turns") or []
        if isinstance(raw_turns, list):
            started = sorted(
                (turn for turn in raw_turns if isinstance(turn, dict)),
                key=lambda turn: float(turn.get("startMs") or 0),
            )
            for index, turn in enumerate(started):
                text = str(turn.get("transcript") or "").strip()
                if not text:
                    continue
                try:
                    start_sec = max(0.0, float(turn.get("startMs") or 0) / 1000.0)
                except (TypeError, ValueError):
                    start_sec = 0.0
                turns.append(
                    {
                        "turn_id": turn.get("turnId", index),
                        "start_sec": start_sec,
                        "speaker": turn.get("speaker"),
                        "text": text,
                    }
                )

        transcript = str(payload.get("transcript") or "").strip()
        if not transcript:
            transcript = " ".join(turn["text"] for turn in turns).strip()
        return {"transcript": transcript, "turns": turns}
