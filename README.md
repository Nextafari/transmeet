# Transmeet

Transmeet is a Chrome extension + FastAPI backend for recording browser tabs with **live translated captions**. It captures tab audio/video, streams or chunks audio to the backend for transcription + translation, shows a live caption overlay, and can generate or burn in subtitles for exported recordings.

## Features

- Tab screen/audio recording from a Chrome extension (Manifest V3)
- Live caption overlay with realtime translation
- Realtime PCM streaming over WebSocket, with automatic fallback to a chunked (~1 s) caption path
- Server-side transcription/translation backend chain (Meta, Baseten, Whisper-local, Gemini, Google)
- Subtitle generation (`srt`/`vtt`) and burn-in for exported videos
- Media conversion scaffold (ffmpeg-backed)

## Repository layout

```
transmeet/
├── frontend/          # Chrome extension (Manifest V3, no build step)
│   ├── manifest.json
│   ├── popup.html / popup.js
│   ├── background.js
│   ├── offscreen.html / offscreen.js / live-stream.js
│   ├── audio-worklet/pcm-tap.js
│   └── ARCHITECTURE.md
└── backend/           # FastAPI API (transcription, translation, media)
    ├── app/
    │   ├── api/       # routers and routes
    │   ├── core/      # settings/config
    │   ├── schemas/   # request/response models
    │   └── services/  # translator, media, baseten/meta integrations
    ├── tests/
    ├── requirements.txt
    ├── Dockerfile
    └── compose.yaml
```

See [frontend/ARCHITECTURE.md](frontend/ARCHITECTURE.md) for the extension's message flow, offscreen-document design, and streaming-vs-chunked caption paths.

## Prerequisites

- Python 3.11+ and `ffmpeg` (for media endpoints) to run the backend
- Docker (optional) for containerized backend runs
- Chrome/Chromium to load the extension
- API keys only for the providers you use (see [Configuration](#configuration)); `whisper-local` + `nllb` run fully offline

## Backend setup

From `backend/`:

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in the keys you need
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

Or with Docker Compose (backend only):

```bash
cd backend
docker compose up --build
```

Details: [backend/README.md](backend/README.md)

## Frontend setup

The extension has no build step. Load it unpacked:

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** and select the `frontend/` directory
4. Click the Transmeet icon to start recording; make sure the backend is running first

## API overview

Base prefix: `/api/v1`

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET` | `/` | Root message |
| `GET` | `/api/v1/health/` | Health check |
| `POST` | `/api/v1/translation/text` | Text translation |
| `POST` | `/api/v1/media/captions` | Transcription + translation + subtitles (`srt`/`vtt`) |
| `POST` | `/api/v1/media/live-caption` | Short-chunk transcription for the live overlay |
| `GET` | `/api/v1/media/live-caption/config` | Overlay timing/config, incl. `streaming` capability block |
| `WS` | `/api/v1/media/live-stream` | Realtime PCM streaming with `downgrade` to chunked path |
| `POST` | `/api/v1/media/burn-subtitles` | Burn subtitles into an exported video |
| `POST` | `/api/v1/media/convert` | Video conversion scaffold |

## Configuration

Configured via `backend/.env` (see `backend/.env.example`). Key settings:

- `LIVE_CAPTION_BACKENDS` — ordered backend chain, primary first (e.g. `meta,whisper-local`); entries: `meta`, `baseten`, `whisper-google`, `whisper-local`, `gemini`
- `META_MODEL_API_KEY` — required for the `meta` live backend and `spark` translation leg
- `BASETEN_API_KEY` — required for the `baseten` live backend and translation leg
- `GOOGLE_API_KEY` — required for Gemini / Google Translate routes
- `LOCAL_TRANSLATION_PROVIDER` / `LOCAL_TRANSLATION_MODEL` — local translation (default `nllb`)
- `LIVE_STREAM_ENABLED`, `LIVE_STREAM_FRAME_MS`, `LIVE_STREAM_PARTIAL_*` — realtime transport tuning

Model/engine selection is server-side only: clients send audio plus source/target languages and never choose backends. Missing keys skip those chain rungs; quota-like failures trigger a per-rung cooldown.

## Tests

From `backend/`:

```bash
pytest
```

## License

No license specified yet.
