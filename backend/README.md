# Screen Recorder Backend (FastAPI)

## Run locally

1. Create and activate a virtual environment.
1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Copy environment template:

```bash
cp .env.example .env
```

1. Start the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Run with Docker Compose

1. From this backend directory, start the backend service:

```bash
docker compose up --build
```

1. Open the API docs at `http://localhost:8000/docs`.

The Compose file is intentionally scoped to backend-only deployment and does not include the Chrome extension frontend. It uses `Dockerfile` to build the image and provides sensible defaults for app settings. If you want custom values, export them in your shell or run `docker compose --env-file .env up --build` after creating `.env` from `.env.example`.

If you want a one-off container instead of Compose, you can still build and run the Docker image directly:

```bash
docker build -t screen-recorder-backend .
docker run --rm -p 8000:8000 --env-file .env screen-recorder-backend
```

The image includes `ffmpeg`, which is required by the media conversion endpoint.

## Endpoints

- `GET /` root message
- `GET /api/v1/health/` health check
- `POST /api/v1/translation/text` text translation
- `POST /api/v1/media/captions` transcription + translation + subtitle generation (`srt`/`vtt`)
- `POST /api/v1/media/live-caption` short chunk transcription for live overlay
- `GET /api/v1/media/live-caption/config` live overlay timing/config values, including the `streaming` capability block
- `WS /api/v1/media/live-stream` realtime PCM streaming (Meta) with automatic `downgrade` to the chunked path
- `POST /api/v1/media/burn-subtitles` burn subtitles into an exported video
- `POST /api/v1/media/convert` basic video conversion scaffold

## Notes

- Gemini and Google Translate routes require `GOOGLE_API_KEY` in `.env`; `whisper-local` runs transcription and translation locally. Meta routes (`meta` live backend, `spark` translation leg) require `META_MODEL_API_KEY`. Baseten routes (`baseten` live backend, `baseten` translation leg) require `BASETEN_API_KEY`.
- Live caption serving walks the ordered `LIVE_CAPTION_BACKENDS` chain (primary first, then fallbacks). The `meta` and `baseten` rungs are skipped without their keys; quota-like failures start a per-rung cooldown. Translation legs are `nllb`, `spark`, and `baseten`. Model/engine selection is server-side only — clients send audio plus source/target languages and never choose backends.
- `media/convert`, `media/captions`, and `media/burn-subtitles` require ffmpeg installed on the runtime host.
