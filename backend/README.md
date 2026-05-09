---
title: Interview Mirror
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Real-time feedback on how you come across in interviews.
---

# Interview Mirror backend

FastAPI service that runs the face emotion CNN, the voice emotion CNN, the
fusion network, and Whisper transcription. WebSocket endpoint at `/ws/{session_id}`
streams realtime per-frame analytics. HTTP API under `/api/`.

## Endpoints
- `POST /api/session/start`
- `POST /api/session/{id}/end`
- `GET  /api/session/{id}/analysis`
- `GET  /api/session/{id}/timeline`
- `GET  /api/sessions`
- `GET  /api/questions`, `POST /api/questions/set`
- `POST /api/session/{id}/question` (record question event with optional self-rating)
- `POST /api/session/{id}/face-ticks` (record head pose / eye / smile batch)
- `POST /api/session/{id}/bookmark`
- `GET  /api/model/metrics`
- `GET  /api/health`
- `WS   /ws/{session_id}` (video frames + audio PCM + face metrics in, realtime payload out)

## Local
```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

## Spaces deploy
This directory is the root of the Space. The `Dockerfile` installs ffmpeg + libsndfile,
copies `app/` and `saved_models/`, and runs uvicorn on port 7860. Whisper-base downloads
on first run and caches inside the container. Mount a Persistent Storage volume at
`/data` to keep past sessions across rebuilds (otherwise SQLite is ephemeral).

Configure environment variables in the Space settings:
- `CORS_ORIGINS` set to your Vercel domain, e.g. `https://your-app.vercel.app`
- `SESSION_DB_PATH=/data/sessions.db` if persistent storage is mounted
