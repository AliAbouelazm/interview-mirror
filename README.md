# Interview Mirror

Interview Mirror watches you during a mock interview through your webcam and microphone, processing three simultaneous streams in real time: facial expressions, voice acoustics, and the words you speak. The technically interesting part is that the face and voice models are trained from scratch on labelled emotion data from HuggingFace, Whisper-base handles transcription locally, and a small custom fusion network combines the three streams into a continuous confidence and engagement score every 500ms. The system gives moment-by-moment feedback during the session and a full breakdown of strongest and weakest moments after.

## Architecture

```
   Webcam (320x240 JPEG)         Mic (22.05kHz Float32 PCM)
            |                              |
       WebSocket frames every 500ms        |
            |                              |
            v                              v
+-------------------------+      +-------------------------+
|   Face CNN (FER-2013)   |      |  Voice CNN (RAVDESS)    |
|   7-class softmax       |      |  8-class softmax        |
|   + interview signal    |      |  + acoustic features    |
+-------------------------+      +-------------------------+
            \                              /
             \                            /
              v                          v
            +---------------------------------+
            |   Whisper-base (every 1s)       |
            |   transcript -> filler/hedge    |
            |   language confidence           |
            +---------------------------------+
                          |
                          v
                +--------------------+
                |   Fusion network   |
                |   21 dim -> conf,  |
                |   engagement (MLP) |
                +--------------------+
                          |
                          v
                  WebSocket payload
                          |
                          v
                React frontend (gauge,
                timeline, transcript,
                signal pills)
```

## Components

| Component | Technology | Purpose |
| --- | --- | --- |
| Face model | PyTorch CNN, 48x48 grayscale | Classify facial emotion, map to interview signal |
| Voice model | PyTorch CNN on log-mel + 80 acoustic stats | Classify vocal emotion, extract speaking rate and energy |
| Transcription | faster-whisper, Whisper-base, int8 | Convert speech to text on a 5s rolling window |
| Language analysis | Custom regex tokenizer | Filler words, hedging phrases, assertive verbs |
| Fusion network | PyTorch MLP, 21-dim input | Two-headed regressor for confidence and engagement |
| Real-time orchestrator | FastAPI WebSocket | 500ms cycle, adaptive to 1s when CPU-bound |
| Session store | aiosqlite | Persists frames and analyses across restarts |
| Frontend | React 18, Vite, Recharts | Live session UI and post-session analysis |

## Face model

- Dataset: FER-2013 from HuggingFace mirror `clip-benchmark/wds_fer2013`. 35,887 labelled 48x48 grayscale faces in 7 classes (neutral, happy, sad, angry, fearful, surprised, disgusted).
- Architecture: 3 conv blocks (1->32->64->128) with batchnorm and 2x2 max pool, dropout 0.5, 256-unit FC, dropout 0.3, 7-class head.
- Training: 50 epochs, Adam lr 1e-3 weight decay 1e-4, cosine annealing, balanced class weights, augmentation with horizontal flip, 10 degree rotation, color jitter. Early stopping on val F1 macro with patience 8.
- Test metrics: see `backend/saved_models/face_best.metrics.json` after training. Run `python -m app.models.face.train` to reproduce.

## Voice model

- Dataset: RAVDESS from HuggingFace, 1,440 audio files in 8 classes (neutral, calm, happy, sad, angry, fearful, disgusted, surprised).
- Architecture: 3 conv blocks over the 128x130 log-mel spectrogram, global average pool, concatenation with 80 statistical features (40 MFCC means + 40 MFCC stds + 6 acoustic descriptors), 256-unit FC, 8-class head.
- Training: same optimizer schedule as face. Augmentation through gaussian noise injection and time-shift roll.
- Test metrics: see `backend/saved_models/voice_best.metrics.json` after training. Run `python -m app.models.voice.train` to reproduce.

## Fusion model

- Architecture: LayerNorm over 21 input features, Linear 21->64 with dropout 0.2, Linear 64->32, two sigmoid heads scaled to 0-100 (confidence, engagement).
- Inputs per 500ms cycle: 7-dim face softmax, 8-dim voice softmax, 3 acoustic descriptors (speaking rate, energy, filler probability), 3 language descriptors (filler rate, hedge rate, language confidence).
- Synthetic training is valid because the fusion model is learning to combine signals that already carry semantic meaning from the trained base models. It is not learning low-level features. The training distribution samples three behavioural profiles (high, mid, low) and adds gaussian noise so the fusion network does not overfit to a single canonical pattern.
- Run `python -m app.models.fusion.train` after the base models are trained.

## Real-time pipeline

The 500ms cycle is allocated like this on Apple Silicon CPU inference:

| Component | Typical latency |
| --- | --- |
| Face detection + CNN | 20-40ms |
| Voice CNN (3s window) | 30-60ms |
| Fusion forward | 1-3ms |
| Whisper-base (every 2 cycles, 5s window) | 200-700ms |
| WebSocket round trip | 5-15ms |

If total cycle time exceeds the target, the orchestrator slows to a maximum of 1000ms and reports `actual_cycle_ms` in every payload. The frontend reads this and shows it as a latency indicator.

Whisper does not block the cycle. It runs every 2 cycles and the latest transcript is folded into the next available payload.

## Post-session analysis

Insights are generated from real session data, not from a fixed template list. Each insight references a specific timestamp or measured rate. Examples:

- "You used 'um' 14 times" cites the actual most-frequent filler and its count.
- "Lowest stretch was 02:30 to 02:40" cites the worst 10s window by mean confidence and which signal drove it.
- "Voice trailed off near 03:50" cites the first window where vocal energy fell more than 30% below session average.

Empty sessions and short sessions are handled with a single explanatory insight rather than fabricating templates.

## Running locally

Backend:

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train models (in order). Apple Silicon uses MPS automatically.
python -m app.models.face.train
python -m app.models.voice.train
python -m app.models.fusion.train

# Run the server.
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
# open http://localhost:5173
```

Run tests:

```bash
cd backend
pytest app/tests
```

## Browser requirements

Chrome or Firefox on desktop. Camera and microphone access required. The mic stream is captured at 48kHz, downsampled to 22.05kHz Float32 PCM in an AudioWorklet, and sent in 4096-sample frames over the WebSocket. The video stream is downscaled to 320x240 in a hidden canvas and encoded as JPEG at quality 0.7 every 500ms.

Permissions are requested up front from the home page. If denied, the app explains how to re-enable access in browser settings and does not navigate into the session.

## Deployment

Backend on HuggingFace Spaces (Docker):

1. Push this repository to a Space configured as a Docker SDK Space.
2. The included `backend/Dockerfile` installs ffmpeg, libsndfile, mediapipe deps and serves uvicorn on port 7860.
3. Trained checkpoints live in `backend/saved_models/`. Whisper-base auto-downloads on first run.
4. SQLite session DB lives in `/data/sessions.db` inside the container. Mount a Persistent Storage volume on the Space to keep past sessions across rebuilds.

Frontend on Vercel:

1. Import the repository and pick the `frontend` directory as the project root.
2. Set environment variables `VITE_API_URL` and `VITE_WS_URL` to your HF Space URL (`https://<user>-<space>.hf.space` and `wss://<user>-<space>.hf.space`).
3. Vercel runs `npm run build` and serves the SPA. The included `vercel.json` rewrites every route to `index.html`.

## Live demo

Add the deployed URL here once deployment is complete.
