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

- Dataset: FER+ relabels of FER-2013 from `deanngkl/ferplus-7cls`. 35,481 labelled 48x48 grayscale faces across 7 classes (neutral, happy, sad, angry, fearful, surprised, disgusted). FER+ uses majority-vote labels from 10 human annotators per image, which is meaningfully cleaner than the original single-annotator FER-2013 labels.
- Architecture: VGG-style from-scratch CNN, four double-conv blocks (1 -> 64 -> 128 -> 256 -> 512) with batch norm and 2x2 max pooling, global average pool, 256-unit FC head with dropout 0.5. About 4.8M parameters.
- Training: 120 epochs, Adam lr 1e-3 weight decay 5e-4, linear warmup for 5 epochs then cosine annealing, balanced class weights with label smoothing 0.05. Augmentation includes random horizontal flip, affine warp, colour jitter, padded random crop, and random erasing. MixUp + CutMix on every batch (50/50 split). EMA weights with decay 0.999 used for the saved checkpoint. Test-time augmentation (centre + horizontal flip) at evaluation. Early stopping on val F1 macro with patience 20.
- Test metrics: f1_macro 0.6684, accuracy 0.7843 on the 5,321-sample held-out FER+ split (per-class breakdown saved alongside the checkpoint). Run `python -m app.models.face.train` to reproduce.

## Voice model

- Datasets combined: RAVDESS (1,440 clips, 8 emotions), CREMA-D (7,442 clips, 6 emotions), and SAVEE (480 clips, 7 emotions). 9,362 clips total covering all 8 of our target classes. CREMA-D drives the bulk of the training signal for the six core emotions; RAVDESS supplies "calm"; RAVDESS and SAVEE together cover "surprised".
- Features: 128x130 log-mel spectrogram plus an 80-dim statistical vector (40 MFCC means + 40 MFCC stds + descriptor padding). Statistical vector is per-sample z-scored in the loader and again LayerNorm'd inside the model so train/eval distributions match.
- Architecture: four double-conv blocks (1 -> 64 -> 128 -> 256 -> 512) over the spectrogram, global average pool, concatenation with the LayerNorm'd 80-d stat vector, 256-unit FC head. About 4.8M parameters.
- Training: 120 epochs, Adam lr 3e-4 weight decay 5e-4, linear warmup for 5 epochs then cosine annealing, balanced class weights with label smoothing 0.05. Augmentation includes gaussian noise injection, time-shift roll, and SpecAugment (time and frequency masks); MixUp on the cached mel + stat tensors at every batch. EMA weights with decay 0.999 used for the saved checkpoint. Early stopping on val F1 macro with patience 20.
- Test metrics: f1_macro 0.6721, accuracy 0.6570 on the 1,408-sample held-out split. Run `python -m app.models.voice.train` to reproduce.

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

The backend deploys as a Docker Space on HuggingFace; the frontend deploys as a static SPA on Vercel.

### Backend on HuggingFace Spaces

Create the Space:
1. Go to https://huggingface.co/new-space
2. Owner: your account. Space name: `interview-mirror` (or whatever).
3. License: MIT. SDK: **Docker**. Hardware: CPU basic (free) is enough.
4. (Optional, paid) Add Persistent Storage 1 GB so past sessions survive container rebuilds.

Push the `backend/` subtree to the Space. From the repo root:
```bash
git remote add hf-space https://huggingface.co/spaces/<USER>/interview-mirror
git subtree push --prefix=backend hf-space main
```
The Space runs Docker against `backend/Dockerfile`, which installs ffmpeg + libsndfile + mediapipe deps and serves uvicorn on port 7860. Whisper-base auto-downloads on first run and caches at `/data/.cache`.

In the Space settings → Variables and secrets, set:
- `CORS_ORIGINS` = your Vercel URL (e.g. `https://interview-mirror.vercel.app`)
- `SESSION_DB_PATH` = `/data/sessions.db` if you mounted persistent storage

The Space takes 5-10 minutes for the first build (Docker layer caching speeds up later pushes).

### Frontend on Vercel

1. Import the repo at https://vercel.com/new.
2. Set the **Root Directory** to `frontend`.
3. Framework preset: Vite. Build command and output directory are picked up from `vercel.json`.
4. Add environment variables:
   - `VITE_API_URL` = `https://<USER>-<space>.hf.space`
   - `VITE_WS_URL`  = `wss://<USER>-<space>.hf.space`
5. Deploy.

The SPA fetches sessions, questions, and analysis through `/api/...` and opens a WebSocket at `/ws/{sessionId}` against the URLs above. Camera and microphone require HTTPS; both Vercel and HF Spaces serve HTTPS by default.

### Local production-style run with Docker
```bash
cd backend
docker build -t interview-mirror .
docker run --rm -p 7860:7860 -v $(pwd)/data:/data interview-mirror
```

## Live demo

Add the deployed URL here once deployment is complete.
