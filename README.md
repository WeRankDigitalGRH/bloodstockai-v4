# BloodstockAI

Equine biomechanical analysis · DeepLabCut SuperAnimal-Quadruped · BAI Score™

## Deploy to Vercel

### 1. Push to GitHub

```bash
git init && git add -A && git commit -m "init"
gh repo create bloodstockai --push --source=. --public
```

### 2. Create Vercel project

Go to [vercel.com/new](https://vercel.com/new), import the repo. No build settings needed — `vercel.json` handles everything.

### 3. Add Postgres database

In your Vercel project dashboard:

1. Go to **Storage** tab → **Create Database** → **Postgres**
2. Follow prompts (uses Neon under the hood, free tier available)
3. Vercel auto-injects `POSTGRES_URL` into your environment

### 4. Deploy

Click **Deploy**. The app is live. Frontend + API + database all working.

### 5. Connect GPU worker (for video analysis)

The Vercel serverless functions handle the web UI and API. Actual DeepLabCut video processing requires a GPU machine running the worker:

```bash
# On any machine with NVIDIA GPU + Python 3.9-3.11
cd worker/
pip install -r requirements.txt
export POSTGRES_URL="postgresql://..."  # same connection string from Vercel
python worker.py
```

Or with Docker:
```bash
cd worker/
docker build -t bai-worker .
docker run --gpus all -e POSTGRES_URL="postgresql://..." bai-worker
```

The worker polls Postgres for pending jobs and processes them automatically.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  Vercel (serverless)                        │
│  ┌──────────┐  ┌──────────────────────────┐ │
│  │  Static   │  │  FastAPI (api/index.py)  │ │
│  │  HTML/JS  │  │  /analyze  /status       │ │
│  │  React    │  │  /results  /history      │ │
│  └──────────┘  └───────────┬──────────────┘ │
└────────────────────────────┼────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Neon Postgres   │
                    │  (Vercel Storage)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  GPU Worker      │
                    │  DeepLabCut 3.0  │
                    │  (your server)   │
                    └─────────────────┘
```

**Flow:**
1. User uploads video → Vercel function creates job in Postgres
2. GPU worker polls Postgres → finds pending job → downloads video
3. Worker runs DLC SuperAnimal-Quadruped → 39 keypoints → maps to 22 BAI points
4. Worker computes conformation (9 metrics), movement (6 metrics), BAI Score™
5. Worker writes results to Postgres → frontend polls and displays

## Project Structure

```
bloodstockai/
├── api/
│   └── index.py            # FastAPI app (Vercel serverless function)
├── public/
│   └── index.html          # React frontend (Racing Post style)
├── worker/
│   ├── worker.py           # GPU worker (runs separately)
│   ├── requirements.txt    # ML dependencies (DLC, OpenCV, etc.)
│   └── Dockerfile          # GPU-ready container
├── vercel.json             # Vercel routing config
├── requirements.txt        # Vercel function deps (lightweight)
├── .env.example            # Environment variables reference
├── .gitignore
└── README.md
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves frontend |
| GET | `/api` | API info + DLC model status |
| GET | `/health` | Health check + DB status |
| POST | `/analyze` | Upload video for analysis |
| GET | `/status/{id}` | Poll job progress |
| GET | `/results/{id}` | Get completed results |
| GET | `/history` | List past analyses |
| DELETE | `/jobs/{id}` | Delete job + data |
| POST | `/api/worker/complete` | Worker callback (internal) |

## Environment Variables

Set in **Vercel Dashboard → Settings → Environment Variables**:

| Variable | Required | Description |
|----------|----------|-------------|
| `POSTGRES_URL` | Yes | Auto-set by Vercel Postgres |
| `BAI_WORKER_SECRET` | Recommended | Auth token for worker callbacks |
| `BAI_WORKER_URL` | Optional | Worker URL for job notifications |
| `BAI_DLC_MODEL` | No | Default: `superanimal_quadruped` |
| `BAI_DLC_BACKBONE` | No | Default: `hrnet_w32` |

## Local Development

```bash
# Install Vercel CLI
npm i -g vercel

# Link to project and pull env vars
vercel link
vercel env pull .env

# Run locally
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8000
```
