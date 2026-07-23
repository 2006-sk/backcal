<div align="center">

# AdAtlas AI — Feature Extractor

*A FastAPI backend that turns video and image ad creatives into rich, structured features using a multi-model AI pipeline.*

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-2094F3?logo=gunicorn&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Whisper-F55036?logo=groq&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-Vision-8E75B2?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Frames-5C3EE8?logo=opencv&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Embeddings-FFC107)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)

</div>

## 📖 Overview

AdAtlas AI is the backend service (`adatlas-backend`) for analyzing advertising creatives. Given an uploaded video or image, it extracts keyframes and runs them through several AI models in parallel to produce a single, unified JSON describing the creative — vision features, an audio transcript, creative-performance signals, and vector embeddings.

The pipeline combines classic computer vision (OpenCV/FFmpeg for frame extraction, color-tone and scene-count estimation) with hosted LLMs: **Google Gemini** for vision analysis and captioning, **Groq Whisper** for audio transcription, **Reka** for creative scoring, and **OpenAI** for embedding generation (plus a vision fallback). Resulting embeddings can optionally be persisted to **ChromaDB** for similarity search. Every external model is opt-in via environment flags, so the service degrades gracefully when a provider is not configured.

## ✨ Features

- **File upload** — accepts videos (`.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`) and images (`.jpg`, `.jpeg`, `.png`, `.webp`), stored with a generated ID.
- **Unified multi-model analysis** (`/analyze`) — for videos, runs Gemini vision, Reka QuickTag, Groq audio transcription, and visual embedding generation concurrently, then merges everything into one response.
- **Image analysis** (`/analyze_image`) — Gemini-based vision analysis and visual embeddings for still images.
- **Frame & scene understanding** — FFmpeg keyframe extraction (with an OpenCV fallback), warm/cool color-tone detection via HSV, and scene-count estimation via histogram correlation.
- **Audio transcription** — extracts audio and transcribes it with Groq Whisper (`whisper-large-v3`).
- **Creative scoring** — Reka QuickTag features such as expected CTR, virality score, keywords, and mood/tone.
- **Vector embeddings** — audio and visual embeddings that can be stored in ChromaDB (ChromaDB Cloud client) when enabled.
- **Batch processing** (`/batch`) — analyze multiple uploaded files in parallel.
- **Chatbot indexing** (`/chatbot/index_video`) — index a video with Reka for downstream chatbot use.
- **Health & debug endpoints** — liveness/readiness probes plus debug routes for benchmarking and testing individual services.

## 🛠️ Tech Stack

| Area | Technology |
| --- | --- |
| Language | Python 3.11 |
| Web framework | FastAPI + Uvicorn (ASGI) |
| Config / validation | Pydantic v2, pydantic-settings |
| Media processing | ffmpeg-python, OpenCV (`opencv-python-headless`), Pillow |
| Vision & captioning | Google Gemini (`google-generativeai`) |
| Audio transcription | Groq (`whisper-large-v3`) |
| Creative scoring | Reka Vision Agent API |
| Embeddings / fallback | OpenAI |
| Vector store | ChromaDB (optional) |
| Numerics | NumPy, scikit-learn |
| HTTP clients | httpx, requests |

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- [FFmpeg](https://ffmpeg.org/) installed and available on your `PATH`
- API keys for the providers you intend to enable (Groq, Gemini, Reka, OpenAI, ChromaDB)

### Installation

```bash
git clone https://github.com/2006-sk/backcal.git
cd backcal/adatlas-backend

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r Requirements.txt
# ChromaDB is imported by the app but not pinned in Requirements.txt; install it too:
pip install chromadb
```

### Configuration

The app reads settings from a `.env` file in the `adatlas-backend` directory (see `app/core/config.py`). Every external service is **disabled by default** — enable each with its `USE_*` flag *and* provide the matching key:

```env
# Groq (audio transcription)
USE_GROQ=true
GROQ_API_KEY=your_groq_key

# Google Gemini (vision + captions)
USE_GEMINI=true
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash   # or gemini-2.5-pro

# Reka (creative scoring / QuickTag)
USE_REKA=true
REKA_API_KEY=your_reka_key

# OpenAI (embedding generation / vision fallback)
OPENAI_API_KEY=your_openai_key

# ChromaDB (optional embedding storage — ChromaDB Cloud)
USE_CHROMA=true
CHROMA_API_KEY=your_chroma_key
CHROMA_TENANT=your_tenant
CHROMA_DATABASE=your_database
CHROMA_COLLECTION_NAME=adatlas_embeddings
```

### Usage

From the `adatlas-backend` directory, start the API server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then upload a file and analyze it:

```bash
# 1) Upload
curl -F "file=@sample.mp4" http://localhost:8000/upload
# -> { "id": "<file_id>", "file_name": "...", "saved_path": "..." }

# 2) Analyze
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"file_id": "<file_id>"}'
```

Interactive API docs are available at `http://localhost:8000/docs`.

## 📁 Project Structure

```
backcal/
└── adatlas-backend/
    ├── app/
    │   ├── main.py              # FastAPI app + all route handlers
    │   ├── core/config.py       # Settings (env vars, model selection)
    │   ├── models/schemas.py    # Pydantic request/response models
    │   ├── services/            # Gemini, Groq, Reka, OpenAI, Chroma,
    │   │                        #   audio & visual embedding helpers
    │   └── utils/               # timing, unified output, Reka client
    ├── Requirements.txt
    ├── run.sh
    ├── API_ROUTES.md            # Endpoint reference
    └── test_*.py                # Standalone integration/model tests
```

## 📚 API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/`, `/health`, `/health/live`, `/health/ready` | Health / liveness / readiness checks |
| `POST` | `/upload` | Upload a video or image file |
| `POST` | `/analyze` | Full multi-model analysis (video or image) |
| `POST` | `/analyze_image` | Image-only Gemini vision analysis |
| `POST` | `/batch` | Analyze multiple uploaded files in parallel |
| `POST` | `/chatbot/index_video` | Index a video with Reka for chatbot use |
| `GET` | `/debug/*` | Demo, benchmark, and per-service test routes |

See [`adatlas-backend/API_ROUTES.md`](adatlas-backend/API_ROUTES.md) for full request/response examples.
