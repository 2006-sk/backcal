# AdAtlas AI Backend - API Routes

## Base URL
```
http://localhost:8000
```

---

## 1. Health Check Endpoints

### GET `/`
**Purpose:** Basic health check
**Response:**
```json
{
  "status": "ok"
}
```

### GET `/health`
**Purpose:** Health check
**Response:**
```json
{
  "status": "ok"
}
```

### GET `/health/live`
**Purpose:** Liveness probe
**Response:**
```json
{
  "live": true
}
```

### GET `/health/ready`
**Purpose:** Readiness probe
**Response:**
```json
{
  "ready": true
}
```

---

## 2. File Upload

### POST `/upload`
**Purpose:** Upload video or image file for analysis
**Request:** multipart/form-data
- `file`: The video/image file

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "file_name": "sample.mp4",
  "saved_path": "/path/to/data/uploads/550e8400...__sample.mp4"
}
```

**Frontend Usage:** Upload files before analysis. Store the `id` for the analyze endpoint.

---

## 3. Video Analysis

### POST `/analyze`
**Purpose:** Analyze video file using Gemini + Groq + Reka + ChromaDB
**Request:** JSON body
```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000"
}
```
OR
```json
{
  "file_path": "/absolute/path/to/video.mp4"
}
```

**Response Structure:**
```json
{
  "gemini": {
    "file_name": "sample.mp4",
    "duration": 15.02,
    "scene_count": 2,
    "color_tone": "warm",
    "vision_features": {
      "faces_detected": 4,
      "objects": ["backpack", "vitamin bottles", "kitchen"],
      "mood": "joyful",
      "color_scheme": "vibrant pinks and yellows",
      "cta_present": false
    },
    "video_summary": "A vibrant, empowering ad showing..."
  },
  "reka_features": {
    "expected_ctr": 1.8,
    "virality_score": 70,
    "keywords": ["health", "daily routine", "family"],
    "mood_tone": ["motivating", "optimistic"]
  },
  "audio_embedding": {
    "transcript": "Being a woman, it's a lot...",
    "embedding_dimensions": 768,
    "error": null
  },
  "visual_embedding": {
    "captions": [
      "A woman unboxes a package",
      "Kitchen with colorful items"
    ],
    "embedding_dimensions": 768,
    "frame_count": 12,
    "error": null
  },
  "unified": {
    "file_name": "sample.mp4",
    "duration": 15.02,
    "scene_count": 2,
    "color_tone": "warm",
    "summary": "A vibrant, empowering ad...",
    "visual": {
      "faces_detected": 4,
      "objects": ["backpack", "vitamin bottles"],
      "mood": "joyful",
      "color_scheme": "vibrant pinks and yellows",
      "cta_present": false
    },
    "audio": {
      "transcript": "Being a woman, it's a lot...",
      "word_count": 25
    },
    "reka": {
      "expected_ctr": 1.8,
      "virality_score": 70,
      "keywords": ["health", "daily routine"],
      "mood_tone": ["motivating", "optimistic"],
      "note": null
    },
    "meta": {
      "file_type": "video/mp4",
      "analysis_timestamp": "2025-01-25T18:00:00Z",
      "analysis_method": "gemini+reka+groq",
      "model_versions": {
        "gemini": "gemini-2.5-flash",
        "reka": "quicktag",
        "whisper": "large-v3"
      }
    },
    "dominant_emotion": "motivating",
    "keyword_count": 5,
    "audio_duration_ratio": 1.6,
    "narrative_coherence_score": 0.75,
    "emotion_variance_score": 0.45
  }
}
```

**Frontend Usage:** 
- Display in dashboard cards
- Show unified analysis (use `unified` field for clean data)
- Audio transcript for subtitles
- Visual captions for frame descriptions
- Reka metrics for performance prediction

**ChromaDB:** Automatically stores audio and visual embeddings for similarity search.

---

## 4. Image Analysis

### POST `/analyze_image`
**Purpose:** Analyze image file using Gemini vision and visual embeddings
**Request:** JSON body
```json
{
  "file_id": "abc123"
}
```

**Response Structure:**
```json
{
  "gemini": {
    "file_name": "image.jpg",
    "duration": 0.0,
    "scene_count": 1,
    "color_tone": null,
    "vision_features": {
      "faces_detected": 1,
      "objects": ["person", "phone"],
      "mood": "casual",
      "color_scheme": "neutral"
    },
    "video_summary": null,
    "error": null
  },
  "reka_features": {
    "note": "Image analysis - Reka not supported for images"
  },
  "visual_embedding": {
    "captions": ["A person using a smartphone"],
    "embedding_dimensions": 768,
    "frame_count": 1,
    "error": null
  },
  "unified": {
    "file_name": "image.jpg",
    "duration": 0.0,
    "scene_count": 1,
    "color_tone": "unknown",
    "summary": "",
    "visual": {
      "faces_detected": 1,
      "objects": ["person", "phone"],
      "mood": "casual",
      "color_scheme": "neutral",
      "cta_present": false
    },
    "audio": {
      "transcript": "",
      "word_count": 0
    },
    "reka": {
      "expected_ctr": null,
      "virality_score": null,
      "keywords": [],
      "mood_tone": [],
      "note": "Image analysis - Reka not supported for images"
    },
    "meta": {
      "file_type": "image/jpeg",
      "analysis_timestamp": "2025-01-25T18:00:00Z",
      "analysis_method": "gemini+reka+groq",
      "model_versions": {
        "gemini": "gemini-2.5-flash",
        "reka": "quicktag",
        "whisper": "large-v3"
      }
    },
    "dominant_emotion": "neutral",
    "keyword_count": 0,
    "audio_duration_ratio": 0.0,
    "narrative_coherence_score": 0.5,
    "emotion_variance_score": 0.0
  }
}
```

**Frontend Usage:** 
- Display image analysis results
- Object detection for tagging
- Visual embeddings stored in ChromaDB

---

## 5. Batch Analysis

### POST `/batch`
**Purpose:** Analyze multiple files in parallel
**Request:** JSON body
```json
{
  "file_ids": [
    "550e8400-e29b-41d4-a716-446655440000",
    "550e8400-e29b-41d4-a716-446655440001"
  ]
}
```

**Response:** Array of analysis results (same format as `/analyze`)

**Frontend Usage:** Process multiple files efficiently for bulk operations.

---

## 6. Chatbot Integration

### POST `/chatbot/index_video`
**Purpose:** Index video for chatbot features (separate from main analysis)
**Request:** Query parameters or body
- `file_id`: Optional
- `file_path`: Optional

**Response:**
```json
{
  "file_name": "sample.mp4",
  "chatbot_indexing": {
    "video_id": "reka_video_123",
    "indexed": true,
    "error": null
  }
}
```

**Frontend Usage:** Enable video Q&A features after main analysis.

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Missing filename"
}
```

### 404 Not Found
```json
{
  "detail": "File not found. Provide file_id from /upload or a valid file_path."
}
```

### 415 Unsupported Media Type
```json
{
  "detail": "Unsupported file type: .txt"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "file_id"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Frontend Integration Tips

### 1. Video Analysis Flow
```
1. Upload file → POST /upload → Get file_id
2. Analyze video → POST /analyze → Get full analysis
3. Display results → Use "unified" field for clean data
4. Store embeddings → Automatically saved to ChromaDB
```

### 2. Key Fields for UI
- **`unified.visual.mood`** - Display as emotion tag
- **`unified.reka.expected_ctr`** - Show CTR prediction badge
- **`unified.reka.virality_score`** - Display virality meter
- **`unified.audio.transcript`** - Show subtitle text
- **`unified.visual.objects`** - Display detected objects chips
- **`unified.meta.analysis_timestamp`** - Show when analyzed

### 3. ChromaDB Search (Future)
- Search similar videos by audio embeddings
- Find videos with similar visual content
- Filter by metadata (duration, color_tone, etc.)

### 4. Supported File Types
**Videos:** `.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`
**Images:** `.jpg`, `.jpeg`, `.png`, `.webp`

---

## API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

