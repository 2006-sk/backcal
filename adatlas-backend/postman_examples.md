# AdAtlas AI FastAPI - Postman Examples

## Environment Setup

### Base URL
```
http://localhost:8000
```

### Environment Variables
Set these in Postman environment:
- `GROQ_API_KEY`: Your Groq API key (optional for testing)
- `USE_GROQ`: Set to `true` to enable Groq analysis

## 1. Health Check

### GET /health
**URL:** `{{base_url}}/health`

**Expected Response:**
```json
{
  "status": "ok"
}
```

## 2. Upload Video/Image

### POST /upload
**URL:** `{{base_url}}/upload`
**Method:** POST
**Content-Type:** multipart/form-data

**Body (form-data):**
- Key: `file`
- Type: File
- Value: Select your video/image file (e.g., `sample.mp4`, `image.jpg`)

**Expected Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "file_name": "sample.mp4",
  "saved_path": "/path/to/data/uploads/550e8400-e29b-41d4-a716-446655440000__sample.mp4"
}
```

## 3. Analyze Video

### POST /analyze
**URL:** `{{base_url}}/analyze`
**Method:** POST
**Content-Type:** application/json

**Body (JSON):**
```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Alternative with file_path:**
```json
{
  "file_path": "/path/to/your/video.mp4"
}
```

**Expected Response:**
```json
{
  "file_name": "sample.mp4",
  "duration": 9.2,
  "fps": 30.0,
  "total_frames": 18,
  "color_tone": "warm",
  "scene_count": 6,
  "groq_raw_output": {
    "model": "llama-3.2-vision",
    "frames_analyzed": 18,
    "analysis_placeholder": "Groq vision analysis will be implemented here",
    "frame_data": [
      {
        "timestamp": 0.0,
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
        "width": 320,
        "height": 240
      }
    ]
  },
  "error": null
}
```

## 4. Analyze Image

### POST /analyze
**URL:** `{{base_url}}/analyze`
**Method:** POST
**Content-Type:** application/json

**Body (JSON):**
```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

**Expected Response:**
```json
{
  "file_name": "image.jpg",
  "duration": 0.0,
  "fps": null,
  "total_frames": 1,
  "color_tone": "cool",
  "scene_count": 1,
  "groq_raw_output": {
    "model": "llama-3.2-vision",
    "image_analyzed": true,
    "analysis_placeholder": "Groq vision analysis will be implemented here"
  },
  "error": null
}
```

## 5. Batch Analyze

### POST /batch
**URL:** `{{base_url}}/batch`
**Method:** POST
**Content-Type:** application/json

**Body (JSON):**
```json
{
  "file_ids": [
    "550e8400-e29b-41d4-a716-446655440000",
    "550e8400-e29b-41d4-a716-446655440001"
  ]
}
```

**Expected Response:**
```json
[
  {
    "file_name": "sample.mp4",
    "duration": 9.2,
    "fps": 30.0,
    "total_frames": 18,
    "color_tone": "warm",
    "scene_count": 6,
    "groq_raw_output": { ... },
    "error": null
  },
  {
    "file_name": "image.jpg",
    "duration": 0.0,
    "fps": null,
    "total_frames": 1,
    "color_tone": "cool",
    "scene_count": 1,
    "groq_raw_output": { ... },
    "error": null
  }
]
```

## Error Responses

### File Not Found (404)
```json
{
  "detail": "File not found. Provide file_id from /upload or a valid file_path."
}
```

### Unsupported File Type (415)
```json
{
  "detail": "Unsupported file type: .txt"
}
```

### Missing Filename (400)
```json
{
  "detail": "Missing filename"
}
```

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## Groq API Key Setup

1. Get your Groq API key from: https://console.groq.com/
2. Set environment variable: `export GROQ_API_KEY=your_key_here`
3. Set: `export USE_GROQ=true`
4. Restart the server

## Supported File Types

**Videos:** `.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`
**Images:** `.jpg`, `.jpeg`, `.png`, `.webp`
