# AdAtlas API Response Format

## POST /analyze

### Request Body (JSON)
```json
{
  "file_id": "abc123-def456-ghi789",
  "file_path": null
}
```

or

```json
{
  "file_id": null,
  "file_path": "/path/to/video.mp4"
}
```

### Response Format

```json
{
  "gemini": {
    "file_name": "video.mp4",
    "duration": 15.5,
    "scene_count": 3,
    "color_tone": "warm",
    "vision_features": {
      "faces_detected": 1,
      "objects": ["Amazon package", "knife block set", "vegetables"],
      "text_present": true,
      "scene_description": "Video shows unboxing and food preparation",
      "mood": "creative, promotional",
      "color_scheme": "warm tones with vibrant vegetables",
      "activity": "unboxing → food prep → artistic plating",
      "motion_style": "smooth transitions",
      "cta_present": false,
      "notes": "Educational cooking content"
    },
    "video_summary": "A creative video showing unboxing → food prep → artistic plating.",
    "error": null
  },
  "reka_features": {
    "expected_ctr": 1.5,
    "virality_score": 65,
    "keywords": ["stress", "wellness", "brand products", "female individuals"],
    "mood_tone": ["happy", "relaxing", "relief", "relaxed"],
    "raw_data": { /* Reka API response */ },
    "method": "quicktag",
    "error": null
  },
  "audio_embedding": {
    "transcript": "Welcome to this cooking tutorial...",
    "embedding_dimensions": 768,
    "error": null
  },
  "visual_embedding": {
    "captions": [
      "A woman unboxes an Amazon package",
      "Kitchen counter with colorful vegetables",
      "Artistic food plating on wooden table"
    ],
    "embedding_dimensions": 768,
    "frame_count": 12,
    "error": null
  },
  "unified": {
    /* Combines all above fields */
    "file_name": "video.mp4",
    "duration": 15.5,
    "scene_count": 3,
    "color_tone": "warm",
    "vision_features": { /* ... */ },
    "video_summary": "A creative video...",
    "reka": { /* Reka features */ },
    "audio": { /* Audio results */ },
    "visual": { /* Visual results */ }
  }
}
```

### Fields Description

#### `gemini` (Gemini vision analysis)
- `file_name`: Name of the analyzed file
- `duration`: Video duration in seconds
- `scene_count`: Estimated number of scenes
- `color_tone`: "warm" or "cool"
- `vision_features`: Detailed vision analysis
  - `faces_detected`: Number of faces
  - `objects`: List of detected objects
  - `text_present`: Boolean
  - `scene_description`: Overall scene description
  - `mood`: Detected mood
  - `color_scheme`: Color analysis
  - `activity`: Sequence of activities
  - `motion_style`: Motion characteristics
  - `cta_present`: Call-to-action detected
  - `notes`: Additional insights
- `video_summary`: High-level summary
- `error`: Error message if failed

#### `reka_features` (Reka QuickTag)
- `expected_ctr`: Expected click-through rate
- `virality_score`: Virality prediction score
- `keywords`: Extracted keywords
- `mood_tone`: Detected moods
- `raw_data`: Complete Reka response
- `method`: "quicktag" or "upload_chat"
- `error`: Error message if failed

#### `audio_embedding`
- `transcript`: Audio transcription
- `embedding_dimensions`: 768 (Gemini embeddings)
- `error`: Error message if failed

#### `visual_embedding`
- `captions`: Frame captions (40% sampling)
- `embedding_dimensions`: 768 (Gemini embeddings)
- `frame_count`: Number of frames analyzed
- `error`: Error message if failed

#### `unified`
- Combines all analysis results into one object

