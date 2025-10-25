from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import base64

from app.core.config import settings
from app.models.schemas import AnalyzeResult, UploadResponse, BatchAnalyzeRequest
from app.services.storage import Storage
from app.services.groq_helper import GroqHelper
from app.services.openai_helper import OpenAIHelper
from app.services.reka_helper import RekaHelper
from app.utils.timing import stopwatch

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Initialize services
storage = Storage()
groq_helper = GroqHelper()
openai_helper = OpenAIHelper()
reka_helper = RekaHelper()

# File extensions
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Health endpoints
@app.get("/")
def root_health():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/health/live")
def liveness():
    return {"live": True}

@app.get("/health/ready")
def readiness():
    return {"ready": True}

# Core API endpoints
@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """Upload a video or image file for analysis."""
    if not file.filename:
        raise HTTPException(400, "Missing filename")
    
    fid, path = storage.save_upload(file.file, file.filename)
    return UploadResponse(id=fid, file_name=Path(path).name, saved_path=str(path))

@app.post("/analyze")
async def analyze(file_id: Optional[str] = None, file_path: Optional[str] = None):
    """
    Analyze a video or image file using Groq + ffmpeg.
    
    Args:
        file_id: ID from /upload endpoint
        file_path: Direct path to file
    """
    # Resolve file path
    path = None
    if file_id:
        path = storage.get_path(file_id)
    elif file_path:
        p = Path(file_path)
        path = p if p.exists() else None

    if not path:
        raise HTTPException(404, "File not found. Provide file_id from /upload or a valid file_path.")

    ext = Path(path).suffix.lower()
    
    with stopwatch("analyze_total"):
        if ext in VIDEO_EXT:
            # Video analysis with OpenAI + ffmpeg + OpenCV
            try:
                # Extract keyframes using ffmpeg
                frame_data = openai_helper.extract_keyframes(path, step_sec=0.5)
                
                # Get OpenCV analysis for color tone and scene count
                opencv_frames = []
                for frame_info in frame_data["frames"]:
                    base64_data = frame_info["image_base64"]
                    image_data = base64.b64decode(base64_data)
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    opencv_frames.append(frame)
                
                color_tone = groq_helper.average_color_tone(opencv_frames)
                scene_count = groq_helper.scene_count_estimate(opencv_frames)
                
                # OpenAI vision analysis
                vision_result = openai_helper.analyze_with_openai(
                    path, 
                    frame_data["frames"], 
                    use_reka=settings.USE_REKA
                )
                
                # Generate video summary
                video_summary = None
                if vision_result.get("vision_features"):
                    features = vision_result["vision_features"]
                    if isinstance(features, dict):
                        mood = features.get("mood", "unknown")
                        activity = features.get("activity", "unknown")
                        video_summary = f"A {mood} video showing {activity}."
                
                # Get Reka features
                reka_features = await reka_helper.get_reka_features(str(path))
                
                # Combine OpenAI and Reka features
                openai_features = {
                    "file_name": Path(path).name,
                    "duration": frame_data["duration"],
                    "scene_count": scene_count,
                    "color_tone": color_tone,
                    "vision_features": vision_result.get("vision_features"),
                    "video_summary": video_summary,
                    "error": vision_result.get("error")
                }
                
                return {
                    "openai": openai_features,
                    "reka_features": reka_features,
                    "unified": {**openai_features, **{"reka": reka_features}}
                }
                
            except Exception as e:
                return AnalyzeResult(
                    file_name=Path(path).name,
                    duration=0.0,
                    scene_count=0,
                    color_tone="unknown",
                    vision_features=None,
                    video_summary=None,
                    error=f"Video analysis failed: {str(e)}"
                )
                
        elif ext in IMAGE_EXT:
            # Image analysis with OpenAI
            try:
                # Extract single frame for image analysis
                frame_data = openai_helper.extract_keyframes(path, step_sec=1.0)
                
                vision_result = openai_helper.analyze_with_openai(
                    path, 
                    frame_data["frames"], 
                    use_reka=False  # Images don't need Reka
                )
                
                # Get Reka features for images too
                reka_features = await reka_helper.get_reka_features(str(path))
                
                # Combine OpenAI and Reka features
                openai_features = {
                    "file_name": Path(path).name,
                    "duration": 0.0,  # Images have no duration
                    "scene_count": 1,
                    "color_tone": None,
                    "vision_features": vision_result.get("vision_features"),
                    "video_summary": None,
                    "error": vision_result.get("error")
                }
                
                return {
                    "openai": openai_features,
                    "reka_features": reka_features,
                    "unified": {**openai_features, **{"reka": reka_features}}
                }
                
            except Exception as e:
                return AnalyzeResult(
                    file_name=Path(path).name,
                    duration=0.0,
                    scene_count=1,
                    color_tone=None,
                    vision_features=None,
                    video_summary=None,
                    error=f"Image analysis failed: {str(e)}"
                )
        else:
            raise HTTPException(415, f"Unsupported file type: {ext}")

@app.post("/batch")
async def batch_analyze(req: BatchAnalyzeRequest):
    """Analyze multiple files in batch."""
    results = []
    for fid in req.file_ids:
        try:
            result = await analyze(file_id=fid)
            results.append(result)
        except Exception as e:
            results.append({
                "file_id": fid,
                "error": str(e),
                "file_name": "unknown"
            })
    return JSONResponse(results)

@app.get("/debug/analyze_demo")
async def debug_analyze_demo():
    """Debug route to test analysis with a demo video."""
    try:
        # Look for any uploaded video file
        upload_dir = Path(settings.UPLOAD_DIR)
        video_files = list(upload_dir.glob("*.mp4"))
        
        if not video_files:
            return {"error": "No demo video files found in uploads directory"}
        
        # Use the first video file found
        demo_video = video_files[0]
        
        # Run analysis
        result = await analyze(file_path=str(demo_video))
        
        return {
            "debug": True,
            "demo_file": str(demo_video),
            "analysis_result": result
        }
        
    except Exception as e:
        return {"error": f"Debug analysis failed: {str(e)}"}
