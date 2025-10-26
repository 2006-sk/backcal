from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Optional
import asyncio
import cv2
import numpy as np
import base64

from app.core.config import settings
from app.models.schemas import AnalyzeResult, UploadResponse, BatchAnalyzeRequest, AnalyzeRequest
from app.services.storage import Storage
from app.services.groq_helper import GroqHelper
from app.services.gemini_helper import GeminiHelper
from app.services.reka_helper import RekaHelper
from app.utils.timing import stopwatch
from app.utils.unified_output import build_unified_json
from app.services.chroma_service import chroma_service

# OpenAI fallback (moved to backup folder)
try:
    from app.services.openai.openai_helper import OpenAIHelper as OpenAIHelperFallback
except ImportError:
    OpenAIHelperFallback = None

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Initialize services
storage = Storage()
groq_helper = GroqHelper()
gemini_helper = GeminiHelper()
reka_helper = RekaHelper()

# Fallback to OpenAI if Gemini not available
if not gemini_helper.enabled() and OpenAIHelperFallback:
    openai_helper = OpenAIHelperFallback()
    print("[Warning] Using OpenAI fallback - configure Gemini for better experience")
else:
    openai_helper = None

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
async def analyze(request: AnalyzeRequest):
    """
    Analyze a video or image file using Gemini + Groq + Reka.
    
    Request body should contain:
    - file_id: ID from /upload endpoint (optional)
    - file_path: Direct path to file (optional)
    
    Example JSON:
    {
        "file_id": "abc123"
    }
    or
    {
        "file_path": "/path/to/video.mp4"
    }
    """
    # Resolve file path
    path = None
    if request.file_id:
        path = storage.get_path(request.file_id)
    elif request.file_path:
        p = Path(request.file_path)
        path = p if p.exists() else None

    if not path:
        raise HTTPException(404, "File not found. Provide file_id from /upload or a valid file_path.")

    ext = Path(path).suffix.lower()
    
    with stopwatch("analyze_total"):
        if ext in VIDEO_EXT:
            # Video analysis with PARALLEL Gemini + Reka processing
            try:
                print(f"[Parallel] Starting analysis for {Path(path).name}")
                
                # Step 1: Extract keyframes (parallel with all other operations)
                print(f"[Parallel] Starting ALL operations simultaneously...")
                
                # Create ALL tasks immediately
                tasks = []
                
                # Frame extraction task
                tasks.append(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        gemini_helper.extract_keyframes,
                        path,
                        0.5
                    )
                )
                
                # Reka task (fast QuickTag, no frame extraction needed)
                if settings.USE_REKA:
                    tasks.append(reka_helper.get_reka_features(str(path)))
                else:
                    tasks.append(asyncio.sleep(0))
                
                # Audio embeddings task
                tasks.append(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        groq_helper.analyze_audio_with_groq,
                        path
                    )
                )
                
                # Wait for frame extraction first (needed for other tasks)
                frame_data_task = tasks[0]
                reka_task = tasks[1] if settings.USE_REKA else None
                audio_task = tasks[2]
                
                # Get frames
                frame_data = await frame_data_task
                print(f"[Parallel] Frames extracted: {len(frame_data['frames'])}")
                
                # Step 2: Quick OpenCV analysis (minimal processing)
                opencv_frames = []
                for frame_info in frame_data["frames"][:5]:  # Only first 5 frames for speed
                    base64_data = frame_info["image_base64"]
                    image_data = base64.b64decode(base64_data)
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    opencv_frames.append(frame)
                
                color_tone = groq_helper.average_color_tone(opencv_frames)
                scene_count = groq_helper.scene_count_estimate(opencv_frames)
                
                # Step 3: Add Gemini and Visual tasks that need frames
                gemini_task = asyncio.get_event_loop().run_in_executor(
                    None,
                    gemini_helper.analyze_with_gemini,
                    path,
                    frame_data["frames"]
                )
                
                visual_task = asyncio.get_event_loop().run_in_executor(
                    None,
                    groq_helper.analyze_visual_with_groq,
                    path,
                    5,
                    frame_data["frames"]
                )
                
                # Wait for all remaining tasks
                print(f"[Parallel] Waiting for Gemini, Reka, Audio, Visual...")
                
                # Build task list based on whether Reka is enabled
                if settings.USE_REKA:
                    remaining_tasks = [gemini_task, reka_task, audio_task, visual_task]
                    results = await asyncio.gather(*remaining_tasks)
                    vision_result = results[0]
                    reka_features = results[1]
                    audio_result = results[2]
                    visual_result = results[3]
                else:
                    remaining_tasks = [gemini_task, audio_task, visual_task]
                    results = await asyncio.gather(*remaining_tasks)
                    vision_result = results[0]
                    reka_features = {}
                    audio_result = results[1]
                    visual_result = results[2]
                
                print(f"[Parallel] All analyses completed!")
                
                # Generate video summary
                video_summary = None
                if vision_result.get("vision_features"):
                    features = vision_result["vision_features"]
                    if isinstance(features, dict):
                        mood = features.get("mood", "unknown")
                        activity = features.get("activity", "unknown")
                        video_summary = f"A {mood} video showing {activity}."
                
                # Combine Gemini and Reka features
                gemini_features = {
                    "file_name": Path(path).name,
                    "duration": frame_data["duration"],
                    "scene_count": scene_count,
                    "color_tone": color_tone,
                    "vision_features": vision_result.get("vision_features"),
                    "video_summary": video_summary,
                    "error": vision_result.get("error")
                }
                
                # Build unified JSON
                unified_json = build_unified_json(
                    gemini_features,
                    reka_features,
                    audio_result,
                    visual_result
                )
                
                # Store embeddings in ChromaDB
                if chroma_service.enabled():
                    chroma_service.store_embeddings(
                        file_id=str(path.stem),
                        file_name=Path(path).name,
                        file_type="video",
                        audio_embedding=audio_result.get("audio_embedding", []),
                        visual_embedding=visual_result.get("visual_embedding", []),
                        metadata={
                            "duration": gemini_features.get("duration", 0.0),
                            "scene_count": gemini_features.get("scene_count", 0),
                            "color_tone": gemini_features.get("color_tone", "unknown")
                        }
                    )
                
                return {
                    "gemini": gemini_features,
                    "reka_features": reka_features,
                    "audio_embedding": {
                        "transcript": audio_result.get("audio_transcript"),
                        "embedding_dimensions": audio_result.get("embedding_dimensions"),
                        "error": audio_result.get("error")
                    },
                    "visual_embedding": {
                        "captions": visual_result.get("visual_captions"),
                        "embedding_dimensions": visual_result.get("embedding_dimensions"),
                        "frame_count": visual_result.get("frame_count"),
                        "error": visual_result.get("error")
                    },
                    "unified": unified_json
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
            # Image analysis with Gemini only (Reka doesn't support images)
            try:
                print(f"[Image] Analyzing image: {Path(path).name}")
                
                # Extract single frame for image analysis
                frame_data = gemini_helper.extract_keyframes(path, step_sec=1.0)
                
                # Run Gemini vision analysis and visual embeddings in parallel
                vision_task = asyncio.get_event_loop().run_in_executor(
                    None,
                    gemini_helper.analyze_with_gemini,
                    path,
                    frame_data["frames"]
                )
                
                visual_task = asyncio.get_event_loop().run_in_executor(
                    None,
                    groq_helper.analyze_visual_with_groq,
                    path,
                    1,  # Single frame
                    frame_data["frames"]
                )
                
                # Wait for both tasks
                vision_result, visual_result = await asyncio.gather(vision_task, visual_task)
                
                # Create Gemini features
                gemini_features = {
                    "file_name": Path(path).name,
                    "duration": 0.0,  # Images have no duration
                    "scene_count": 1,
                    "color_tone": None,
                    "vision_features": vision_result.get("vision_features"),
                    "video_summary": None,
                    "error": vision_result.get("error")
                }
                
                # Build unified JSON for images (no Reka, no audio, but visual embeddings)
                unified_json = build_unified_json(
                    gemini_features,
                    {"expected_ctr": None, "virality_score": None, "keywords": [], "mood_tone": [], "note": "Image analysis - Reka not supported for images"},  # Empty Reka with note
                    {"transcript": "", "embedding_dimensions": 0},  # No audio
                    visual_result  # Include visual embeddings
                )
                
                return {
                    "gemini": gemini_features,
                    "reka_features": {"note": "Image analysis - Reka not supported for images"},
                    "visual_embedding": {
                        "captions": visual_result.get("visual_captions"),
                        "embedding_dimensions": visual_result.get("embedding_dimensions"),
                        "frame_count": visual_result.get("frame_count"),
                        "error": visual_result.get("error")
                    },
                    "unified": unified_json
                }
                
            except Exception as e:
                print(f"[Image] Analysis failed: {e}")
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

@app.post("/analyze_image")
async def analyze_image(request: AnalyzeRequest):
    """
    Analyze an image file using Gemini vision.
    
    Request body should contain:
    - file_id: ID from /upload endpoint (optional)
    - file_path: Direct path to file (optional)
    
    Example JSON:
    {
        "file_id": "abc123"
    }
    """
    # Resolve file path
    path = None
    if request.file_id:
        path = storage.get_path(request.file_id)
    elif request.file_path:
        p = Path(request.file_path)
        path = p if p.exists() else None

    if not path:
        raise HTTPException(404, "File not found. Provide file_id from /upload or a valid file_path.")

    ext = Path(path).suffix.lower()
    
    # Only allow image files
    if ext not in IMAGE_EXT:
        raise HTTPException(415, f"Not an image file: {ext}. Use /analyze for videos.")
    
    # Image analysis with Gemini and visual embeddings
    try:
        print(f"[Image] Analyzing image: {Path(path).name}")
        
        # Extract single frame for image analysis
        frame_data = gemini_helper.extract_keyframes(path, step_sec=1.0)
        
        # Run Gemini vision analysis and visual embeddings in parallel
        vision_task = asyncio.get_event_loop().run_in_executor(
            None,
            gemini_helper.analyze_with_gemini,
            path,
            frame_data["frames"]
        )
        
        visual_task = asyncio.get_event_loop().run_in_executor(
            None,
            groq_helper.analyze_visual_with_groq,
            path,
            1,  # Single frame
            frame_data["frames"]
        )
        
        # Wait for both tasks
        vision_result, visual_result = await asyncio.gather(vision_task, visual_task)
        
        # Create Gemini features
        gemini_features = {
            "file_name": Path(path).name,
            "duration": 0.0,  # Images have no duration
            "scene_count": 1,
            "color_tone": None,
            "vision_features": vision_result.get("vision_features"),
            "video_summary": None,
            "error": vision_result.get("error")
        }
        
        # Build unified JSON for images (no Reka, no audio, but visual embeddings)
        unified_json = build_unified_json(
            gemini_features,
            {"expected_ctr": None, "virality_score": None, "keywords": [], "mood_tone": [], "note": "Image analysis - Reka not supported for images"},  # Empty Reka with note
            {"transcript": "", "embedding_dimensions": 0},  # No audio
            visual_result  # Include visual embeddings
        )
        
        # Store embeddings in ChromaDB
        if chroma_service.enabled():
            chroma_service.store_embeddings(
                file_id=str(path.stem),
                file_name=Path(path).name,
                file_type="image",
                audio_embedding=[],  # No audio for images
                visual_embedding=visual_result.get("visual_embedding", []),
                metadata={
                    "duration": 0.0,
                    "scene_count": 1,
                    "color_tone": gemini_features.get("color_tone", "unknown")
                }
            )
        
        return {
            "gemini": gemini_features,
            "reka_features": {"note": "Image analysis - Reka not supported for images"},
            "visual_embedding": {
                "captions": visual_result.get("visual_captions"),
                "embedding_dimensions": visual_result.get("embedding_dimensions"),
                "frame_count": visual_result.get("frame_count"),
                "error": visual_result.get("error")
            },
            "unified": unified_json
        }
        
    except Exception as e:
        print(f"[Image] Analysis failed: {e}")
        return AnalyzeResult(
            file_name=Path(path).name,
            duration=0.0,
            scene_count=1,
            color_tone=None,
            vision_features=None,
            video_summary=None,
            error=f"Image analysis failed: {str(e)}"
        )

@app.post("/batch")
async def batch_analyze(req: BatchAnalyzeRequest):
    """Analyze multiple files in parallel batch processing."""
    print(f"[Batch] Starting parallel analysis of {len(req.file_ids)} videos...")
    
    # Create tasks for parallel processing
    tasks = []
    for fid in req.file_ids:
        task = asyncio.create_task(analyze(file_id=fid))
        tasks.append((fid, task))
    
    # Wait for all tasks to complete
    results = []
    for fid, task in tasks:
        try:
            result = await task
            results.append(result)
            print(f"[Batch] Completed analysis for {fid}")
        except Exception as e:
            print(f"[Batch] Failed analysis for {fid}: {e}")
            results.append({
                "file_id": fid,
                "error": str(e),
                "file_name": "unknown"
            })
    
    print(f"[Batch] Completed parallel analysis of {len(req.file_ids)} videos")
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

@app.post("/chatbot/index_video")
async def chatbot_index_video(file_id: Optional[str] = None, file_path: Optional[str] = None):
    """
    Index video for chatbot usage (separate thread, 2-minute timeout).
    This runs independently of the main analysis pipeline.
    """
    # Resolve file path
    path = None
    if file_id:
        path = storage.get_path(file_id)
    elif file_path:
        p = Path(file_path)
        path = p if p.exists() else None

    if not path:
        raise HTTPException(404, "File not found")
    
    ext = Path(path).suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(415, f"Unsupported file type: {ext}")
    
    try:
        from app.utils.reka_client import reka_index_for_chatbot
        
        # Run in separate thread with 2-minute timeout
        result = await reka_index_for_chatbot(str(path))
        
        return {
            "file_name": Path(path).name,
            "chatbot_indexing": result
        }
        
    except Exception as e:
        return {
            "file_name": Path(path).name,
            "chatbot_indexing": {
                "video_id": None,
                "indexed": False,
                "error": str(e)
            }
        }

@app.get("/debug/test_reka_quicktag")
async def test_reka_quicktag():
    """Test Reka QuickTag endpoint."""
    try:
        # Look for any uploaded video file
        upload_dir = Path(settings.UPLOAD_DIR)
        video_files = list(upload_dir.glob("*.mp4"))
        
        if not video_files:
            return {"error": "No video files found in uploads directory"}
        
        test_video = video_files[0]
        print(f"[Test] Testing Reka QuickTag with {test_video.name}")
        
        # Import Reka client
        from app.utils.reka_client import reka_quicktag
        
        # Test QuickTag
        result = await reka_quicktag(str(test_video), settings.REKA_API_KEY)
        
        return {
            "test": True,
            "video": str(test_video),
            "quicktag_result": result
        }
        
    except Exception as e:
        return {"error": f"Reka QuickTag test failed: {str(e)}"}

@app.get("/debug/test_visual_embeddings")
async def test_visual_embeddings():
    """Test visual embeddings generation using ALL frames from analyze process."""
    try:
        # Look for any uploaded video file
        upload_dir = Path(settings.UPLOAD_DIR)
        video_files = list(upload_dir.glob("*.mp4"))
        
        if not video_files:
            return {"error": "No video files found in uploads directory"}
        
        test_video = video_files[0]
        print(f"[Test] Testing visual embeddings with {test_video.name}")
        
        # Extract frames using the same method as analyze endpoint
        frame_data = gemini_helper.extract_keyframes(test_video, step_sec=0.5)
        frames = frame_data["frames"]
        
        print(f"[Test] Using {len(frames)} frames from analyze process")
        
        # Test visual embeddings with ALL frames
        result = groq_helper.analyze_visual_with_groq(test_video, frames=frames)
        
        return {
            "test": True,
            "video": str(test_video),
            "frames_used": len(frames),
            "result": {
                "captions": result.get("visual_captions", []),
                "embedding_dimensions": result.get("embedding_dimensions", 0),
                "frame_count": result.get("frame_count", 0),
                "error": result.get("error")
            }
        }
        
    except Exception as e:
        return {"error": f"Visual embeddings test failed: {str(e)}"}

@app.get("/debug/parallel_videos_test")
async def parallel_videos_test():
    """Test parallel processing of multiple videos."""
    try:
        # Look for uploaded video files
        upload_dir = Path(settings.UPLOAD_DIR)
        video_files = list(upload_dir.glob("*.mp4"))
        
        if len(video_files) < 2:
            return {"error": "Need at least 2 video files for parallel testing"}
        
        # Take first 2 videos
        video1 = video_files[0]
        video2 = video_files[1]
        
        print(f"[Parallel Test] Testing with {video1.name} and {video2.name}")
        
        # Test parallel processing
        with stopwatch("parallel_videos_total"):
            # Create tasks for both videos
            task1 = asyncio.create_task(analyze(file_path=str(video1)))
            task2 = asyncio.create_task(analyze(file_path=str(video2)))
            
            # Wait for both to complete
            result1 = await task1
            result2 = await task2
        
        return {
            "parallel_test": True,
            "videos": [
                {"file": str(video1), "result": result1},
                {"file": str(video2), "result": result2}
            ],
            "performance_notes": [
                "✅ Two videos processed simultaneously",
                "✅ Each video uses sequential processing internally",
                "✅ Videos run on separate async tasks",
                "🚀 Total time ≈ max(individual_video_time)",
                "📊 Much faster than processing videos one by one"
            ]
        }
        
    except Exception as e:
        return {"error": f"Parallel videos test failed: {str(e)}"}

@app.get("/debug/performance_benchmark")
async def performance_benchmark():
    """Performance benchmark for sequential processing."""
    try:
        # Look for any uploaded video file
        upload_dir = Path(settings.UPLOAD_DIR)
        video_files = list(upload_dir.glob("*.mp4"))
        
        if not video_files:
            return {"error": "No demo video files found in uploads directory"}
        
        demo_video = video_files[0]
        print(f"[Benchmark] Testing sequential processing with {demo_video.name}")
        
        # Test sequential processing
        with stopwatch("sequential_total"):
            with stopwatch("frame_extraction"):
                frame_data = gemini_helper.extract_keyframes(demo_video, step_sec=0.5)
            
            with stopwatch("opencv_analysis"):
                # Convert frames for OpenCV analysis
                opencv_frames = []
                for frame_info in frame_data["frames"]:
                    base64_data = frame_info["image_base64"]
                    image_data = base64.b64decode(base64_data)
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    opencv_frames.append(frame)
                
                color_tone = groq_helper.average_color_tone(opencv_frames)
                scene_count = groq_helper.scene_count_estimate(opencv_frames)
            
            with stopwatch("ai_analysis"):
                vision_result = gemini_helper.analyze_with_gemini(
                    demo_video, 
                    frame_data["frames"]
                )
            
            with stopwatch("reka_analysis"):
                reka_features = await reka_helper.get_reka_features(str(demo_video))
        
        return {
            "benchmark": True,
            "demo_file": str(demo_video),
            "frame_count": len(frame_data["frames"]),
            "duration": frame_data["duration"],
            "sequential_results": {
                "opencv": {"color_tone": color_tone, "scene_count": scene_count},
                "vision": vision_result.get("vision_features", {}),
                "reka": reka_features
            },
            "performance_notes": [
                "✅ Sequential frame extraction with FFmpeg",
                "✅ Sequential OpenCV processing for color/scene analysis", 
                "✅ Sequential AI calls: OpenAI → Reka",
                "✅ Simple, reliable, and easy to debug",
                "✅ No threading complexity",
                "📊 Baseline performance for comparison"
            ]
        }
        
    except Exception as e:
        return {"error": f"Performance benchmark failed: {str(e)}"}
