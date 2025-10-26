"""
Visual embeddings generator for AdAtlas AI.
Extracts frames from video or images, generates captions with GPT-4o vision, and creates embeddings.
"""
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Union
import requests
import base64
import ssl
import time
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from app.core.config import settings

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def extract_frames(video_path: Union[str, Path], frame_count: int = 5) -> List[str]:
    """
    Extract frames from video file using ffmpeg.
    
    Args:
        video_path: Path to video file
        frame_count: Number of frames to extract
        
    Returns:
        List of paths to extracted frame images (temporary files)
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"[Visual] Extracting {frame_count} frames from {video_path.name}...")
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frame_paths = []
    
    try:
        import subprocess
        
        # Get video duration
        import ffmpeg
        probe = ffmpeg.probe(str(video_path))
        duration = float(probe['format']['duration'])
        
        # Extract frames evenly spaced
        frame_indices = frame_count
        interval = duration / (frame_indices + 1)
        
        for i in range(frame_indices):
            timestamp = interval * (i + 1)
            frame_path = os.path.join(temp_dir, f"frame_{i:03d}.jpg")
            
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-ss', str(timestamp),
                '-vframes', '1',
                '-q:v', '2',  # High quality
                '-y',
                frame_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if os.path.exists(frame_path):
                frame_paths.append(frame_path)
        
        print(f"[Visual] Extracted {len(frame_paths)} frames")
        return frame_paths
        
    except Exception as e:
        # Clean up on error
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                os.unlink(frame_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        raise Exception(f"Frame extraction failed: {e}")


def generate_captions(image_paths: List[str]) -> List[str]:
    """
    Generate captions for images using Gemini with parallel processing.
    
    Args:
        image_paths: List of paths to image files
        
    Returns:
        List of caption strings
    """
    # Check Gemini is configured
    if not settings.USE_GEMINI or not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not configured")
    
    print(f"[Visual] Using Gemini for {len(image_paths)} image captions (parallel)...")
    import google.generativeai as genai
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(settings.GEMINI_MODEL)
    
    def generate_single_caption(image_path: str) -> str:
        """Generate caption for a single image."""
        try:
            import PIL.Image
            img = PIL.Image.open(image_path)
            
            response = model.generate_content([
                "Describe this image in one sentence. Focus on key objects, actions, and text.",
                img
            ])
            
            return response.text.strip()
        except Exception as e:
            print(f"[Visual] Caption failed for {image_path}: {e}")
            return "Failed to generate caption"
    
    # Process in parallel with max 8 workers (Gemini handles this well)
    captions = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(generate_single_caption, path): path for path in image_paths}
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_path)):
            caption = future.result()
            captions.append(caption)
            print(f"[Visual] Caption {i+1}/{len(image_paths)} complete")
    
    print(f"[Visual] Generated {len(captions)} captions with Gemini (parallel)")
    return captions


def create_visual_embedding(captions: List[str]) -> List[float]:
    """
    Generate embedding by averaging individual caption embeddings.
    Currently requires OpenAI (embeddings not supported by Gemini/Groq).
    
    Args:
        captions: List of caption strings
        
    Returns:
        Averaged embedding vector (or zeros if OpenAI unavailable)
    """
    if not captions:
        raise ValueError("No captions provided")
    
    print(f"[Visual] Generating embeddings for {len(captions)} captions...")
    
    # Use Gemini embeddings API
    if not settings.USE_GEMINI or not settings.GEMINI_API_KEY:
        print("[Visual] Gemini not configured - using zero vector")
        return [0.0] * 768  # Gemini embeddings dimension
    
    import google.generativeai as genai
    genai.configure(api_key=settings.GEMINI_API_KEY)
    
    # Generate embeddings for all captions
    embeddings = []
    for i, caption in enumerate(captions):
        try:
            result = genai.embed_content(
                model='models/text-embedding-004',
                content=caption,
                task_type='retrieval_document'
            )
            embedding = result['embedding']
            embeddings.append(embedding)
            print(f"[Visual] Generated embedding {i+1}/{len(captions)}")
        except Exception as e:
            print(f"[Visual] Failed to generate embedding {i+1}: {e}")
            embeddings.append([0.0] * 768)  # Fallback to zero vector
    
    if not embeddings:
        return [0.0] * 768
    
    # Average embeddings
    import numpy as np
    avg_embedding = np.mean(embeddings, axis=0).tolist()
    print(f"[Visual] Generated averaged embedding: {len(avg_embedding)} dimensions")
    return avg_embedding


def get_visual_embeddings(media_path: Union[str, Path], frame_count: int = 5, frames: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Complete pipeline: extract frames/images, generate captions, and create embeddings.
    
    Args:
        media_path: Path to video or image file
        frame_count: Number of frames to extract (for videos only, if frames not provided)
        frames: Pre-extracted frames from gemini_helper (optional)
        
    Returns:
        Dictionary with captions and embedding
    """
    media_path = Path(media_path)
    temp_files = []
    
    try:
        # Handle images vs videos
        if media_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
            print(f"[Visual] Processing image: {media_path.name}")
            image_paths = [str(media_path)]
        
        elif media_path.suffix.lower() in ['.mp4', '.mov', '.mkv', '.avi', '.m4v']:
            print(f"[Visual] Processing video: {media_path.name}")
            
            # Use pre-extracted frames if available
            if frames:
                # Use ALL frames for captions (no sampling)
                sampled_frames = frames
                
                print(f"[Visual] Using ALL {len(sampled_frames)} frames for captions")
                
                # Convert base64 frames to temporary image files
                for i, frame_data in enumerate(sampled_frames):
                    base64_image = frame_data.get("image_base64")
                    if base64_image:
                        # Create temporary file
                        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                        temp_file.write(base64.b64decode(base64_image))
                        temp_file.close()
                        temp_files.append(temp_file.name)
                
                image_paths = temp_files
            else:
                # Extract frames (these will be cleaned up later)
                temp_files = extract_frames(media_path, frame_count)
                image_paths = temp_files
        
        else:
            raise ValueError(f"Unsupported file type: {media_path.suffix}")
        
        # Generate captions
        captions = generate_captions(image_paths)
        
        # Create embedding
        embedding = create_visual_embedding(captions)
        
        return {
            "captions": captions,
            "embedding": embedding
        }
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Clean up temp directory if it exists
        if temp_files:
            temp_dir = os.path.dirname(temp_files[0])
            if os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass
        
        print(f"[Visual] Cleaned up temporary files")


if __name__ == "__main__":
    # Test with sample media
    import sys
    
    if len(sys.argv) > 1:
        media_path = sys.argv[1]
    else:
        # Look for any media in uploads
        from pathlib import Path
        upload_dir = Path("data/uploads")
        videos = list(upload_dir.glob("*.mp4"))
        images = list(upload_dir.glob("*.jpg")) + list(upload_dir.glob("*.png"))
        
        if videos:
            media_path = videos[0]
        elif images:
            media_path = images[0]
        else:
            print("No test media found. Usage: python visual_embeddings.py <media_path>")
            sys.exit(1)
    
    print(f"\n👁️  Testing visual embeddings with: {media_path}\n")
    
    try:
        result = get_visual_embeddings(media_path, frame_count=5)
        
        print("\n✅ Result:")
        print(f"Captions ({len(result['captions'])}):")
        for i, caption in enumerate(result['captions'], 1):
            print(f"  {i}. {caption}")
        print(f"\nEmbedding dimensions: {len(result['embedding'])}")
        print(f"Sample embedding (first 5 values): {result['embedding'][:5]}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

