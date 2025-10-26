"""
Audio embeddings generator for AdAtlas AI.
Extracts audio from video, transcribes with Groq Whisper, and generates embeddings with OpenAI.
"""
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from app.core.config import settings


def extract_audio(video_path: str | Path) -> str:
    """
    Extract audio track from video file using ffmpeg.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Path to extracted audio file (temporary WAV file)
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create temporary audio file
    temp_audio = tempfile.NamedTemporaryFile(
        suffix='.wav',
        delete=False
    )
    temp_audio_path = temp_audio.name
    temp_audio.close()
    
    print(f"[Audio] Extracting audio from {video_path.name}...")
    
    # Extract audio using ffmpeg
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # WAV format
        '-ar', '16000',  # 16kHz sample rate (optimal for Whisper)
        '-ac', '1',  # Mono
        '-y',  # Overwrite output
        temp_audio_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[Audio] Audio extracted successfully: {temp_audio_path}")
        return temp_audio_path
    except subprocess.CalledProcessError as e:
        # Clean up on error
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        raise Exception(f"FFmpeg audio extraction failed: {e.stderr}")


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio using Groq's Whisper API.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Transcript text
    """
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not configured")
    
    print(f"[Audio] Transcribing audio with Groq Whisper...")
    
    # Groq Whisper endpoint (OpenAI-compatible)
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}"
    }
    
    # Read audio file
    with open(audio_path, 'rb') as audio_file:
        files = {
            'file': (os.path.basename(audio_path), audio_file, 'audio/wav')
        }
        data = {
            'model': 'whisper-large-v3'
        }
        
        try:
            response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            transcript = result.get('text', '')
            
            print(f"[Audio] Transcription complete: {len(transcript)} characters")
            return transcript
            
        except requests.exceptions.SSLError as e:
            print(f"[Audio] SSL Error: {e}")
            print("[Audio] Retrying with SSL verification disabled...")
            # Retry without SSL verification as fallback
            try:
                response = requests.post(url, headers=headers, files=files, data=data, timeout=60, verify=False)
                response.raise_for_status()
                result = response.json()
                transcript = result.get('text', '')
                print(f"[Audio] Transcription complete after retry: {len(transcript)} characters")
                return transcript
            except Exception as retry_e:
                raise Exception(f"Groq transcription failed after retry: {str(retry_e)}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Groq transcription failed: {str(e)}")


def create_audio_embedding(transcript: str) -> list[float]:
    """
    Generate embedding vector from transcript using OpenAI.
    Returns zero vector if OpenAI unavailable.
    
    Args:
        transcript: Text transcript
        
    Returns:
        Embedding vector (list of floats) or zero vector
    """
    print(f"[Audio] Generating embedding for transcript...")
    
    # Use Gemini embeddings API
    if not settings.USE_GEMINI or not settings.GEMINI_API_KEY:
        print("[Audio] Gemini not configured - using zero vector")
        return [0.0] * 768  # Gemini embeddings dimension
    
    import google.generativeai as genai
    genai.configure(api_key=settings.GEMINI_API_KEY)
    
    try:
        result = genai.embed_content(
            model='models/text-embedding-004',
            content=transcript,
            task_type='retrieval_document'
        )
        embedding = result['embedding']
        print(f"[Audio] Embedding generated: {len(embedding)} dimensions")
        return embedding
    except Exception as e:
        print(f"[Audio] Gemini embedding failed: {e}")
        return [0.0] * 768  # Fallback to zero vector


def get_audio_embeddings(video_path: str | Path) -> Dict[str, Any]:
    """
    Complete pipeline: extract audio, transcribe, and generate embeddings.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with transcript and embedding
    """
    audio_path = None
    
    try:
        # Step 1: Extract audio
        audio_path = extract_audio(video_path)
        
        # Step 2: Transcribe audio
        transcript = transcribe_audio(audio_path)
        
        # Step 3: Generate embedding
        embedding = create_audio_embedding(transcript)
        
        return {
            "transcript": transcript,
            "embedding": embedding
        }
        
    finally:
        # Clean up temporary audio file
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
            print(f"[Audio] Cleaned up temporary audio file")


if __name__ == "__main__":
    # Test with sample video if available
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Look for any video in uploads
        from pathlib import Path
        upload_dir = Path("data/uploads")
        videos = list(upload_dir.glob("*.mp4"))
        
        if not videos:
            print("No test videos found. Usage: python audio_embeddings.py <video_path>")
            sys.exit(1)
        
        video_path = videos[0]
    
    print(f"\n🎵 Testing audio embeddings with: {video_path}\n")
    
    try:
        result = get_audio_embeddings(video_path)
        
        print("\n✅ Result:")
        print(f"Transcript (first 100 chars): {result['transcript'][:100]}...")
        print(f"Embedding dimensions: {len(result['embedding'])}")
        print(f"Sample embedding (first 5 values): {result['embedding'][:5]}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

