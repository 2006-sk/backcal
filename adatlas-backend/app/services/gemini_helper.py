"""
Gemini helper for vision analysis and captioning.
Uses Gemini-1.5-Flash for all image/video analysis.
"""
import base64
import io
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import ffmpeg
from PIL import Image
import google.generativeai as genai
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings


class GeminiHelper:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.use_gemini = settings.USE_GEMINI and bool(self.api_key)
        
        if self.use_gemini:
            genai.configure(api_key=self.api_key)
            model_name = settings.GEMINI_MODEL
            self.model = genai.GenerativeModel(model_name)
            print(f"[Gemini] Using model: {model_name}")
    
    def enabled(self) -> bool:
        return self.use_gemini
    
    def _extract_image_frame(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract single frame from image (treat image as single-frame video).
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with single frame data
        """
        try:
            from PIL import Image
            import base64
            import io
            
            # Read image
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to base64 JPEG
            jpeg_buffer = io.BytesIO()
            img.save(jpeg_buffer, format='JPEG', quality=85)
            jpeg_data = jpeg_buffer.getvalue()
            base64_image = base64.b64encode(jpeg_data).decode('utf-8')
            
            frame = {
                "timestamp": 0.0,
                "image_base64": base64_image,
                "width": img.width,
                "height": img.height
            }
            
            return {
                "duration": 0.0,
                "fps": 1.0,
                "frames": [frame],
                "total_frames": 1
            }
            
        except Exception as e:
            raise Exception(f"Image extraction failed: {str(e)}")
    
    def extract_keyframes(self, video_path: Path, step_sec: float = 0.5) -> Dict[str, Any]:
        """
        Extract keyframes from video or image using ffmpeg at specified intervals.
        Returns frame data with base64 encoded images and timestamps.
        """
        try:
            # Check if it's an image (not a video)
            if video_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                return self._extract_image_frame(video_path)
            
            # Get video metadata
            probe = ffmpeg.probe(str(video_path))
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            duration = float(probe['format']['duration'])
            fps = eval(video_stream['r_frame_rate'])

            frames = []
            current_time = 0.0

            while current_time < duration:
                try:
                    # Extract frame at current_time
                    out, _ = (
                        ffmpeg
                        .input(str(video_path), ss=current_time)
                        .output('pipe:', vframes=1, format='image2', vcodec='png', pix_fmt='rgb24')
                        .run(capture_stdout=True, quiet=True)
                    )

                    # Convert to base64 JPEG
                    try:
                        # Ensure we have valid image data
                        if not out or len(out) == 0:
                            print(f"Warning: Empty frame data at {current_time:.1f}s, skipping...")
                            current_time += step_sec
                            continue
                            
                        image = Image.open(io.BytesIO(out))
                        if image.mode != 'RGB':
                            image = image.convert('RGB')

                        jpeg_buffer = io.BytesIO()
                        image.save(jpeg_buffer, format='JPEG', quality=85)
                        jpeg_data = jpeg_buffer.getvalue()
                        base64_image = base64.b64encode(jpeg_data).decode('utf-8')

                        frames.append({
                            "timestamp": current_time,
                            "image_base64": base64_image,
                            "width": image.width,
                            "height": image.height
                        })
                    except Exception as img_error:
                        print(f"Warning: Could not process frame at {current_time:.1f}s: {img_error}")
                        # Continue to next frame instead of breaking

                except Exception as frame_error:
                    print(f"Warning: Could not extract frame at {current_time:.1f}s: {frame_error}")
                    # Continue to next frame instead of breaking

                current_time += step_sec

            if not frames:
                raise Exception("No frames could be extracted from the video")

            return {
                "duration": duration,
                "fps": fps,
                "frames": frames,
                "total_frames": len(frames)
            }

        except Exception as e:
            raise Exception(f"Frame extraction failed: {str(e)}")
    
    def analyze_with_gemini(self, video_path: Path, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze ALL frames with Gemini for comprehensive insights.
        Uses batch processing for speed.
        """
        if not self.use_gemini:
            return {
                "error": "Gemini not enabled. Set USE_GEMINI=true and GEMINI_API_KEY in environment.",
                "vision_features": None
            }
        
        try:
            print(f"[Gemini] Analyzing {len(frames)} frames for insights...")
            
            # Convert frames to PIL Images in parallel
            def decode_frame(frame):
                base64_data = frame["image_base64"]
                image_data = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(image_data))
            
            print(f"[Gemini] Decoding {len(frames)} frames...")
            with ThreadPoolExecutor(max_workers=8) as executor:
                images = list(executor.map(decode_frame, frames))
            
            # Create prompt for insights
            prompt = """Analyze all these video frames together and return ONLY valid JSON.

These frames represent different moments throughout the video timeline.

Use this schema:
{
"faces_detected": int,
"objects": [string],
"text_present": bool,
"scene_description": string,
"mood": string,
"color_scheme": string,
"activity": string,
"motion_style": string,
"cta_present": bool,
"notes": string
}

IMPORTANT: 
- Look for temporal changes across frames (beginning vs middle vs end)
- If multiple activities occur (e.g., unboxing → cooking → text display), describe the sequence
- Check for text/banners that appear in later frames
- Consider the overall narrative flow of the video
- Be specific about what you see in different parts of the video"""
            
            # Call Gemini with all images in batch (Gemini handles this efficiently)
            print(f"[Gemini] Sending {len(images)} images to Gemini in single batch...")
            response = self.model.generate_content([prompt] + images)
            
            # Parse response
            text = response.text.strip()
            
            # Extract JSON
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception as e:
                    print(f"[Gemini] JSON parsing failed: {e}")
                    parsed = {"raw_text": text}
            else:
                parsed = {"raw_text": text}
            
            print(f"[Gemini] Analysis complete!")
            
            return {
                "vision_features": parsed
            }
            
        except Exception as e:
            print(f"[Gemini] Analysis failed: {e}")
            return {
                "error": str(e),
                "vision_features": None
            }

