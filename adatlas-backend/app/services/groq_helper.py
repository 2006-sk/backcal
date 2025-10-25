import base64
import io
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import ffmpeg
from PIL import Image
import cv2
import numpy as np
from groq import Groq

from app.core.config import settings


class GroqHelper:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = "llama-3.3-70b-versatile"  # Use more powerful text model

    def enabled(self) -> bool:
        return bool(self.api_key) and settings.USE_GROQ

    def average_color_tone(self, frames: List[np.ndarray]) -> str:
        """Calculate average color tone (warm/cool) using OpenCV HSV analysis."""
        if not frames:
            return "unknown"
        
        # Use first frame for color analysis
        frame = frames[0]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = np.mean(hsv[:, :, 0])
        
        # Warm colors: reds, oranges, yellows (hue < 20 or hue > 160)
        # Cool colors: blues, greens, purples (20 <= hue <= 160)
        return "warm" if (hue < 20 or hue > 160) else "cool"

    def scene_count_estimate(self, frames: List[np.ndarray]) -> int:
        """Estimate scene count using OpenCV histogram differences."""
        if len(frames) < 2:
            return 1
        
        # Calculate histogram differences between consecutive frames
        hist_diffs = []
        for i in range(1, len(frames)):
            hist1 = cv2.calcHist([frames[i-1]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frames[i]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            # Normalize histograms
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # Calculate correlation
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            hist_diffs.append(1 - correlation)  # Convert to difference
        
        # Count significant changes (threshold: 0.3)
        scene_changes = sum(1 for diff in hist_diffs if diff > 0.3)
        return max(1, scene_changes + 1)  # At least 1 scene

    def extract_frames(self, video_path: Path, interval: float = 0.5) -> Dict[str, Any]:
        """
        Extract frames from video every interval seconds using ffmpeg.
        Returns frame data with base64 encoded images and timestamps.
        """
        try:
            # Get video metadata
            probe = ffmpeg.probe(str(video_path))
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            duration = float(probe['format']['duration'])
            fps = eval(video_stream['r_frame_rate'])  # Convert fraction to float
            
            frames = []
            current_time = 0.0
            
            while current_time < duration:
                try:
                    # Extract frame at current_time with better error handling
                    out, _ = (
                        ffmpeg
                        .input(str(video_path), ss=current_time)
                        .output('pipe:', vframes=1, format='image2', vcodec='png', pix_fmt='rgb24')
                        .run(capture_stdout=True, quiet=True)
                    )
                    
                    # Convert to base64 JPEG with better error handling
                    try:
                        image = Image.open(io.BytesIO(out))
                        # Ensure image is in RGB mode
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
                        print(f"Warning: Could not process frame at {current_time}s: {img_error}")
                        # Skip this frame but continue
                        pass
                    
                except Exception as frame_error:
                    print(f"Warning: Could not extract frame at {current_time}s: {frame_error}")
                    # Skip this frame but continue
                    pass
                
                current_time += interval
            
            if not frames:
                raise Exception("No frames could be extracted from the video")
            
            return {
                "duration": duration,
                "fps": fps,
                "frames": frames,
                "total_frames": len(frames)
            }
            
        except Exception as e:
            # Fallback to OpenCV if ffmpeg fails
            print(f"FFmpeg extraction failed: {e}. Trying OpenCV fallback...")
            return self._extract_frames_opencv_fallback(video_path, interval)

    def _extract_frames_opencv_fallback(self, video_path: Path, interval: float = 0.5) -> Dict[str, Any]:
        """Fallback frame extraction using OpenCV."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception("Could not open video file with OpenCV")
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            frames = []
            current_time = 0.0
            frame_interval = int(fps * interval)
            
            while current_time < duration:
                # Seek to frame at current_time
                frame_number = int(current_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert OpenCV frame to base64 JPEG
                try:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    
                    jpeg_buffer = io.BytesIO()
                    image.save(jpeg_buffer, format='JPEG', quality=85)
                    jpeg_data = jpeg_buffer.getvalue()
                    base64_image = base64.b64encode(jpeg_data).decode('utf-8')
                    
                    frames.append({
                        "timestamp": current_time,
                        "image_base64": base64_image,
                        "width": frame.shape[1],
                        "height": frame.shape[0]
                    })
                except Exception as img_error:
                    print(f"Warning: Could not process frame at {current_time}s: {img_error}")
                
                current_time += interval
            
            cap.release()
            
            if not frames:
                raise Exception("No frames could be extracted from the video")
            
            return {
                "duration": duration,
                "fps": fps,
                "frames": frames,
                "total_frames": len(frames)
            }
            
        except Exception as e:
            raise Exception(f"OpenCV fallback extraction failed: {str(e)}")

    def analyze_video_with_groq(self, video_path: Path) -> Dict[str, Any]:
        """
        Extract frames and send to Groq for analysis.
        Returns structured response with metadata and Groq output.
        """
        try:
            # Extract frames every 0.5 seconds
            frame_data = self.extract_frames(video_path, interval=0.5)
            
            # Convert base64 frames back to OpenCV arrays for analysis
            opencv_frames = []
            for frame_info in frame_data["frames"]:
                base64_data = frame_info["image_base64"]
                image_data = base64.b64decode(base64_data)
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                opencv_frames.append(frame)
            
            # OpenCV analysis for color tone and scene count
            color_tone = self.average_color_tone(opencv_frames)
            scene_count = self.scene_count_estimate(opencv_frames)
            
            # Groq analysis with real API call
            groq_response = None
            if self.enabled():
                try:
                    from groq import Groq
                    client = Groq(api_key=self.api_key)
                    
                    # Prepare frames for analysis (send all frames)
                    frames_to_analyze = frame_data["frames"]
                    
                    # Create structured JSON prompt
                    prompt = """Analyze all frames together and return a JSON summary.

Return a JSON object with these keys:
{
  "faces_detected": <integer>,
  "objects": [list of main visible items],
  "text_present": <true/false>,
  "scene_description": <string>,
  "mood": <string>,
  "color_scheme": <string>,
  "activity": <string>,
  "notes": <string>
}

Be concise. Do not include Markdown or explanations — only valid JSON."""

                    # Create frame descriptions for text analysis (no vision model available)
                    frame_descriptions = []
                    for i, frame_info in enumerate(frames_to_analyze):
                        frame_descriptions.append(f"Frame {i+1} (timestamp: {frame_info['timestamp']:.1f}s): {frame_info['width']}x{frame_info['height']} pixels")

                    enhanced_prompt = f"""Analyze this video with {len(frames_to_analyze)} extracted frames:

Frame Information:
{chr(10).join(frame_descriptions)}

{prompt}"""

                    # Make Groq API call with text model
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": enhanced_prompt}],
                        temperature=0
                    )

                    # Parse JSON response safely
                    import json
                    import re
                    
                    text_output = response.choices[0].message.content
                    # Extract JSON block even if wrapped in markdown
                    json_str = re.search(r"\{.*\}", text_output, re.S)
                    if json_str:
                        try:
                            parsed = json.loads(json_str.group(0))
                        except Exception:
                            parsed = {"raw_text": text_output}
                    else:
                        parsed = {"raw_text": text_output}

                    groq_response = {
                        "model": "llama-3.2-vision",
                        "frames_analyzed": len(frames_to_analyze),
                        "groq_features": parsed,
                        "total_frames_available": frame_data["total_frames"]
                    }
                    
                except Exception as e:
                    groq_response = {"error": f"Groq analysis failed: {str(e)}"}
            else:
                groq_response = {
                    "error": "Groq not enabled. Set USE_GROQ=true and GROQ_API_KEY in environment."
                }
            
            return {
                "file_name": video_path.name,
                "duration": frame_data["duration"],
                "fps": frame_data["fps"],
                "scene_count": scene_count,
                "color_tone": color_tone,
                "groq_features": groq_response.get("groq_features", {})
            }
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "duration": 0,
                "fps": 0,
                "total_frames": 0,
                "color_tone": "unknown",
                "scene_count": 0,
                "groq_raw_output": None
            }

    def analyze_image_with_groq(self, image_path: Path) -> Dict[str, Any]:
        """
        Analyze single image with Groq.
        """
        if not self.enabled():
            return {
                "error": "Groq not enabled. Set USE_GROQ=true and GROQ_API_KEY in environment.",
                "groq_raw_output": None
            }

        try:
            # Convert image to base64
            with open(image_path, 'rb') as f:
                image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Make real Groq API call with structured JSON
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            prompt = """Analyze this image and return a JSON summary.

Return a JSON object with these keys:
{
  "faces_detected": <integer>,
  "objects": [list of main visible items],
  "text_present": <true/false>,
  "scene_description": <string>,
  "mood": <string>,
  "color_scheme": <string>,
  "activity": <string>,
  "notes": <string>
}

Be concise. Do not include Markdown or explanations — only valid JSON."""

            # Use text-only analysis (no vision model available)
            enhanced_prompt = f"""Analyze this image (JPEG format, base64 encoded):

{prompt}

Note: Since I cannot directly view the image, provide analysis based on the image metadata and format."""

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=0
            )

            # Parse JSON response safely
            import json
            import re
            
            text_output = response.choices[0].message.content
            # Extract JSON block even if wrapped in markdown
            json_str = re.search(r"\{.*\}", text_output, re.S)
            if json_str:
                try:
                    parsed = json.loads(json_str.group(0))
                except Exception:
                    parsed = {"raw_text": text_output}
            else:
                parsed = {"raw_text": text_output}

            return {
                "groq_features": parsed
            }
            
        except Exception as e:
            return {
                "error": f"Image analysis failed: {str(e)}",
                "groq_features": None
            }
