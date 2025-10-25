import base64
import io
import json
import re
import os
import ssl
from pathlib import Path
from typing import Dict, Any, List, Optional
import ffmpeg
from PIL import Image
import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.core.config import settings


class OpenAIHelper:
    def __init__(self, openai_api_key: str | None = None, reka_api_key: str | None = None):
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        self.reka_api_key = reka_api_key or settings.REKA_API_KEY
        self.use_reka = settings.USE_REKA

    def enabled(self) -> bool:
        return bool(self.openai_api_key)
    
    def _create_session(self):
        """Create a robust requests session with SSL handling."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Configure SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Apply SSL context to session
        session.verify = False
        
        return session

    def extract_keyframes(self, video_path: Path, step_sec: float = 0.5) -> Dict[str, Any]:
        """
        Extract keyframes from video using ffmpeg at 0.5s intervals.
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
                    # Extract frame at current_time
                    out, _ = (
                        ffmpeg
                        .input(str(video_path), ss=current_time)
                        .output('pipe:', vframes=1, format='image2', vcodec='png', pix_fmt='rgb24')
                        .run(capture_stdout=True, quiet=True)
                    )

                    # Convert to base64 JPEG
                    try:
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

                except Exception as frame_error:
                    print(f"Warning: Could not extract frame at {current_time:.1f}s: {frame_error}")

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
            # Fallback to OpenCV if ffmpeg fails
            print(f"FFmpeg extraction failed: {e}. Trying OpenCV fallback...")
            return self._extract_frames_opencv_fallback(video_path, step_sec)

    def _extract_frames_opencv_fallback(self, video_path: Path, step_sec: float = 0.5) -> Dict[str, Any]:
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

            while current_time < duration:
                frame_number = int(current_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                ret, frame = cap.read()
                if not ret:
                    break

                try:
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

                current_time += step_sec

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

    def analyze_with_reka(self, video_path: Path) -> Dict[str, Any]:
        """
        Analyze video with Reka Flash for holistic video understanding.
        """
        if not self.reka_api_key:
            return {"error": "Reka API key not configured"}

        try:
            # Reka Flash API endpoint
            url = "https://api.reka.ai/v1/chat"
            
            headers = {
                "Authorization": f"Bearer {self.reka_api_key}",
                "Content-Type": "application/json"
            }

            # Read video file
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            video_base64 = base64.b64encode(video_data).decode('utf-8')

            payload = {
                "model": "reka-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this video and return a JSON summary with these keys:
{
  "faces_detected": <integer>,
  "objects": [list of main visible items],
  "text_present": <true/false>,
  "scene_description": <string>,
  "mood": <string>,
  "color_scheme": <string>,
  "activity": <string>,
  "motion_style": <string>,
  "cta_present": <true/false>,
  "notes": <string>
}

Be concise. Return only valid JSON."""
                            },
                            {
                                "type": "video",
                                "video_url": f"data:video/mp4;base64,{video_base64}"
                            }
                        ]
                    }
                ],
                "temperature": 0
            }

            session = self._create_session()
            response = session.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Parse JSON response
            json_str = re.search(r"\{.*\}", content, re.S)
            if json_str:
                try:
                    parsed = json.loads(json_str.group(0))
                    return {"reka_analysis": parsed}
                except Exception:
                    return {"reka_analysis": {"raw_text": content}}
            else:
                return {"reka_analysis": {"raw_text": content}}

        except Exception as e:
            return {"error": f"Reka analysis failed: {str(e)}"}

    def analyze_with_openai(self, video_path: Path, frames: List[Dict[str, Any]], use_reka: bool = False) -> Dict[str, Any]:
        """
        Analyze video frames with either Groq or OpenAI based on environment variables.
        Returns structured JSON with the same schema regardless of backend.
        """
        # Check environment variables
        USE_GROQ = settings.USE_GROQ
        GROQ_KEY = settings.GROQ_API_KEY
        OPENAI_KEY = settings.OPENAI_API_KEY
        
        # Determine which backend to use
        use_groq = USE_GROQ and bool(GROQ_KEY)
        use_openai = bool(OPENAI_KEY)
        
        print(f"[Vision] Using {'Groq' if use_groq else 'OpenAI GPT-4o-mini'} backend.")
        
        try:
            if use_groq:
                return self._analyze_with_groq(video_path, frames)
            elif use_openai:
                return self._analyze_with_openai_gpt4o(video_path, frames)
            else:
                return {
                    "error": "No vision backend available. Set either GROQ_API_KEY or OPENAI_API_KEY.",
                    "vision_features": None
                }
        except Exception as e:
            return {
                "error": str(e),
                "vision_features": None
            }

    def _analyze_with_groq(self, video_path: Path, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze with Groq using existing logic."""
        try:
            from groq import Groq
            client = Groq(api_key=settings.GROQ_API_KEY)
            
            # Create frame descriptions for text analysis
            frame_descriptions = []
            for i, frame_info in enumerate(frames):
                frame_descriptions.append(f"Frame {i+1} (timestamp: {frame_info['timestamp']:.1f}s): {frame_info['width']}x{frame_info['height']} pixels")

            prompt = f"""Analyze this video with {len(frames)} extracted frames:

Frame Information:
{chr(10).join(frame_descriptions)}

Return a JSON object with these keys:
{{
  "faces_detected": <integer>,
  "objects": [list of main visible items],
  "text_present": <true/false>,
  "scene_description": <string>,
  "mood": <string>,
  "color_scheme": <string>,
  "activity": <string>,
  "motion_style": <string>,
  "cta_present": <true/false>,
  "notes": <string>
}}

Be concise. Return only valid JSON."""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            # Parse JSON response
            content = response.choices[0].message.content
            json_str = re.search(r"\{.*\}", content, re.S)
            
            if json_str:
                try:
                    parsed = json.loads(json_str.group(0))
                except Exception:
                    parsed = {"raw_text": content}
            else:
                parsed = {"raw_text": content}

            return {
                "vision_features": parsed
            }

        except Exception as e:
            raise Exception(f"Groq analysis failed: {str(e)}")

    def _analyze_with_openai_gpt4o(self, video_path: Path, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze with OpenAI GPT-4o-mini using sequential processing."""
        try:
            # Use all frames for comprehensive analysis
            frames_to_analyze = frames
            print(f"[Vision] Analyzing all {len(frames_to_analyze)} frames sequentially with OpenAI GPT-4o-mini")
            
            # Prepare content for OpenAI API
            content = [{
                "type": "text",
                "text": """You are a video creative feature extraction model.

Analyze all frames together and return ONLY valid JSON. These frames represent different moments in the video timeline.

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
            }]
            
            # Add all frames to content
            for frame in frames_to_analyze:
                content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame['image_base64']}"
                    }
                })

            # Make OpenAI API call using requests
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.2,
                "max_tokens": 1000
            }
            
            session = self._create_session()
            response = session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            
            if response.status_code != 200:
                print(f"[Vision] OpenAI API call failed: {response.status_code} - {response.text}")
                raise Exception(f"OpenAI API call failed: {response.status_code} - {response.text}")
            
            result = response.json()
            txt = result['choices'][0]['message']['content']
            print(f"[Vision] OpenAI response: {txt[:200]}...")

            # Extract JSON safely
            match = re.search(r"\{.*\}", txt, re.S)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    print(f"[Vision] JSON parsed successfully: {parsed}")
                except Exception as e:
                    print(f"[Vision] JSON parsing failed: {e}")
                    parsed = {"raw_text": txt}
            else:
                print(f"[Vision] No JSON found in response")
                parsed = {"raw_text": txt}

            return {
                "vision_features": parsed
            }

        except Exception as e:
            raise Exception(f"OpenAI analysis failed: {str(e)}")

