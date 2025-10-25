"""
Reka helper with stable upload + indexed + chat workflow.
"""
import json
import asyncio
from typing import Dict, Any
import httpx
from app.core.config import settings
from app.utils.reka_client import reka_upload_video, reka_wait_indexed, reka_chat_video


class RekaHelper:
    def __init__(self):
        self.api_key = settings.REKA_API_KEY
        self.enabled = settings.USE_REKA and bool(self.api_key)
    
    async def get_reka_features(self, file_path: str) -> Dict[str, Any]:
        """
        Get Reka features using stable upload + indexed + chat workflow.
        """
        if not self.enabled:
            print("[Reka] Disabled")
            return {"raw_data": None, "error": "Reka disabled"}
        
        try:
            print(f"[Reka] Getting features for: {file_path}")
            
            # Step 1: Upload video
            print("[Reka] Step 1: Uploading video...")
            video_id = await reka_upload_video(file_path)
            if not video_id:
                raise Exception("Video upload failed")
            
            # Step 2: Wait for indexing
            print("[Reka] Step 2: Waiting for video to be indexed...")
            indexed = await reka_wait_indexed(video_id)
            if not indexed:
                raise Exception("Video indexing failed or timed out")
            
            # Step 2.5: Safety buffer before querying
            print("[Reka] Step 2.5: Waiting 2 seconds for backend to finalize...")
            await asyncio.sleep(2)
            
            # Step 3: Chat with video
            print("[Reka] Step 3: Chatting with video...")
            chat_response = await reka_chat_video(video_id)
            
            # Step 4: Parse answer string from chat JSON
            print("[Reka] Step 4: Parsing chat response...")
            print(f"[Reka] Full chat response keys: {list(chat_response.keys())}")
            
            # Try different possible response fields
            answer = chat_response.get('answer', '') or chat_response.get('chat_response', '') or chat_response.get('response', '')
            print(f"[Reka] Raw answer length: {len(answer)} characters")
            print(f"[Reka] Raw answer preview: {answer[:200]}...")
            
            # Clean up markdown code blocks if present
            if answer.startswith('```json'):
                answer = answer[7:]  # Remove ```json
            if answer.endswith('```'):
                answer = answer[:-3]  # Remove ```
            answer = answer.strip()
            
            print(f"[Reka] Cleaned answer length: {len(answer)} characters")
            print(f"[Reka] Cleaned answer preview: {answer[:200]}...")
            
            try:
                # Try to parse the answer as JSON
                parsed_data = json.loads(answer)
                print(f"[Reka] Parsed JSON successfully: {len(parsed_data)} keys")
                
                # Extract data from the sections structure
                moods = []
                objects = []
                colors = []
                emotions = []
                scenes = []
                
                sections = parsed_data.get('sections', [])
                for section in sections:
                    if section.get('section_type') == 'video-clips-info':
                        video_clips = section.get('section_content', {}).get('video_clips', [])
                        for clip in video_clips:
                            clip_info = clip.get('video_clip_info', '')
                            
                            # Extract moods from bold text
                            import re
                            mood_matches = re.findall(r'\*\*([^*]+)\*\*', clip_info)
                            for match in mood_matches:
                                if any(word in match.lower() for word in ['mood', 'feeling', 'emotion']):
                                    moods.append(match)
                            
                            # Extract objects from bold text
                            object_matches = re.findall(r'\*\*([^*]+)\*\*', clip_info)
                            for match in object_matches:
                                if any(word in match.lower() for word in ['vehicle', 'car', 'suv', 'cone', 'trees', 'fence', 'station']):
                                    objects.append(match)
                            
                            # Extract colors from bold text
                            color_matches = re.findall(r'\*\*([^*]+)\*\*', clip_info)
                            for match in color_matches:
                                if any(word in match.lower() for word in ['blue', 'silver', 'green', 'color']):
                                    colors.append(match)
                            
                            # Create scene summary
                            scene = {
                                "start_time": clip.get('video_clip_start_time', 0),
                                "end_time": clip.get('video_clip_end_time', 0),
                                "summary": clip_info,
                                "moods": mood_matches,
                                "objects": [m for m in object_matches if any(word in m.lower() for word in ['vehicle', 'car', 'suv', 'cone', 'trees', 'fence', 'station'])],
                                "colors": [m for m in color_matches if any(word in m.lower() for word in ['blue', 'silver', 'green', 'color'])]
                            }
                            scenes.append(scene)
                
                # Remove duplicates
                moods = list(set(moods))
                objects = list(set(objects))
                colors = list(set(colors))
                
                # Extract emotions from the text
                emotions = []
                for scene in scenes:
                    scene_text = scene['summary'].lower()
                    if 'happy' in scene_text or 'joyful' in scene_text:
                        emotions.append('happy')
                    if 'calm' in scene_text or 'serene' in scene_text:
                        emotions.append('calm')
                    if 'confident' in scene_text:
                        emotions.append('confident')
                    if 'practical' in scene_text or 'informative' in scene_text:
                        emotions.append('practical')
                
                emotions = list(set(emotions))
                
                result = {
                    "moods": moods,
                    "objects": objects,
                    "colors": colors,
                    "emotions": emotions,
                    "scenes": scenes,
                    "raw_data": parsed_data,
                    "method": "upload_chat",
                    "error": None
                }
                
                return result
                
            except json.JSONDecodeError as e:
                print(f"[Reka] JSON parsing failed: {e}")
                # Return raw answer if JSON parsing fails
                return {
                    "moods": [],
                    "objects": [],
                    "colors": [],
                    "emotions": [],
                    "scenes": [],
                    "raw_data": {"answer": answer},
                    "method": "upload_chat",
                    "error": f"JSON parsing failed: {e}"
                }
            
        except httpx.TimeoutException as e:
            print(f"[Reka] Timeout error: {e}")
            return {
                "moods": [],
                "objects": [],
                "colors": [],
                "emotions": [],
                "scenes": [],
                "raw_data": None,
                "method": "timeout",
                "error": f"Timeout: {str(e)}"
            }
        except httpx.HTTPStatusError as e:
            print(f"[Reka] HTTP error: {e}")
            return {
                "moods": [],
                "objects": [],
                "colors": [],
                "emotions": [],
                "scenes": [],
                "raw_data": None,
                "method": "http_error",
                "error": f"HTTP error: {str(e)}"
            }
        except Exception as e:
            print(f"[Reka] General error: {e}")
            return {
                "moods": [],
                "objects": [],
                "colors": [],
                "emotions": [],
                "scenes": [],
                "raw_data": None,
                "method": "failed",
                "error": str(e)
            }
