"""
Reka helper with stable upload + indexed + chat workflow.
"""
import json
import re
import asyncio
from typing import Dict, Any
import httpx
from app.core.config import settings
from app.utils.reka_client import reka_upload_video, reka_wait_indexed, reka_chat_video


async def _parse_reka_with_groq(raw_text: str) -> Dict[str, Any]:
    """
    Use Groq LLaMA to parse Reka raw text into structured JSON.
    
    Args:
        raw_text: Raw Reka response text
        
    Returns:
        Dict with expected_ctr, virality_score, keywords, mood_tone
    """
    try:
        from groq import Groq
        
        if not settings.GROQ_API_KEY:
            return None
        
        client = Groq(api_key=settings.GROQ_API_KEY)
        
        prompt = f"""Parse this Reka video analysis response and extract key metrics.

Raw Response:
{raw_text}

Return ONLY valid JSON with these exact fields:
{{
  "expected_ctr": <number between 0.5 and 5.0>,
  "virality_score": <number between 0 and 100>,
  "keywords": [<list of 5-10 relevant keywords>],
  "mood_tone": [<list of 2-5 mood descriptors>]
}}

Analyze the content and provide reasonable estimates based on the description.
If specific metrics aren't mentioned, use sensible defaults:
- expected_ctr: 1.5 (moderate)
- virality_score: 50 (average)
- Extract actual keywords and moods from the text

Return ONLY the JSON, no other text."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        text_output = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', text_output)
        if json_match:
            parsed = json.loads(json_match.group(0))
            return parsed
        
        return None
        
    except Exception as e:
        print(f"[Reka] Groq parsing error: {e}")
        return None


class RekaHelper:
    def __init__(self):
        self.api_key = settings.REKA_API_KEY
        self.enabled = settings.USE_REKA and bool(self.api_key)
    
    async def get_reka_features(self, file_path: str) -> Dict[str, Any]:
        """
        Get Reka features using QuickTag priority, fallback to upload + indexed + chat.
        """
        if not self.enabled:
            print("[Reka] Disabled")
            return {"raw_data": None, "error": "Reka disabled"}
        
        try:
            print(f"[Reka] Getting features for: {file_path}")
            
            # Initialize variables for QuickTag results
            expected_ctr = None
            virality_score = None
            keywords = []
            mood_tone = []
            quicktag_result = None
            
            # PRIORITY: Try QuickTag first (fast, no indexing needed)
            print("[Reka] Priority: Trying QuickTag endpoint...")
            try:
                from app.utils.reka_client import reka_quicktag
                import json
                quicktag_result = await reka_quicktag(file_path, self.api_key)
                
                # QuickTag succeeded! Parse response if it's a string
                print("[Reka] QuickTag successful!")
                
                # Handle string response (sometimes Reka returns JSON string)
                if isinstance(quicktag_result, str):
                    try:
                        # Try to parse as JSON
                        quicktag_result = json.loads(quicktag_result)
                    except json.JSONDecodeError:
                        # If JSON fails, try stripping quotes first
                        try:
                            stripped = quicktag_result.strip().strip('"').strip("'")
                            quicktag_result = json.loads(stripped)
                            print("[Reka] Parsed QuickTag string after stripping quotes")
                        except:
                            print(f"[Reka] Failed to parse QuickTag string: {quicktag_result[:100]}")
                            # Don't raise exception - let it fall through to smart parsing
                            quicktag_result = {"raw_response": quicktag_result}
                
                # Check if data is in raw_response wrapper
                if "raw_response" in quicktag_result:
                    raw_str = quicktag_result["raw_response"]
                    try:
                        # raw_response contains a JSON string - parse it
                        if isinstance(raw_str, str):
                            # Try json.loads first
                            try:
                                parsed_json = json.loads(raw_str)
                                quicktag_result = parsed_json
                                print(f"[Reka] Parsed raw_response with json.loads")
                            except json.JSONDecodeError:
                                # If json.loads fails, try ast.literal_eval
                                import ast
                                try:
                                    parsed_json = ast.literal_eval(raw_str)
                                    quicktag_result = parsed_json
                                    print(f"[Reka] Parsed raw_response with ast.literal_eval")
                                except Exception as ast_err:
                                    print(f"[Reka] Failed to parse with ast.literal_eval: {ast_err}")
                                    # Try extracting JSON block as last resort
                                    json_match = re.search(r'\{[\s\S]*\}', str(raw_str))
                                    if json_match:
                                        try:
                                            json_str = json_match.group(0)
                                            quicktag_result = json.loads(json_str)
                                            print(f"[Reka] Parsed extracted JSON block")
                                        except:
                                            print(f"[Reka] Failed to parse extracted JSON")
                        else:
                            quicktag_result = raw_str
                    except Exception as e:
                        print(f"[Reka] Failed to parse raw_response: {e}")
                
                # Parse values (may be strings or numbers)
                expected_ctr = quicktag_result.get("ExpectedCTR")
                if expected_ctr is not None:
                    if isinstance(expected_ctr, str):
                        try:
                            expected_ctr = float(expected_ctr)
                        except:
                            expected_ctr = None
                    elif isinstance(expected_ctr, (int, float)):
                        expected_ctr = float(expected_ctr)
                    
                    # Normalize CTR: if value > 100, divide by 100 (e.g., 150 → 1.5%)
                    if expected_ctr is not None and expected_ctr > 100:
                        original_ctr = expected_ctr
                        expected_ctr = expected_ctr / 100
                        print(f"[Reka] Normalized ExpectedCTR from {original_ctr} to {expected_ctr}%")
                
                virality_score = quicktag_result.get("ViralityScore")
                if virality_score is not None:
                    if isinstance(virality_score, str):
                        try:
                            virality_score = float(virality_score)
                        except:
                            virality_score = None
                    elif isinstance(virality_score, (int, float)):
                        virality_score = float(virality_score)
                
                keywords = quicktag_result.get("Keyword", [])
                if not isinstance(keywords, list):
                    keywords = []
                
                mood_tone = quicktag_result.get("MoodTone", [])
                if not isinstance(mood_tone, list):
                    mood_tone = []
                
                # If this is not a proper QuickTag response (missing expected fields), try smart parsing
                if expected_ctr is None and virality_score is None and not keywords and not mood_tone:
                    print("[Reka] QuickTag response missing expected fields, attempting smart parsing...")
                    
                    # Try to extract useful data from raw text response
                    extracted_keywords = []
                    extracted_moods = []
                    
                    # Search for common keywords in the raw response
                    raw_text = str(quicktag_result)
                    import re
                    
                    # Extract keywords from raw text (adaptable parsing)
                    # Find capitalized words (likely objects/brands)
                    words = re.findall(r'\b[A-Z][a-z]+\b', raw_text)
                    
                    # Common objects (expand this list as needed)
                    common_objects = ['bag', 'pen', 'vitamin', 'jar', 'toy', 'table', 'door', 'kitchen', 
                                     'lighting', 'bench', 'package', 'knife', 'cutting', 'board', 'spoon',
                                     'fork', 'plate', 'cardboard', 'box', 'delivery', 'unboxing']
                    
                    # Filter out unwanted words
                    filter_words = ['false', 'drugs', 'violence', 'profanity', 'alcohol', 'gambling', 
                                   'political', 'background', 'scene', 'there', 'which', 'these', 'constituents']
                    
                    # Extract meaningful keywords
                    for word in words:
                        word_lower = word.lower()
                        # Skip filter words
                        if word_lower in filter_words:
                            continue
                        # Include if it's a common object or a significant word (>4 chars)
                        if word_lower in common_objects or (len(word) > 4 and word_lower not in filter_words):
                            extracted_keywords.append(word_lower)
                    
                    # Extract mood indicators (flexible matching)
                    mood_words = ['casual', 'lively', 'homely', 'well-lit', 'everyday', 'routine', 'familial',
                                 'satisfied', 'skillful', 'creative', 'promotional', 'happy', 'relaxing', 
                                 'relief', 'relaxed', 'stressed', 'worried', 'distress']
                    text_lower = raw_text.lower()
                    for mood in mood_words:
                        if mood in text_lower:
                            extracted_moods.append(mood)
                    
                    # Remove duplicates
                    extracted_keywords = list(set(extracted_keywords[:10]))  # Limit to 10
                    extracted_moods = list(set(extracted_moods[:5]))  # Limit to 5
                    
                    if extracted_keywords or extracted_moods:
                        print(f"[Reka] Extracted {len(extracted_keywords)} keywords and {len(extracted_moods)} moods from raw text")
                        
                        # Try Groq to format the response properly
                        try:
                            print("[Reka] Using Groq to parse Reka response into structured format...")
                            groq_result = await _parse_reka_with_groq(raw_text)
                            if groq_result and groq_result.get("expected_ctr") is not None:
                                expected_ctr = groq_result.get("expected_ctr")
                                virality_score = groq_result.get("virality_score")
                                keywords = groq_result.get("keywords", extracted_keywords)
                                mood_tone = groq_result.get("mood_tone", extracted_moods)
                                print(f"[Reka] Groq parsed successfully: CTR={expected_ctr}, Virality={virality_score}")
                            else:
                                # Fall through to upload+index+chat
                                expected_ctr = None
                                virality_score = None
                                keywords = extracted_keywords
                                mood_tone = extracted_moods
                        except Exception as groq_err:
                            print(f"[Reka] Groq parsing failed: {groq_err}")
                            # Fall through to upload+index+chat
                            expected_ctr = None
                            virality_score = None
                            keywords = extracted_keywords
                            mood_tone = extracted_moods
                    else:
                        print("[Reka] Could not extract useful data from raw text")
                        # Clear variables to trigger fallback
                        expected_ctr = None
                        virality_score = None
                        keywords = []
                        mood_tone = []
                
                return {
                    "expected_ctr": expected_ctr,
                    "virality_score": virality_score,
                    "keywords": keywords,
                    "mood_tone": mood_tone,
                    "raw_data": quicktag_result,
                    "method": "quicktag",
                    "error": None
                }
                
            except Exception as quicktag_error:
                print(f"[Reka] QuickTag failed: {quicktag_error}")
                print("[Reka] Fallback: Using upload + indexing + chat workflow...")
            
            # ALWAYS fallback if QuickTag didn't provide structured data
            if expected_ctr is None and virality_score is None:
                print("[Reka] QuickTag didn't provide structured data, forcing fallback...")
            
            # FALLBACK: Upload + Indexing + Chat workflow
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
                    section_type = section.get('section_type', '')
                    section_content = section.get('section_content', '')
                    
                    if section_type == 'video-clips-info':
                        # Handle video-clips-info format
                        video_clips = section_content.get('video_clips', [])
                        for clip in video_clips:
                            clip_info = clip.get('video_clip_info', '')
                            
                            # Extract moods from bold text
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
                    
                    elif section_type == 'markdown':
                        # Handle markdown format
                        content = str(section_content)
                        
                        # Extract moods from markdown
                        mood_section = re.search(r'### Moods & Emotions\n(.*?)\n\n###', content, re.DOTALL)
                        if mood_section:
                            mood_text = mood_section.group(1)
                            mood_items = re.findall(r'- \*\*([^*]+)\*\*: ([^\n]+)', mood_text)
                            for mood_type, mood_list in mood_items:
                                if 'mood' in mood_type.lower():
                                    mood_list_clean = mood_list.replace('**', '').strip()
                                    moods.extend([m.strip() for m in mood_list_clean.split(',')])
                                elif 'emotion' in mood_type.lower():
                                    emotion_list_clean = mood_list.replace('**', '').strip()
                                    emotions.extend([e.strip() for e in emotion_list_clean.split(',')])
                        
                        # Extract objects from markdown
                        object_section = re.search(r'### Objects\n(.*?)\n\n###', content, re.DOTALL)
                        if object_section:
                            object_text = object_section.group(1)
                            object_items = re.findall(r'- ([^\n]+)', object_text)
                            objects.extend([item.strip() for item in object_items])
                        
                        # Extract colors from markdown
                        color_section = re.search(r'### Colors\n(.*?)\n\n###', content, re.DOTALL)
                        if color_section:
                            color_text = color_section.group(1)
                            color_items = re.findall(r'- ([^\n]+)', color_text)
                            colors.extend([item.strip() for item in color_items])
                        
                        # Extract scene summaries from markdown
                        scene_section = re.search(r'### Short Scene Summaries\n(.*?)$', content, re.DOTALL)
                        if scene_section:
                            scene_text = scene_section.group(1)
                            scene_items = re.findall(r'- \*\*From ([^:]+) to ([^:]+):\*\* ([^\n]+)', scene_text)
                            for start_time, end_time, summary in scene_items:
                                # Parse time strings (e.g., "0:00" to "0:08")
                                def parse_time(time_str):
                                    parts = time_str.split(':')
                                    return float(parts[0]) * 60 + float(parts[1])
                                
                                scene = {
                                    "start_time": parse_time(start_time),
                                    "end_time": parse_time(end_time),
                                    "summary": summary.strip(),
                                    "moods": [],
                                    "objects": [],
                                    "colors": []
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
