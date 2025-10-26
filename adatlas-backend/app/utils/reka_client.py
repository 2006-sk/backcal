"""
Reka API client - implemented exactly as per documentation.
"""
import httpx
import json
import os
from typing import Dict, Any, Optional
from app.core.config import settings


async def reka_upload_video(file_path: str) -> Optional[str]:
    """
    Upload video to Reka and return video_id.
    
    Args:
        file_path: Path to video file
        
    Returns:
        video_id if successful, None if failed
    """
    url = f"{settings.REKA_BASE}/videos/upload"
    
    headers = {
        "X-Api-Key": settings.REKA_API_KEY
    }
    
    print(f"[Reka] Uploading video: {file_path}")
    print(f"[Reka] URL: {url}")
    
    try:
        with open(file_path, 'rb') as f:
            files = {
                'file': (os.path.basename(file_path), f, 'video/mp4')
            }
            data = {
                'video_name': os.path.basename(file_path),
                'index': 'true'
            }
            print(f"[Reka] Files: {files}")
            print(f"[Reka] Data: {data}")
            
            async with httpx.AsyncClient(timeout=settings.REKA_TIMEOUT) as client:
                response = await client.post(url, headers=headers, files=files, data=data)
                
                print(f"[Reka] Upload Status: {response.status_code}")
                print(f"[Reka] Upload Response: {response.text[:200]}...")
                
                if response.status_code == 200:
                    result = response.json()
                    video_id = result.get('video_id')
                    print(f"[Reka] Upload successful, video_id: {video_id}")
                    return video_id
                else:
                    print(f"[Reka] Upload failed: {response.status_code} - {response.text}")
                    return None
                    
    except Exception as e:
        print(f"[Reka] Upload error: {e}")
        return None


async def reka_indexed_tag(video_id: str) -> Dict[str, Any]:
    """
    Get indexed tags for an uploaded video.
    
    Args:
        video_id: Video ID from upload
        
    Returns:
        Dict with Reka response data
    """
    url = f"{settings.REKA_BASE}/qa/indexedtag"
    
    headers = {
        "X-Api-Key": settings.REKA_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "video_id": video_id
    }
    
    print(f"[Reka] Getting indexed tags for video_id: {video_id}")
    print(f"[Reka] URL: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=settings.REKA_TIMEOUT) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            print(f"[Reka] Indexed Tag Status: {response.status_code}")
            print(f"[Reka] Indexed Tag Response: {response.text[:200]}...")
            
            if response.status_code == 200:
                return response.json()
            else:
                try:
                    err_json = response.json()
                    print("Error JSON:\n", json.dumps(err_json, indent=4))
                    raise Exception(f"Reka indexed tag error {response.status_code}: {err_json}")
                except ValueError:
                    print("Error Text:", response.text)
                    raise Exception(f"Reka indexed tag error {response.status_code}: {response.text}")
                    
    except Exception as e:
        print(f"[Reka] Indexed tag error: {e}")
        raise


async def reka_wait_indexed(video_id: str, timeout_minutes: int = 1.5) -> bool:
    """
    Poll /videos/get every 3 seconds until indexing_status == indexed.
    Stop after timeout_minutes and return True / False.
    
    Args:
        video_id: Video ID to check
        timeout_minutes: Timeout in minutes (default 1.5 minutes for QuickTag fallback)
    """
    url = f"{settings.REKA_BASE}/videos/get"
    
    headers = {
        "X-Api-Key": settings.REKA_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "video_ids": [video_id]
    }
    
    print(f"[Reka] Waiting for video {video_id} to be indexed (timeout: {timeout_minutes}m)...")
    
    import asyncio
    import time
    start_time = time.time()
    max_attempts = int(timeout_minutes * 20)  # 20 attempts per minute (3s interval)
    attempt = 0
    
    while attempt < max_attempts:
        try:
            async with httpx.AsyncClient(timeout=settings.REKA_TIMEOUT) as client:
                response = await client.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    if results and len(results) > 0:
                        video_data = results[0]
                        status = video_data.get('indexing_status', 'unknown')
                        
                        # Print progress every 5 seconds (every ~1.67 attempts)
                        if attempt % 2 == 0:  # Every 6 seconds (2 * 3 seconds)
                            print(f"[Reka] Indexing: {status} (attempt {attempt + 1}/{max_attempts})")
                        
                        if status == 'indexed':
                            elapsed = time.time() - start_time
                            print(f"[Reka] Video {video_id} is now indexed! (took {elapsed:.1f}s)")
                            return True
                        elif status == 'failed':
                            print(f"[Reka] Video {video_id} indexing failed!")
                            return False
                        else:
                            # Still processing, wait 3 seconds
                            await asyncio.sleep(3)
                            attempt += 1
                    else:
                        print(f"[Reka] No video data found in response")
                        await asyncio.sleep(3)
                        attempt += 1
                else:
                    print(f"[Reka] Status check failed: {response.status_code} - {response.text}")
                    await asyncio.sleep(3)
                    attempt += 1
                    
        except Exception as e:
            print(f"[Reka] Error checking status: {e}")
            await asyncio.sleep(3)
            attempt += 1
    
    elapsed = time.time() - start_time
    print(f"[Reka] Timeout: Video {video_id} not indexed after {elapsed:.1f}s ({timeout_minutes}m)")
    return False


async def reka_index_for_chatbot(file_path: str) -> Dict[str, Any]:
    """
    Index video specifically for chatbot usage (separate thread, 2min timeout).
    This runs independently and can be used for chatbot features.
    
    Args:
        file_path: Path to video file
        
    Returns:
        Dictionary with video_id and indexing status
    """
    print(f"[Reka Chatbot] Starting chatbot indexing for {file_path}...")
    
    try:
        # Upload video
        video_id = await reka_upload_video(file_path)
        if not video_id:
            return {
                "video_id": None,
                "indexed": False,
                "error": "Upload failed"
            }
        
        # Wait for indexing with 2-minute timeout for chatbot
        indexed = await reka_wait_indexed(video_id, timeout_minutes=2.0)
        
        return {
            "video_id": video_id,
            "indexed": indexed,
            "error": None if indexed else "Indexing timeout"
        }
        
    except Exception as e:
        print(f"[Reka Chatbot] Error: {e}")
        return {
            "video_id": None,
            "indexed": False,
            "error": str(e)
        }


async def reka_chat_video(video_id: str) -> Dict[str, Any]:
    """
    Chat with video to get meta-tag JSON.
    """
    url = f"{settings.REKA_BASE}/qa/chat"
    
    headers = {
        "X-Api-Key": settings.REKA_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "video_id": video_id,
        "messages": [
            {
                "role": "user",
                "content": "Return JSON only with moods, objects, colors, emotions, and short scene summaries"
            }
        ]
    }
    
    print(f"[Reka] Chatting with video {video_id}...")
    print(f"[Reka] URL: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=settings.REKA_TIMEOUT) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            print(f"[Reka] Chat Status: {response.status_code}")
            print(f"[Reka] Chat Response: {response.text[:200]}...")
            
            if response.status_code == 200:
                return response.json()
            else:
                try:
                    err_json = response.json()
                    print("Error JSON:\n", json.dumps(err_json, indent=4))
                    raise Exception(f"Reka chat error {response.status_code}: {err_json}")
                except ValueError:
                    print("Error Text:", response.text)
                    raise Exception(f"Reka chat error {response.status_code}: {response.text}")
                    
    except Exception as e:
        print(f"[Reka] Chat error: {e}")
        raise


async def reka_quicktag(file_path: str, api_key: str) -> Dict[str, Any]:
    """
    Call Reka QuickTag API exactly as per documentation.
    """
    url = f"{settings.REKA_BASE}/qa/quicktag"
    
    # Exactly as per docs: "X-Api-Key" (not "X-API-Key")
    headers = {
        "X-Api-Key": api_key
    }
    
    print(f"[Reka] Calling QuickTag for: {file_path}")
    print(f"[Reka] URL: {url}")
    print(f"[Reka] Headers: {headers}")
    
    try:
        with open(file_path, 'rb') as f:
            # Exactly as per docs: files = {'video': (filename, file, 'video/mp4')}
            files = {'video': (os.path.basename(file_path), f, 'video/mp4')}
            print(f"[Reka] Files: {files}")
            
            async with httpx.AsyncClient(timeout=settings.REKA_TIMEOUT) as client:
                response = await client.post(url, headers=headers, files=files)
                
                print(f"[Reka] Status: {response.status_code}")
                print(f"[Reka] Response: {response.text[:200]}...")
                
                if response.status_code == 200:
                    # Try to parse as JSON first
                    try:
                        result = response.json()
                    except:
                        # If JSON parsing fails, try parsing response text
                        result = response.text
                    
                    # Handle QuickTag response (may be string wrapped in dict or just string)
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except:
                            print(f"[Reka] Could not parse QuickTag string response")
                            result = {"raw_response": result}
                    
                    return result
                else:
                    # Error handling as per docs
                    try:
                        err_json = response.json()
                        print("Error JSON:\n", json.dumps(err_json, indent=4))
                        raise Exception(f"Reka API error {response.status_code}: {err_json}")
                    except ValueError:
                        print("Error Text:", response.text)
                        raise Exception(f"Reka API error {response.status_code}: {response.text}")
                    
    except Exception as e:
        print(f"[Reka] Error: {e}")
        raise
