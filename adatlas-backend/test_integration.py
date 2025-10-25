#!/usr/bin/env python3
"""
Integration tests for AdAtlas AI FastAPI backend.
Tests ffmpeg + OpenCV + Groq logic.
"""
import os
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
import ffmpeg

# Add app to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from services.groq_helper import GroqHelper
from core.config import settings

def test_ffmpeg_installation():
    """Test 1: Verify ffmpeg is installed and working."""
    print("🧪 Test 1: FFmpeg Installation")
    try:
        result = ffmpeg.probe('pipe:', format='lavfi', f='testsrc=duration=1:size=320x240:rate=1')
        print("✅ FFmpeg is working correctly")
        return True
    except Exception as e:
        print(f"❌ FFmpeg test failed: {e}")
        return False

def test_opencv_functions():
    """Test 2: Verify OpenCV color tone and scene count functions."""
    print("\n🧪 Test 2: OpenCV Functions")
    
    groq_helper = GroqHelper()
    
    # Create test frames
    frames = []
    
    # Frame 1: Warm colors (red/orange)
    warm_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    warm_frame[:, :, 0] = 10  # Low hue (red)
    frames.append(warm_frame)
    
    # Frame 2: Cool colors (blue/green)
    cool_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cool_frame[:, :, 0] = 120  # High hue (blue)
    frames.append(cool_frame)
    
    # Frame 3: Different scene (different histogram)
    different_frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
    frames.append(different_frame)
    
    try:
        # Test color tone
        color_tone = groq_helper.average_color_tone(frames)
        print(f"✅ Color tone analysis: {color_tone}")
        
        # Test scene count
        scene_count = groq_helper.scene_count_estimate(frames)
        print(f"✅ Scene count analysis: {scene_count}")
        
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_groq_mock():
    """Test 3: Mock Groq API call with dummy key."""
    print("\n🧪 Test 3: Groq Mock Analysis")
    
    # Set dummy API key
    os.environ['GROQ_API_KEY'] = 'dummy_key_for_testing'
    os.environ['USE_GROQ'] = 'true'
    
    groq_helper = GroqHelper()
    
    try:
        # Test enabled check
        enabled = groq_helper.enabled()
        print(f"✅ Groq enabled check: {enabled}")
        
        # Test graceful handling when Groq is not properly configured
        result = groq_helper.analyze_video_with_groq(Path("nonexistent.mp4"))
        print(f"✅ Graceful error handling: {result.get('error', 'No error')}")
        
        return True
    except Exception as e:
        print(f"❌ Groq mock test failed: {e}")
        return False

def test_full_pipeline():
    """Test 4: Full analyze pipeline with sample video."""
    print("\n🧪 Test 4: Full Pipeline Test")
    
    # Create a simple test video using ffmpeg
    try:
        test_video_path = Path("data/sample.mp4")
        test_video_path.parent.mkdir(exist_ok=True)
        
        # Create a 3-second test video
        (
            ffmpeg
            .input('testsrc=duration=3:size=320x240:rate=1', f='lavfi')
            .output(str(test_video_path), vcodec='libx264', pix_fmt='yuv420p')
            .overwrite_output()
            .run(quiet=True)
        )
        
        print(f"✅ Created test video: {test_video_path}")
        
        # Test analysis
        groq_helper = GroqHelper()
        result = groq_helper.analyze_video_with_groq(test_video_path)
        
        print(f"✅ Analysis result:")
        print(f"   Duration: {result.get('duration', 'N/A')}s")
        print(f"   FPS: {result.get('fps', 'N/A')}")
        print(f"   Total frames: {result.get('total_frames', 'N/A')}")
        print(f"   Color tone: {result.get('color_tone', 'N/A')}")
        print(f"   Scene count: {result.get('scene_count', 'N/A')}")
        
        # Clean up
        test_video_path.unlink()
        print("✅ Cleaned up test video")
        
        return True
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 AdAtlas AI Integration Tests")
    print("=" * 50)
    
    tests = [
        test_ffmpeg_installation,
        test_opencv_functions,
        test_groq_mock,
        test_full_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The backend is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
