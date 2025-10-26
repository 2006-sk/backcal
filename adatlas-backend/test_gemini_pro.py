#!/usr/bin/env python3
"""Test Gemini Pro model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
import google.generativeai as genai

def test_gemini_pro():
    print("Testing Gemini 2.5 Pro model...")
    
    if not settings.GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not configured")
        return False
    
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Test text generation
        print("\n1. Testing text generation...")
        response = model.generate_content("Say hello in one word")
        print(f"   Response: {response.text}")
        
        print("\nSUCCESS: Gemini 2.5 Pro is working!")
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_gemini_pro()
    sys.exit(0 if success else 1)

