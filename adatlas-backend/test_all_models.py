#!/usr/bin/env python3
"""Test speed of different Gemini models."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
import google.generativeai as genai

def test_model_speed(model_name):
    """Test a model's speed."""
    print(f"\nTesting {model_name}...")
    
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name)
        
        start = time.time()
        response = model.generate_content("Say hello in one word")
        elapsed = time.time() - start
        
        print(f"   Response: {response.text}")
        print(f"   Time: {elapsed:.2f}s")
        return elapsed
    except Exception as e:
        print(f"   ERROR: {e}")
        return None

if __name__ == "__main__":
    models = [
        'gemini-2.5-flash',
        'gemini-2.0-flash',
        'gemini-2.5-pro',
        'gemini-flash-latest'
    ]
    
    results = {}
    for model in models:
        elapsed = test_model_speed(model)
        if elapsed:
            results[model] = elapsed
    
    print("\n=== Speed Summary ===")
    for model, time in sorted(results.items(), key=lambda x: x[1]):
        print(f"{model}: {time:.2f}s")

