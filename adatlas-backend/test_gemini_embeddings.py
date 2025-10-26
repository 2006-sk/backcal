#!/usr/bin/env python3
"""Test Gemini embeddings API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
import google.generativeai as genai

def test_gemini_embeddings():
    print("Testing Gemini Embeddings API...")
    print(f"USE_GEMINI: {settings.USE_GEMINI}")
    print(f"API Key: {settings.GEMINI_API_KEY[:20]}..." if settings.GEMINI_API_KEY else "API Key: Not configured")
    
    if not settings.USE_GEMINI or not settings.GEMINI_API_KEY:
        print("ERROR: Gemini not configured")
        return False
    
    try:
        # Configure Gemini
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Test embedding generation
        print("\n1. Testing text embedding...")
        test_text = "A video showing a person walking in a park"
        
        result = genai.embed_content(
            model='models/text-embedding-004',
            content=test_text,
            task_type='retrieval_document'
        )
        
        embedding = result['embedding']
        print(f"   SUCCESS: Generated embedding")
        print(f"   Dimensions: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        # Test multiple embeddings
        print("\n2. Testing multiple embeddings...")
        texts = [
            "A video showing a person walking",
            "A video showing a person running",
            "A video showing a person sitting"
        ]
        
        embeddings = []
        for i, text in enumerate(texts):
            result = genai.embed_content(
                model='models/text-embedding-004',
                content=text,
                task_type='retrieval_document'
            )
            embeddings.append(result['embedding'])
            print(f"   Generated embedding {i+1}/{len(texts)}")
        
        # Test averaging
        import numpy as np
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        print(f"   Averaged embedding dimensions: {len(avg_embedding)}")
        
        print("\nSUCCESS: Gemini embeddings are working!")
        return True
        
    except Exception as e:
        print(f"\nERROR: Gemini embeddings failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gemini_embeddings()
    sys.exit(0 if success else 1)

