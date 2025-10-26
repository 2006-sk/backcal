"""
Unified JSON builder for AdAtlas AI.
Merges outputs from Gemini, Reka, Audio, and Visual modules into frontend-ready JSON.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import numpy as np


def build_unified_json(
    gemini_output: Dict[str, Any],
    reka_output: Dict[str, Any],
    audio_output: Dict[str, Any],
    visual_output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build unified JSON from multiple analysis outputs.
    
    Args:
        gemini_output: Gemini vision analysis results
        reka_output: Reka QuickTag features
        audio_output: Audio transcription and embeddings
        visual_output: Visual captions and embeddings
        
    Returns:
        Unified frontend-ready JSON dictionary
    """
    
    # Extract base information from Gemini output
    gemini_features = gemini_output.get("vision_features", {})
    
    # Build unified structure
    unified = {
        "file_name": gemini_output.get("file_name", "unknown"),
        "duration": round(gemini_output.get("duration", 0.0), 2),
        "scene_count": gemini_output.get("scene_count", 0),
        "color_tone": gemini_output.get("color_tone", "unknown"),
        
        # Summary (truncated to 300 chars)
        "summary": _truncate_text(gemini_output.get("video_summary", ""), 300),
        
        # Visual features
        "visual": {
            "faces_detected": gemini_features.get("faces_detected", 0),
            "objects": _limit_list(gemini_features.get("objects", []), 5),
            "mood": gemini_features.get("mood", "unknown"),
            "color_scheme": gemini_features.get("color_scheme", "unknown"),
            "cta_present": gemini_features.get("cta_present", False)
        },
        
        # Audio features
        "audio": {
            "transcript": audio_output.get("transcript", "") or audio_output.get("audio_transcript", ""),
            "word_count": _count_words(audio_output.get("transcript", "") or audio_output.get("audio_transcript", ""))
        },
        
        # Reka features
        "reka": {
            "expected_ctr": round(reka_output.get("expected_ctr", 0.0), 2) if reka_output.get("expected_ctr") else None,
            "virality_score": round(reka_output.get("virality_score", 0.0), 2) if reka_output.get("virality_score") else None,
            "keywords": _limit_list(reka_output.get("keywords", []), 5),
            "mood_tone": _limit_list(reka_output.get("mood_tone", []), 5),
            "note": reka_output.get("note")  # Reka note for images
        },
        
        # Analysis note
        "note": _generate_analysis_note(gemini_output, reka_output, audio_output, visual_output)
    }
    
    # Add computed fields
    unified["dominant_emotion"] = _get_dominant_emotion(reka_output)
    unified["keyword_count"] = len(reka_output.get("keywords", []))
    unified["audio_duration_ratio"] = _calculate_audio_duration_ratio(
        audio_output.get("transcript", "") or audio_output.get("audio_transcript", ""),
        gemini_output.get("duration", 0.0)
    )
    
    # Add embedding dimensions
    unified["embeddings"] = {
        "audio_dimensions": audio_output.get("embedding_dimensions", 0),
        "visual_dimensions": visual_output.get("embedding_dimensions", 0),
        "audio_generated": audio_output.get("embedding_dimensions", 0) > 0,
        "visual_generated": visual_output.get("embedding_dimensions", 0) > 0
    }
    
    # Add advanced scores
    advanced_scores = _calculate_advanced_scores(
        gemini_output, reka_output, audio_output, visual_output
    )
    unified.update(advanced_scores)
    
    return unified


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length, adding ellipsis if needed."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def _limit_list(items: List[Any], max_items: int) -> List[Any]:
    """Limit list to max_items."""
    if not isinstance(items, list):
        return []
    return items[:max_items]


def _count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def _detect_file_type(filename: str) -> str:
    """Detect file type from filename."""
    if not filename:
        return "unknown"
    
    ext = filename.lower().split('.')[-1]
    type_map = {
        'mp4': 'video/mp4',
        'mov': 'video/quicktime',
        'mkv': 'video/x-matroska',
        'avi': 'video/x-msvideo',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'webp': 'image/webp'
    }
    return type_map.get(ext, 'unknown')


def _get_dominant_emotion(reka_output: Dict[str, Any]) -> str:
    """Extract dominant emotion from Reka mood_tone."""
    mood_tone = reka_output.get("mood_tone", [])
    if mood_tone and isinstance(mood_tone, list):
        return mood_tone[0]
    return "neutral"


def _calculate_audio_duration_ratio(transcript: str, duration: float) -> float:
    """Calculate audio words per second (talk density)."""
    if not duration or duration == 0:
        return 0.0
    
    if not transcript:
        return 0.0
    
    word_count = _count_words(transcript)
    return round(word_count / duration, 2)


def _calculate_advanced_scores(
    gemini_output: Dict[str, Any],
    reka_output: Dict[str, Any],
    audio_output: Dict[str, Any],
    visual_output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate advanced analysis scores.
    Returns Narrative Coherence Score, Emotion Variance Score, and Audio-Visual Alignment Score.
    """
    scores = {}
    
    # 1. Narrative Coherence Score (simplified version)
    narrative_score = _calculate_narrative_coherence(visual_output)
    scores["narrative_coherence_score"] = narrative_score
    
    # 2. Emotion Variance Score
    emotion_score = _calculate_emotion_variance(reka_output)
    scores["emotion_variance_score"] = emotion_score
    
    # 3. Audio-Visual Alignment Score (optional)
    alignment_score = _calculate_audio_visual_alignment(audio_output, visual_output)
    scores["audio_visual_alignment_score"] = alignment_score
    
    return scores


def _calculate_narrative_coherence(visual_output: Dict[str, Any]) -> float:
    """
    Compute Narrative Coherence Score using visual embeddings.
    Simplified: check if embeddings are diverse or similar.
    """
    captions = visual_output.get("captions", [])
    if len(captions) < 2:
        return 0.5  # Neutral score for insufficient data
    
    # Simple heuristic: longer, more detailed captions = better coherence
    avg_caption_length = sum(len(c) for c in captions) / len(captions)
    
    # Normalize to 0-1 scale (assuming avg caption is 50-200 chars)
    normalized = min(max((avg_caption_length - 50) / 150, 0), 1)
    
    return round(normalized, 2)


def _calculate_emotion_variance(reka_output: Dict[str, Any]) -> float:
    """
    Calculate Emotion Variance Score from Reka mood_tone.
    Higher variance = more emotional range.
    """
    mood_tone = reka_output.get("mood_tone", [])
    
    if not mood_tone or len(mood_tone) < 2:
        return 0.5  # Neutral score
    
    # Map emotions to numeric values (simplified)
    emotion_map = {
        "happy": 0.8, "joyful": 0.9, "satisfied": 0.7,
        "relaxing": 0.6, "relief": 0.7, "relaxed": 0.6,
        "motivating": 0.8, "optimistic": 0.8,
        "stressed": 0.3, "worried": 0.2, "distress": 0.1,
        "casual": 0.5, "neutral": 0.5
    }
    
    # Convert emotions to numeric values
    numeric_values = [emotion_map.get(mood.lower(), 0.5) for mood in mood_tone]
    
    # Calculate variance
    variance = np.var(numeric_values)
    
    # Normalize to 0-1 scale (variance of emotions typically 0-0.2)
    normalized = min(variance * 5, 1.0)
    
    return round(normalized, 2)


def _calculate_audio_visual_alignment(audio_output: Dict[str, Any], visual_output: Dict[str, Any]) -> float:
    """
    Calculate Audio-Visual Alignment Score.
    Simplified: check if both audio and visual embeddings exist and are meaningful.
    """
    # Check if both embeddings were generated
    audio_dim = audio_output.get("embedding_dimensions", 0)
    visual_dim = visual_output.get("embedding_dimensions", 0)
    
    # If neither exists, return neutral score
    if audio_dim == 0 and visual_dim == 0:
        return 0.5
    
    # If only one exists, return lower score
    if audio_dim == 0 or visual_dim == 0:
        return 0.3
    
    # Check if transcript and captions exist
    transcript = audio_output.get("transcript", "") or audio_output.get("audio_transcript", "")
    captions = visual_output.get("captions", [])
    
    # Simple heuristic: if both have content, alignment is higher
    if transcript and captions:
        # More captions and transcript = better alignment
        caption_count = len(captions)
        transcript_length = len(transcript)
        
        # Normalize based on content richness
        score = min(0.5 + (caption_count * 0.05) + (min(transcript_length / 100, 0.3)), 1.0)
        return round(score, 2)
    
    return 0.5


def _generate_analysis_note(
    gemini_output: Dict[str, Any],
    reka_output: Dict[str, Any],
    audio_output: Dict[str, Any],
    visual_output: Dict[str, Any]
) -> str:
    """
    Generate interpretive insights from the analysis.
    """
    insights = []
    
    # Analysis quality assessment
    audio_dim = audio_output.get("embedding_dimensions", 0)
    visual_dim = visual_output.get("embedding_dimensions", 0)
    
    if audio_dim > 0 and visual_dim > 0:
        insights.append("High-quality multimodal analysis")
    elif audio_dim > 0 or visual_dim > 0:
        insights.append("Partial analysis completed")
    
    # Content type inference
    vision_features = gemini_output.get("vision_features", {})
    activity = vision_features.get("activity", "")
    if activity and activity != "unknown":
        insights.append(f"Content type: {activity}")
    
    # Narrative quality
    narrative_score = _calculate_narrative_coherence(visual_output)
    if narrative_score > 0.7:
        insights.append("Strong narrative coherence")
    elif narrative_score < 0.3:
        insights.append("Fragmented narrative detected")
    
    # Emotional range
    emotion_score = _calculate_emotion_variance(reka_output)
    if emotion_score > 0.6:
        insights.append("Wide emotional range")
    elif emotion_score < 0.2:
        insights.append("Consistent emotional tone")
    
    # Alignment quality
    alignment_score = _calculate_audio_visual_alignment(audio_output, visual_output)
    if alignment_score > 0.7:
        insights.append("Strong audio-visual alignment")
    elif alignment_score < 0.4:
        insights.append("Weak audio-visual alignment")
    
    # CTA detection
    cta_present = vision_features.get("cta_present", False)
    if cta_present:
        insights.append("Call-to-action detected")
    
    # Performance prediction
    expected_ctr = reka_output.get("expected_ctr")
    if expected_ctr is not None:
        if expected_ctr > 2.0:
            insights.append("High engagement potential")
        elif expected_ctr < 1.0:
            insights.append("Optimization recommended")
    
    if not insights:
        return "Analysis complete - standard processing"
    
    return ". ".join(insights) + "."


# Demo with mock data
if __name__ == "__main__":
    print("=" * 60)
    print("UNIFIED JSON BUILDER - DEMO")
    print("=" * 60)
    
    # Mock input data
    gemini_mock = {
        "file_name": "ad_video_1.mp4",
        "duration": 15.02,
        "scene_count": 2,
        "color_tone": "cool",
        "video_summary": "A vibrant, empowering Olly ad showing a woman's daily journey through stress and relief",
        "vision_features": {
            "faces_detected": 4,
            "objects": ["backpack", "vitamin bottles", "kitchen", "family", "child", "toys"],
            "mood": "joyful",
            "color_scheme": "vibrant pinks and yellows",
            "cta_present": False
        }
    }
    
    reka_mock = {
        "expected_ctr": 1.8,
        "virality_score": 70,
        "keywords": ["health", "daily routine", "family", "vibrant", "wellness"],
        "mood_tone": ["motivating", "optimistic", "relaxing"]
    }
    
    audio_mock = {
        "transcript": "Being a woman, it's a lot. I'm trying to balance everything.",
        "embedding_dimensions": 768
    }
    
    visual_mock = {
        "captions": [
            "A woman unboxes a package",
            "Kitchen with colorful items",
            "Family enjoying time together"
        ],
        "embedding_dimensions": 768
    }
    
    # Build unified JSON
    unified_json = build_unified_json(
        gemini_mock,
        reka_mock,
        audio_mock,
        visual_mock
    )
    
    # Print result
    import json
    print("\n✅ Unified JSON Output:\n")
    print(json.dumps(unified_json, indent=2))
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

