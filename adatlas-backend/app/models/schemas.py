from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class AnalyzeResult(BaseModel):
    file_name: str
    duration: Optional[float] = Field(None, description="seconds")
    scene_count: Optional[int] = Field(None, description="estimated number of scenes")
    color_tone: Optional[str] = Field(None, description="warm or cool color tone")
    vision_features: Optional[Dict[str, Any]] = Field(None, description="structured vision analysis features")
    video_summary: Optional[str] = Field(None, description="overall video summary")
    error: Optional[str] = Field(None, description="error message if analysis failed")

class UploadResponse(BaseModel):
    id: str
    file_name: str
    saved_path: str

class BatchAnalyzeRequest(BaseModel):
    file_ids: List[str]

class AnalyzeRequest(BaseModel):
    file_id: Optional[str] = None
    file_path: Optional[str] = None
