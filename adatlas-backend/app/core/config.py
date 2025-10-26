from pydantic import BaseModel
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # App Configuration
    APP_NAME: str = "AdAtlas AI – Feature Extractor"
    APP_VERSION: str = "0.1.0"

    # Data Directories
    DATA_DIR: str = "./data"
    UPLOAD_DIR: str = "./data/uploads"
    RESULT_DIR: str = "./data/results"

    # API Keys and Settings (loaded from .env)
    USE_GROQ: bool = False
    GROQ_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    USE_GEMINI: bool = False
    GEMINI_API_KEY: str | None = None
    USE_REKA: bool = False
    REKA_API_KEY: str | None = None
    REKA_BASE: str = "https://vision-agent.api.reka.ai"
    REKA_TIMEOUT: int = 120
    
    # Frame Analysis Limits
    MAX_VISION_FRAMES: int = 20  # Max frames for OpenAI vision analysis (to avoid rate limits)
    
    # Gemini Model Selection
    GEMINI_MODEL: str = "gemini-2.5-flash"  # Options: "gemini-2.5-flash" (fast) or "gemini-2.5-pro" (accurate)
    
    # ChromaDB Configuration
    USE_CHROMA: bool = False
    CHROMA_API_KEY: str | None = None
    CHROMA_TENANT: str | None = None
    CHROMA_DATABASE: str | None = None
    CHROMA_COLLECTION_NAME: str = "adatlas_embeddings"

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from env file

settings = Settings()

# Ensure runtime dirs exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULT_DIR, exist_ok=True)
