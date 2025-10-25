from pydantic import BaseModel
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    APP_NAME: str = "AdAtlas AI – Feature Extractor"
    APP_VERSION: str = "0.1.0"

    DATA_DIR: str = "./data"
    UPLOAD_DIR: str = "./data/uploads"
    RESULT_DIR: str = "./data/results"

    # Groq Configuration
    USE_GROQ: bool = False
    GROQ_API_KEY: str | None = None

    # OpenAI Configuration
    OPENAI_API_KEY: str | None = None

    # Reka Configuration
    USE_REKA: bool = False
    REKA_API_KEY: str | None = None

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from env file

settings = Settings()

# Ensure runtime dirs exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULT_DIR, exist_ok=True)
