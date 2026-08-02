import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# ---------- Force load .env from project root ----------
load_dotenv(override=True)

# ---------- Debug: Check if keys are loaded ----------
api_key = os.getenv("OPENAI_API_KEY")
print(f"🔑 OPENAI_API_KEY: {api_key[:15] if api_key else 'NOT FOUND'}...")
print(f"📡 OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'NOT FOUND')}")
print(f"🤖 OPENAI_MODEL_NAME: {os.getenv('OPENAI_MODEL_NAME', 'NOT FOUND')}")

class Settings(BaseSettings):
    # Groq API
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "llama-3.3-70b-versatile")
    
    # LangFuse
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # Evaluation
    EVALUATION_MODEL: str = os.getenv("EVALUATION_MODEL", "llama-3.3-70b-versatile")
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", 3))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()