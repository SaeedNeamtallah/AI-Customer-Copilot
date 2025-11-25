from pydantic_settings import BaseSettings 
from typing import Optional, List
from pathlib import Path
import json

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    FILE_MAX_SIZE: Optional[int]  # 10 MB
    FILE_ALLOWED_TYPES: str
    CHUNK_SIZE: Optional[int]  # characters
    CHUNK_OVERLAP: Optional[int]  # characters'
    MONGO_URI: str
    MONGO_DB_NAME: str
    # ========================= LLM Config =========================
    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str
    OPENAI_API_KEY: str
    OPENAI_API_URL: str
    COHERE_API_KEY: str
    GENERATION_MODEL_ID: str
    EMBEDDING_MODEL_ID: str
    EMBEDDING_MODEL_SIZE: int
    INPUT_DEFAULT_MAX_CHARACTERS: int
    GENERATION_DEFAULT_MAX_TOKENS: int
    GENERATION_DEFAULT_TEMPERATURE: float

    # ========================= RAG Config =========================
    RAG_MAX_PROMPT_LENGTH: int = 3000  # Maximum prompt length
    RAG_MAX_DOCUMENTS: int = 5  # Maximum number of documents

    # ========================= DB Config =========================
    POSTGRES_USERNAME: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_MAIN_DATABASE: str
    DATABASE_URL: str
    

    # vector store config
    VECTOR_DB_BACKEND : str
    VECTOR_DB_PATH : str
    VECTOR_DB_DISTANCE_METHOD: str = None
    PGVECTOR_INDEX_THRESHOLD: int = 100  # Minimum records before creating index

    @property
    def EMBEDDING_SIZE(self) -> int:
        """Alias for EMBEDDING_MODEL_SIZE"""
        return self.EMBEDDING_MODEL_SIZE

    @property
    def file_allowed_types_list(self) -> List[str]:
        """Parse FILE_ALLOWED_TYPES from string to list"""
        if isinstance(self.FILE_ALLOWED_TYPES, str):
            try:
                return json.loads(self.FILE_ALLOWED_TYPES)
            except json.JSONDecodeError:
                # Fallback: split by comma
                return [ext.strip() for ext in self.FILE_ALLOWED_TYPES.split(',')]
        return self.FILE_ALLOWED_TYPES

    class Config:
        # Get the path to the .env file in the src directory
        env_file = str(Path(__file__).resolve().parent.parent / ".env")
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()

