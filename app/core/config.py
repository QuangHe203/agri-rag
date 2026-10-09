from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    APP_ENV: str = "development"
    APP_TITLE: str = "Agriculture RAG API"
    APP_VERSION: str = "1.0.0"

    # Paths
    UPLOAD_DIR: str = "./data/uploads"
    VECTOR_DB_PATH: str = "./data/vector_db"
    SQLITE_DB_PATH: str = "./data/documents.db"

    # Collection
    COLLECTION_NAME: str = "agriculture_knowledge"

    # Embedding
    EMBEDDING_MODEL: str = "AITeamVN/Vietnamese_Embedding"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_DIMENSION: int = 1024

    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    # Retrieval
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 10
    MIN_RELEVANCE_SCORE: float = 0.45

    # LLM
    LLM_PROVIDER: str = "gemini"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-flash-latest"
    LLM_TEMPERATURE: float = 0.1

    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = 50
    MAX_QUESTION_LENGTH: int = 2000

    # CORS
    CORS_ORIGINS: list[str] = ["*"]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
