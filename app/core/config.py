from functools import lru_cache
from pydantic import Field
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

    # Reranking (10.1)
    RERANK_ENABLED: bool = False
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_CANDIDATE_MULTIPLIER: int = 4   # fetch k*N candidates before reranking
    RERANK_MIN_SCORE: float = 0.0          # cross-encoder threshold (logit scale)

    # Hybrid search — BM25 + Vector (10.3)
    HYBRID_SEARCH_ENABLED: bool = False
    HYBRID_CANDIDATE_MULTIPLIER: int = 4   # fetch k*N candidates from each leg
    BM25_INDEX_PATH: str = "./data/bm25_index.pkl"

    # Query transformation — Query Rewrite / Multi-query / HyDE (10.4)
    # These run BEFORE retrieval and use the LLM. All default OFF so the
    # baseline pipeline is unchanged. When enabled, the extra query variants
    # (and/or HyDE documents) are fused with Reciprocal Rank Fusion.
    QUERY_REWRITE_ENABLED: bool = False    # rewrite the question into one cleaner search query
    MULTI_QUERY_ENABLED: bool = False      # generate several alternative phrasings
    MULTI_QUERY_COUNT: int = 3             # number of extra variants (excludes the base question)
    HYDE_ENABLED: bool = False             # embed a hypothetical answer passage for the dense leg
    QUERY_TRANSFORM_CANDIDATE_MULTIPLIER: int = 4  # fetch k*N candidates per query/leg before fusion

    # LLM
    LLM_PROVIDER: str = "gemini"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-flash-latest"
    LLM_TEMPERATURE: float = 0.1
    GEMINI_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0)
    GEMINI_MAX_RETRIES: int = Field(default=1, ge=0)

    # Query execution
    # Keep this low for local CPU inference; raise only after load-testing RAM
    # and verifying concurrent use of the shared embedding/vector-store clients.
    QUERY_MAX_CONCURRENCY: int = Field(default=1, ge=1)

    # Image extraction (response enrichment)
    IMAGE_EXTRACTION_ENABLED: bool = True
    IMAGE_STORE_PATH: str = "./data/images"
    MAX_IMAGES_PER_RESPONSE: int = 3

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
