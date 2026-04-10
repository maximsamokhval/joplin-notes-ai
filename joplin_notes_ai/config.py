from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    joplin_token: str = Field(..., alias="JOPLIN_TOKEN")
    joplin_base_url: str = Field("http://localhost:41184", alias="JOPLIN_BASE_URL")

    llm_base_url: str = Field("https://api.deepseek.com/v1", alias="LLM_BASE_URL")
    llm_api_key: str = Field(..., alias="LLM_API_KEY")
    llm_model_name: str = Field("deepseek-chat", alias="LLM_MODEL_NAME")

    processed_tag: str = Field("ai-audited", alias="PROCESSED_TAG")
    prompt_file: str = Field("system_prompt.txt", alias="PROMPT_FILE")

    chroma_db_path: str = Field("./chroma_db", alias="CHROMA_DB_PATH")
    embedding_model_name: str = Field(
        "all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )
    similarity_threshold: float = Field(0.8, alias="SIMILARITY_THRESHOLD")
    similarity_top_k: int = Field(5, alias="SIMILARITY_TOP_K")

    request_timeout: int = Field(90, alias="REQUEST_TIMEOUT")
    llm_timeout: int = Field(90, alias="LLM_TIMEOUT")
    pause_between_notes: float = Field(1.5, alias="PAUSE_BETWEEN_NOTES")
    max_retries: int = Field(2, alias="MAX_RETRIES")
    retry_backoff_seconds: float = Field(0.8, alias="RETRY_BACKOFF_SECONDS")
    log_file: str = Field("logs/joplin_agent.log", alias="LOG_FILE")
