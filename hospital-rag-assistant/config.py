"""
config.py

Centralized configuration for the Hospital RAG Assistant.

Loads all runtime settings (database credentials, Groq API key, model names,
file paths) from environment variables using python-dotenv. Every other
module imports the single `settings` instance from here instead of reading
os.environ directly, so configuration stays in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise EnvironmentError(
            f"Missing required environment variable '{name}'. "
            f"Copy .env.example to .env and fill in the value."
        )
    return value


@dataclass(frozen=True)
class Settings:
    # --- PostgreSQL (already-built hospital_db) ---
    db_host: str = field(default_factory=lambda: _get_env("DB_HOST", "localhost"))
    db_port: str = field(default_factory=lambda: _get_env("DB_PORT", "5432"))
    db_name: str = field(default_factory=lambda: _get_env("DB_NAME", "hospital_db"))
    db_user: str = field(default_factory=lambda: _get_env("DB_USER", "postgres"))
    db_password: str = field(default_factory=lambda: _get_env("DB_PASSWORD", "12345678"))

   
    groq_api_key: str = field(default_factory=lambda: _get_env("GROQ_API_KEY", required=True))
    groq_model: str = field(default_factory=lambda: _get_env("GROQ_MODEL", "llama-3.3-70b-versatile"))
    llm_temperature: float = field(default_factory=lambda: float(_get_env("LLM_TEMPERATURE", "0.0")))

    
    embedding_model: str = field(
        default_factory=lambda: _get_env(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

   
    chunk_size: int = field(default_factory=lambda: int(_get_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_get_env("CHUNK_OVERLAP", "120")))
    retrieval_top_k: int = field(default_factory=lambda: int(_get_env("RETRIEVAL_TOP_K", "4")))

    
    policies_dir: Path = field(default_factory=lambda: BASE_DIR / "policies")
    faiss_index_dir: Path = field(default_factory=lambda: BASE_DIR / "faiss_index")

    @property
    def sqlalchemy_uri(self) -> str:
        """Connection string consumed by langchain_community.utilities.SQLDatabase."""
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


settings = Settings()
