from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ExperimentSettings(BaseModel):
    dedup_mode: str
    dedup_cross_encoder_model: str
    importance_mode: str
    importance_model: str


class Settings(BaseSettings):
    openai_api_key: str | None = None
    database_url: str = (
        "postgresql+psycopg://signalbridge:signalbridge@localhost:5432/signalbridge"
    )
    app_env: str = "local"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_mb: int = 25
    speaker_classifier_model: str = "gpt-4.1-mini"
    candidate_extractor_model: str = "gpt-4.1-mini"
    evidence_validator_model: str = "gpt-4.1"
    business_scorer_model: str = "gpt-4.1-mini"
    final_reranker_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    dedup_similarity_threshold: float = Field(default=0.86, ge=0.0, le=1.0)
    dedup_experiment_mode: str = "existing_embedding_only"
    dedup_cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    dedup_cross_encoder_revision: str | None = None
    dedup_cross_encoder_device: str = "cpu"
    dedup_cross_encoder_batch_size: int = Field(default=16, ge=1)
    dedup_cross_encoder_max_length: int = Field(default=512, ge=32)
    dedup_cross_encoder_normalization: str = "sigmoid"
    dedup_cross_encoder_cache_dir: str | None = None
    dedup_cross_encoder_local_files_only: bool = False
    dedup_cross_encoder_trust_remote_code: bool = False
    dedup_cross_encoder_representation: str = "decision_factor_evidence"
    dedup_shortlist_threshold: float = Field(default=0.76, ge=0.0, le=1.0)
    dedup_cross_encoder_merge_threshold: float = Field(
        default=0.80, ge=0.0, le=1.0
    )
    dedup_cross_encoder_fallback_to_embedding: bool = False
    importance_mode: str = "importance_disabled"
    importance_model: str = "gpt-4.1-mini"

    @property
    def experiments(self) -> ExperimentSettings:
        return ExperimentSettings(
            dedup_mode=self.dedup_experiment_mode,
            dedup_cross_encoder_model=self.dedup_cross_encoder_model,
            importance_mode=self.importance_mode,
            importance_model=self.importance_model,
        )

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
