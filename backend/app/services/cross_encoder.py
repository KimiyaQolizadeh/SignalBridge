"""Optional, lazily loaded cross-encoder inference for deduplication experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class CrossEncoderError(Exception):
    """Raised when optional cross-encoder loading or inference fails."""


@dataclass(frozen=True)
class CrossEncoderConfig:
    model_id: str
    revision: str | None
    device: str
    batch_size: int
    max_length: int
    normalization: str
    cache_dir: str | None = None
    local_files_only: bool = False
    trust_remote_code: bool = False


class CrossEncoderScorer(Protocol):
    model_id: str
    revision: str | None

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def normalize_score(raw_score: float, strategy: str) -> float:
    if not math.isfinite(raw_score):
        raise CrossEncoderError("Cross-encoder returned a malformed score")
    if strategy == "sigmoid":
        if raw_score >= 0:
            value = 1.0 / (1.0 + math.exp(-raw_score))
        else:
            exponential = math.exp(raw_score)
            value = exponential / (1.0 + exponential)
    elif strategy == "identity":
        value = raw_score
    else:
        raise CrossEncoderError("Unsupported score normalization strategy")
    if not 0.0 <= value <= 1.0:
        raise CrossEncoderError("Normalized cross-encoder score is out of range")
    return value


class SentenceTransformersCrossEncoder:
    """Adapter imported only when an experimental cross-encoder mode is enabled."""

    def __init__(self, config: CrossEncoderConfig) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise CrossEncoderError(
                "Optional sentence-transformers runtime is not installed"
            ) from None
        try:
            self._model = CrossEncoder(
                config.model_id,
                revision=config.revision,
                device=config.device,
                max_length=config.max_length,
                cache_folder=config.cache_dir,
                local_files_only=config.local_files_only,
                trust_remote_code=config.trust_remote_code,
            )
        except Exception:
            raise CrossEncoderError("Cross-encoder model could not be loaded") from None
        self.model_id = config.model_id
        self.revision = config.revision
        self._batch_size = config.batch_size
        self._normalization = config.normalization
        self._model.model.eval()

    def score_pairs_raw(self, pairs: list[tuple[str, str]]) -> list[float]:
        try:
            import torch

            with torch.inference_mode():
                raw_scores = self._model.predict(
                    pairs,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    activation_fn=torch.nn.Identity(),
                )
            scores = [float(score) for score in raw_scores]
        except Exception:
            raise CrossEncoderError("Cross-encoder inference failed") from None
        if len(scores) != len(pairs):
            raise CrossEncoderError("Cross-encoder returned an unexpected score count")
        if any(not math.isfinite(score) for score in scores):
            raise CrossEncoderError("Cross-encoder returned a malformed score")
        return scores

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        try:
            scores = [
                normalize_score(float(score), self._normalization)
                for score in self.score_pairs_raw(pairs)
            ]
        except CrossEncoderError:
            raise
        except Exception:
            raise CrossEncoderError("Cross-encoder inference failed") from None
        if len(scores) != len(pairs):
            raise CrossEncoderError("Cross-encoder returned an unexpected score count")
        return scores


def load_cross_encoder(config: CrossEncoderConfig) -> CrossEncoderScorer:
    return SentenceTransformersCrossEncoder(config)
