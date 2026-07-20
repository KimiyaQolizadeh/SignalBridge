import os
import sys
import types

import pytest

from backend.app.services.cross_encoder import (
    CrossEncoderConfig,
    CrossEncoderError,
    SentenceTransformersCrossEncoder,
)
from backend.scripts.evaluate_dedup_cross_encoder import (
    serialize_pair,
    split_pair_ids,
)
from backend.app.services.dedup_evaluation import LABELED_PAIRS


class FakeModel:
    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True


class FakeCrossEncoder:
    created: list[dict] = []

    def __init__(self, model_id: str, **kwargs: object) -> None:
        self.created.append({"model_id": model_id, **kwargs})
        self.model = FakeModel()

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
        return [2.0 if left == right else -2.0 for left, right in pairs]


def config() -> CrossEncoderConfig:
    return CrossEncoderConfig(
        model_id="local/test-reranker",
        revision="fixed-revision",
        device="cpu",
        batch_size=4,
        max_length=128,
        normalization="sigmoid",
        cache_dir="local-cache",
        local_files_only=True,
        trust_remote_code=False,
    )


def test_cross_encoder_runtime_is_imported_lazily() -> None:
    sys.modules.pop("sentence_transformers", None)
    import backend.app.services.cross_encoder  # noqa: F401

    assert "sentence_transformers" not in sys.modules


def test_adapter_passes_offline_cpu_and_length_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    FakeCrossEncoder.created.clear()

    scorer = SentenceTransformersCrossEncoder(config())

    created = FakeCrossEncoder.created[0]
    assert created["device"] == "cpu"
    assert created["max_length"] == 128
    assert created["local_files_only"] is True
    assert created["trust_remote_code"] is False
    assert scorer._model.model.eval_called is True


def test_batched_raw_and_normalized_scores_are_one_per_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    scorer = SentenceTransformersCrossEncoder(config())
    pairs = [("same", "same"), ("left", "right")]

    raw = scorer.score_pairs_raw(pairs)
    normalized = scorer.score_pairs(pairs)

    assert raw == [2.0, -2.0]
    assert len(normalized) == len(pairs)
    assert normalized[0] > normalized[1]


def test_invalid_or_offline_revision_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCrossEncoder:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise OSError("not cached")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FailingCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    with pytest.raises(CrossEncoderError, match="could not be loaded"):
        SentenceTransformersCrossEncoder(config())


def test_split_is_stable_balanced_and_disjoint() -> None:
    first_development, first_holdout = split_pair_ids()
    second_development, second_holdout = split_pair_ids()
    labels = {pair.pair_id: pair.duplicate for pair in LABELED_PAIRS}

    assert first_development == second_development
    assert first_holdout == second_holdout
    assert first_development.isdisjoint(first_holdout)
    assert len(first_development) == 36
    assert len(first_holdout) == 14
    assert sum(labels[pair_id] for pair_id in first_holdout) == 7


def test_serialization_supports_symmetry_and_preserves_evidence() -> None:
    pair = LABELED_PAIRS[0]
    forward = serialize_pair(pair, "B")
    reverse = serialize_pair(pair, "B", reverse=True)
    rationale_representation = serialize_pair(pair, "C")

    assert pair.evidence_a in forward[0]
    assert pair.evidence_b in forward[1]
    assert pair.evidence_b in reverse[0]
    assert pair.evidence_a in reverse[1]
    assert "Generated rationale" not in " ".join(forward)
    assert "Generated rationale" in " ".join(rationale_representation)


@pytest.mark.skipif(
    os.getenv("RUN_CROSS_ENCODER_INTEGRATION") != "1",
    reason="real local-model integration is opt-in",
)
def test_real_cached_base_model_runs_on_cpu() -> None:
    scorer = SentenceTransformersCrossEncoder(
        CrossEncoderConfig(
            model_id="BAAI/bge-reranker-base",
            revision="2cfc18c9415c912f9d8155881c133215df768a70",
            device="cpu",
            batch_size=2,
            max_length=512,
            normalization="sigmoid",
            cache_dir="data/model-cache",
            local_files_only=True,
            trust_remote_code=False,
        )
    )

    scores = scorer.score_pairs([("same factor", "same factor")])

    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0
