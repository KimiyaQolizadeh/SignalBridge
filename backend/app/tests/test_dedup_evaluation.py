from backend.app.services.dedup_evaluation import (
    LABELED_PAIRS,
    calibrate_merge_threshold,
    classification_metrics,
)


def test_labeled_evaluation_set_is_balanced_and_complete() -> None:
    assert len(LABELED_PAIRS) == 50
    assert sum(pair.duplicate for pair in LABELED_PAIRS) == 25
    assert sum(not pair.duplicate for pair in LABELED_PAIRS) == 25
    assert all(pair.label_rationale for pair in LABELED_PAIRS)
    assert all(pair.difficulty for pair in LABELED_PAIRS)


def test_classification_metrics_include_precision_weighted_score() -> None:
    result = classification_metrics(
        [True, True, False, False], [0.95, 0.75, 0.7, 0.1], 0.8
    )

    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["false_positive"] == 0
    assert result["false_negative"] == 1
    assert 0.0 < result["f0.5"] < 1.0


def test_calibration_prefers_minimum_precision_then_recall() -> None:
    result = calibrate_merge_threshold(
        [True, True, False, False],
        [0.95, 0.8, 0.75, 0.1],
        minimum_precision=1.0,
    )

    assert result["threshold"] == 0.8
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
