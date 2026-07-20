"""Run the authorized local dedup cross-encoder evaluation without app startup."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import psutil
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.cross_encoder import (  # noqa: E402
    CrossEncoderConfig,
    SentenceTransformersCrossEncoder,
    normalize_score,
)
from backend.app.services.dedup_evaluation import (  # noqa: E402
    LABELED_PAIRS,
    LabeledPair,
    calibrate_merge_threshold,
    classification_metrics,
)


MODEL_SPECS = {
    "BAAI/bge-reranker-base": {
        "revision": "2cfc18c9415c912f9d8155881c133215df768a70",
        "license": "MIT",
        "architecture": "XLM-RoBERTa sequence-classification reranker",
    },
    "BAAI/bge-reranker-v2-m3": {
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "license": "Apache-2.0",
        "architecture": "XLM-RoBERTa multilingual sequence-classification reranker",
    },
}
REPRESENTATIONS = ("A", "B", "C", "D")
ORDER_POLICIES = ("forward", "mean", "minimum")
PAIR_QUESTION = (
    "Do these candidates represent the same underlying recruiting decision "
    "factor, require substantially the same business response, and deserve one "
    "final signal slot?"
)


def _candidate_text(pair: LabeledPair, side: str, representation: str) -> str:
    item_type = getattr(pair, f"item_type_{side}")
    category = getattr(pair, f"category_{side}")
    evidence = getattr(pair, f"evidence_{side}")
    rationale = getattr(pair, f"rationale_{side}")
    direction = "supports move" if item_type == "driver" else "opposes move"
    if representation == "A":
        return f"Type: {item_type}\nCategory: {category}\nAdvisor evidence: {evidence}"
    if representation == "B":
        return (
            f"Type: {item_type}\nDecision direction: {direction}\n"
            f"Decision factor: {category}\nAdvisor evidence: {evidence}"
        )
    if representation == "C":
        return (
            f"Type: {item_type}\nDecision direction: {direction}\n"
            f"Decision factor: {category}\nAdvisor evidence: {evidence}\n"
            f"Generated rationale: {rationale}"
        )
    return f"Advisor evidence: {evidence}"


def serialize_pair(
    pair: LabeledPair, representation: str, *, reverse: bool = False
) -> tuple[str, str]:
    left_side, right_side = (("b", "a") if reverse else ("a", "b"))
    left = _candidate_text(pair, left_side, representation)
    right = _candidate_text(pair, right_side, representation)
    return (
        f"Candidate A:\n{left}\n\nQuestion: {PAIR_QUESTION}",
        f"Candidate B:\n{right}",
    )


def split_pair_ids() -> tuple[set[str], set[str]]:
    def ordered(duplicate: bool) -> list[LabeledPair]:
        rows = [pair for pair in LABELED_PAIRS if pair.duplicate is duplicate]
        return sorted(
            rows,
            key=lambda pair: hashlib.sha256(
                f"signalbridge-dedup-v1:{pair.pair_id}".encode()
            ).hexdigest(),
        )

    holdout = ordered(True)[:7] + ordered(False)[:7]
    holdout_ids = {pair.pair_id for pair in holdout}
    development_ids = {pair.pair_id for pair in LABELED_PAIRS} - holdout_ids
    return development_ids, holdout_ids


def _artifact_size(cache_dir: Path, model_id: str, revision: str) -> int:
    snapshot = cache_dir / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    return sum(path.stat().st_size for path in snapshot.rglob("*") if path.is_file())


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _score_in_chunks(
    scorer: SentenceTransformersCrossEncoder,
    pairs: list[tuple[str, str]],
    batch_size: int,
) -> tuple[list[float], list[float]]:
    raw_scores: list[float] = []
    per_pair_ms: list[float] = []
    for offset in range(0, len(pairs), batch_size):
        chunk = pairs[offset : offset + batch_size]
        started = time.perf_counter()
        raw_scores.extend(scorer.score_pairs_raw(chunk))
        elapsed = (time.perf_counter() - started) * 1000
        per_pair_ms.extend([elapsed / len(chunk)] * len(chunk))
    return raw_scores, per_pair_ms


def _token_stats(
    scorer: SentenceTransformersCrossEncoder,
    pairs: list[tuple[str, str]],
    max_length: int,
) -> dict:
    lengths = [
        len(scorer._model.tokenizer(left, right, truncation=False)["input_ids"])
        for left, right in pairs
    ]
    return {
        "minimum": min(lengths),
        "mean": statistics.fmean(lengths),
        "p95": _percentile([float(value) for value in lengths], 0.95),
        "maximum": max(lengths),
        "truncation_count": sum(value > max_length for value in lengths),
    }


def _metrics(labels: list[bool], scores: list[float], threshold: float) -> dict:
    result = classification_metrics(labels, scores, threshold)
    tn = int(result["true_negative"])
    fp = int(result["false_positive"])
    specificity = tn / (tn + fp) if tn + fp else 0.0
    result.update(
        specificity=specificity,
        false_positive_rate=1.0 - specificity,
        false_negative_rate=1.0 - float(result["recall"]),
        pr_auc=float(average_precision_score(labels, scores)),
    )
    return result


def evaluate_model(
    model_id: str,
    cache_dir: Path,
    batch_size: int,
    max_length: int,
) -> dict:
    spec = MODEL_SPECS[model_id]
    process = psutil.Process()
    memory_before = process.memory_info().rss
    load_started = time.perf_counter()
    scorer = SentenceTransformersCrossEncoder(
        CrossEncoderConfig(
            model_id=model_id,
            revision=spec["revision"],
            device="cpu",
            batch_size=batch_size,
            max_length=max_length,
            normalization="sigmoid",
            cache_dir=str(cache_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
    )
    load_seconds = time.perf_counter() - load_started
    memory_after_load = process.memory_info().rss
    development_ids, holdout_ids = split_pair_ids()
    labels_by_id = {pair.pair_id: pair.duplicate for pair in LABELED_PAIRS}
    model_result = {
        "model_id": model_id,
        "revision": spec["revision"],
        "license": spec["license"],
        "architecture": spec["architecture"],
        "device": "cpu",
        "max_length": max_length,
        "raw_output": "single relevance logit",
        "normalization": "stable sigmoid per direction",
        "artifact_size_bytes": _artifact_size(cache_dir, model_id, spec["revision"]),
        "load_seconds": load_seconds,
        "rss_delta_after_load_bytes": max(0, memory_after_load - memory_before),
        "development_ids": sorted(development_ids),
        "holdout_ids": sorted(holdout_ids),
        "representations": {},
    }
    for representation in REPRESENTATIONS:
        forward_inputs = [serialize_pair(pair, representation) for pair in LABELED_PAIRS]
        reverse_inputs = [
            serialize_pair(pair, representation, reverse=True) for pair in LABELED_PAIRS
        ]
        evaluation_started = time.perf_counter()
        forward_raw, forward_times = _score_in_chunks(scorer, forward_inputs, batch_size)
        reverse_raw, reverse_times = _score_in_chunks(scorer, reverse_inputs, batch_size)
        normalized_forward = [normalize_score(score, "sigmoid") for score in forward_raw]
        normalized_reverse = [normalize_score(score, "sigmoid") for score in reverse_raw]
        repeated = scorer.score_pairs_raw(forward_inputs[:2])
        repeat_max_difference = max(
            abs(left - right) for left, right in zip(forward_raw[:2], repeated, strict=True)
        )
        rows = []
        for index, pair in enumerate(LABELED_PAIRS):
            forward = normalized_forward[index]
            reverse = normalized_reverse[index]
            rows.append(
                {
                    **asdict(pair),
                    "forward_raw": forward_raw[index],
                    "reverse_raw": reverse_raw[index],
                    "forward_score": forward,
                    "reverse_score": reverse,
                    "absolute_order_difference": abs(forward - reverse),
                    "mean_score": (forward + reverse) / 2,
                    "minimum_score": min(forward, reverse),
                    "maximum_score": max(forward, reverse),
                }
            )
        representation_result = {
            "token_lengths": _token_stats(scorer, forward_inputs + reverse_inputs, max_length),
            "order_difference_mean": statistics.fmean(
                row["absolute_order_difference"] for row in rows
            ),
            "order_difference_maximum": max(
                row["absolute_order_difference"] for row in rows
            ),
            "repeat_raw_max_difference": repeat_max_difference,
            "mean_inference_ms_per_direction": statistics.fmean(
                forward_times + reverse_times
            ),
            "p50_inference_ms_per_direction": _percentile(
                forward_times + reverse_times, 0.50
            ),
            "p95_inference_ms_per_direction": _percentile(
                forward_times + reverse_times, 0.95
            ),
            "total_evaluation_seconds": time.perf_counter() - evaluation_started,
            "policies": {},
            "pairs": rows,
        }
        for policy in ORDER_POLICIES:
            score_field = {
                "forward": "forward_score",
                "mean": "mean_score",
                "minimum": "minimum_score",
            }[policy]
            development_rows = [row for row in rows if row["pair_id"] in development_ids]
            holdout_rows = [row for row in rows if row["pair_id"] in holdout_ids]
            selected = calibrate_merge_threshold(
                [labels_by_id[row["pair_id"]] for row in development_rows],
                [row[score_field] for row in development_rows],
                minimum_precision=0.95,
            )
            threshold = float(selected["threshold"])
            holdout_metrics = _metrics(
                [labels_by_id[row["pair_id"]] for row in holdout_rows],
                [row[score_field] for row in holdout_rows],
                threshold,
            )
            errors = [
                {
                    "pair_id": row["pair_id"],
                    "expected": row["duplicate"],
                    "predicted": row[score_field] >= threshold,
                    "score": row[score_field],
                    "threshold": threshold,
                    "candidate_a": f"{row['category_a']}: {row['evidence_a']}",
                    "candidate_b": f"{row['category_b']}: {row['evidence_b']}",
                    "difficulty": row["difficulty"],
                    "likely_cause": row["label_rationale"],
                }
                for row in holdout_rows
                if (row[score_field] >= threshold) is not row["duplicate"]
            ]
            representation_result["policies"][policy] = {
                "development_count": len(development_rows),
                "holdout_count": len(holdout_rows),
                "development_selected": selected,
                "holdout_metrics": holdout_metrics,
                "holdout_errors": errors,
            }
        model_result["representations"][representation] = representation_result
    model_result["rss_after_evaluation_bytes"] = process.memory_info().rss
    del scorer
    gc.collect()
    return model_result


def _write_markdown(report: dict, path: Path) -> None:
    lines = ["# SignalBridge Cross-Encoder Evaluation", ""]
    lines.append(
        f"Development: {report['split']['development_count']} pairs; "
        f"holdout: {report['split']['holdout_count']} pairs."
    )
    lines.extend(["", "| Model | Rep | Policy | Threshold | Precision | Recall | F1 | F0.5 | FP | FN |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for model in report["models"]:
        for representation, result in model["representations"].items():
            for policy, policy_result in result["policies"].items():
                metrics = policy_result["holdout_metrics"]
                lines.append(
                    f"| {model['model_id']} | {representation} | {policy} | "
                    f"{metrics['threshold']:.6f} | {metrics['precision']:.3f} | "
                    f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | "
                    f"{metrics['f0.5']:.3f} | {metrics['false_positive']} | "
                    f"{metrics['false_negative']} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=[*MODEL_SPECS, "all"], default=["all"])
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "model-cache")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "outputs" / "dedup-cross-encoder")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    selected_models = list(MODEL_SPECS) if "all" in args.models else args.models
    development_ids, holdout_ids = split_pair_ids()
    report = {
        "dataset": {"total": len(LABELED_PAIRS), "duplicates": 25, "distinct": 25},
        "split": {
            "strategy": "stable stratified 36-development/14-untouched-holdout split",
            "development_count": len(development_ids),
            "holdout_count": len(holdout_ids),
            "development_ids": sorted(development_ids),
            "holdout_ids": sorted(holdout_ids),
        },
        "models": [
            evaluate_model(model_id, args.cache_dir, args.batch_size, args.max_length)
            for model_id in selected_models
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "evaluation.json"
    markdown_path = args.output_dir / "evaluation.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, markdown_path)
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
