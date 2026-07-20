"""Offline/opt-in live evaluation for Validation 2.0; never mutates production data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from ..config import settings
from ..llm_schemas import EvidenceValidationOutput
from .evidence_validator import _compatible_scores, derive_validation_decision
from .llm_client import LLMClientError, call_llm_json
from .prompt_loader import prompt_sha256


FIELDS = (
    "quote_traceability", "source_turn_match", "advisor_ownership",
    "context_sufficiency", "context_scope", "decision_relevance",
    "supported_decision_effect", "direction_support",
    "validated_evidence_strength", "rationale_grounding",
    "representative_dependency", "procedural_status", "contradiction_status",
    "semantic_consistency",
)
LIST_FIELDS = ("unsupported_rationale_claims", "unsupported_escalations", "consistency_issues")
ESCALATIONS = (
    "pain_to_move", "preference_to_requirement", "question_to_blocker",
    "representative_claim_to_advisor_claim", "scheduling_to_urgency",
    "interest_to_commitment", "discussion_to_approval_dependency",
    "current_state_to_future_condition",
)


@dataclass(frozen=True)
class EvalCandidate:
    item_type: str
    advisor_quote: str
    rationale: str


@dataclass(frozen=True)
class EvaluationExample:
    example_id: str
    split: str
    item_type: str
    advisor_quote: str
    rationale: str
    supporting_turns: tuple[dict[str, Any], ...]
    deterministic_facts: dict[str, Any]
    expert: dict[str, Any]
    evidence_factor_validity: str
    rationale_validity: str
    difficulty: str
    expert_explanation: str
    precheck_verdict: str | None
    precheck_reason: str | None


def _expert(item_type: str, verdict: str, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "quote_traceability": "exact", "source_turn_match": "exact",
        "advisor_ownership": "reliable_advisor", "context_sufficiency": "sufficient",
        "context_scope": "quote_only", "decision_relevance": "material",
        "supported_decision_effect": "increases_move_likelihood" if item_type == "driver" else "decreases_move_likelihood",
        "direction_support": "supports_driver" if item_type == "driver" else "supports_blocker",
        "validated_evidence_strength": "explicit", "rationale_grounding": "fully_grounded",
        "unsupported_rationale_claims": [], "representative_dependency": "independent",
        "procedural_status": "substantive_factor", "unsupported_escalations": [],
        "contradiction_status": "none", "semantic_consistency": "consistent",
        "consistency_issues": [], "expected_verdict": verdict,
    }
    values.update(overrides)
    return values


def _scenario(key: str, verdict: str, item_type: str, quotes: list[str], rationale: str,
              *, overrides: dict[str, Any] | None = None, evidence: str = "valid",
              rationale_validity: str = "valid", precheck: tuple[str, str] | None = None,
              difficulty: str = "medium", explanation: str = "Expert-labeled recruiting evidence relationship.") -> dict[str, Any]:
    return locals()


SCENARIOS = [
    _scenario("explicit_value", "pass", "driver", ["The stronger service platform is a real reason for me to move.", "That client support makes me want to join.", "The technology advantage increases my willingness to switch.", "I would move for that level of operational help.", "That offering is valuable enough to influence my decision."], "The advisor explicitly identifies value that increases willingness to move."),
    _scenario("implied_value", "pass", "driver", ["I am stretched too thin. That is why I am looking for a firm with deeper support.", "My team cannot keep absorbing this work, so I am considering a better-supported platform.", "I need time back for clients; that is what is driving this search.", "The workload is unsustainable, which is why a stronger operating model appeals to me.", "I want to leave the administrative burden behind by moving to a firm with support."], "Contiguous advisor language directly links the desired outcome to moving.", overrides={"validated_evidence_strength": "tightly_implied", "context_scope": "local_advisor_thought"}, difficulty="hard"),
    _scenario("explicit_blocker", "pass", "blocker", ["I cannot proceed unless the transition costs are capped.", "We will not move if client fees increase.", "The deal cannot happen without digital record access.", "I need the contract restriction removed before proceeding.", "We cannot transition unless the platform supports our required workflow."], "The advisor states an explicit condition that decreases move likelihood."),
    _scenario("implied_blocker", "pass", "blocker", ["Our agreement runs through December. We have to clear it before any transition.", "Compliance must release the records first; only then can we move.", "My partner has final authority, so nothing proceeds before her approval.", "The integration has to work for our client process before a switch is possible.", "The loan covenant must be resolved ahead of a move."], "Contiguous advisor wording establishes a required dependency.", overrides={"validated_evidence_strength": "tightly_implied", "context_scope": "local_advisor_thought"}, difficulty="hard"),
    _scenario("commitment", "pass", "driver", ["We are moving forward with Optimize.", "I have decided to make the move.", "This is the direction I am taking.", "We are ready to proceed with the transition.", "I am committing to the switch."], "The advisor makes an explicit positive commitment."),
    _scenario("timing_dependency", "pass", "blocker", ["I cannot leave before my contract expires in September.", "The move has to wait until the retention period ends.", "We cannot proceed until the tax year closes.", "Nothing can happen before my licensing transfer is approved.", "The transition must wait for the current lease to expire."], "The advisor states a material timing dependency.", overrides={"supported_decision_effect": "creates_timing_dependency", "direction_support": "supports_timing_blocker"}),
    _scenario("pain_linked", "pass", "driver", ["I am working every weekend, and that is why I want a firm with more support.", "The administrative load is pushing me to look elsewhere.", "I am exhausted by the current model, so I want to change firms.", "This service burden is the reason I am considering a move.", "I cannot keep operating this way; I am seeking a supported platform."], "The advisor explicitly links current pain to transition motivation."),
    _scenario("weak_implication", "needs_review", "driver", ["The workload has become heavy.", "I spend too much time on administration.", "Client service takes nearly every evening.", "Our systems create extra work.", "The current setup feels inefficient."], "The evidence describes pain but does not directly connect it to moving.", overrides={"decision_relevance": "weak", "validated_evidence_strength": "weakly_implied"}, evidence="ambiguous", rationale_validity="partially_valid", difficulty="hard"),
    _scenario("partial_quote", "needs_review", "driver", ["More support would help us", "A better platform matters", "That would give me time", "The service model appeals", "This could improve things"], "The excerpt appears relevant but is only partially traceable.", overrides={"quote_traceability": "partial"}, evidence="ambiguous"),
    _scenario("ambiguous_source", "needs_review", "blocker", ["That has to be resolved first.", "We need that approval.", "It cannot happen before then.", "That issue may stop us.", "This must be available."], "The evidence may state a condition, but repeated wording makes source resolution ambiguous.", overrides={"source_turn_match": "ambiguous"}, evidence="ambiguous", difficulty="hard"),
    _scenario("incomplete_context", "needs_review", "driver", ["That is the main benefit for me.", "This is why I am interested.", "That outcome matters to my decision.", "It could be a reason to switch.", "This would make the change worthwhile."], "The excerpt is potentially material but its referent is missing.", overrides={"context_sufficiency": "incomplete"}, evidence="ambiguous", difficulty="hard"),
    _scenario("partial_rationale", "needs_review", "blocker", ["The conversion cost concerns me.", "I am uneasy about the transition expense.", "Those fees are a significant concern.", "The implementation cost gives me pause.", "I worry about what the move will cost."], "The rationale correctly identifies cost concern but overstates it as mandatory.", overrides={"rationale_grounding": "partially_grounded", "unsupported_rationale_claims": ["cost is a mandatory condition"]}, rationale_validity="partially_valid", difficulty="hard"),
    _scenario("unresolved_contradiction", "needs_review", "driver", ["I see real value, although I am still uncertain.", "I want the support, but I have not decided.", "The platform is compelling, though I remain hesitant.", "This makes a move attractive, but I need more clarity.", "I like the direction while still having reservations."], "The advisor expresses value and unresolved uncertainty.", overrides={"contradiction_status": "unresolved"}, evidence="ambiguous", difficulty="hard"),
    _scenario("procedural", "reject", "driver", ["I will review the packet this weekend.", "Send the deck and I will read it.", "Let us schedule another conversation.", "I will take a few days to think.", "Please circle back after I review the notes."], "The statement describes process, not a recruiting factor.", overrides={"decision_relevance": "none", "supported_decision_effect": "neutral", "direction_support": "unsupported", "validated_evidence_strength": "unsupported", "rationale_grounding": "unsupported", "procedural_status": "procedural_only"}, evidence="invalid", rationale_validity="invalid", precheck=("reject", "polite_or_procedural")),
    _scenario("question", "reject", "blocker", ["Does your system connect to Salesforce?", "What are your platform fees?", "Can client records be exported?", "Do partners have to approve this?", "Is there a minimum production level?"], "The advisor asks for information without stating a dependency.", overrides={"decision_relevance": "none", "supported_decision_effect": "indeterminate", "direction_support": "unsupported", "validated_evidence_strength": "unsupported", "rationale_grounding": "unsupported", "unsupported_escalations": ["question_to_blocker"]}, evidence="invalid", rationale_validity="invalid", precheck=("reject", "question_to_blocker"), difficulty="hard"),
    _scenario("preference_requirement", "reject", "blocker", ["I prefer electronic compliance files.", "I would rather use one client portal.", "My preference is consolidated reporting.", "I like having mobile access.", "I favor a paperless workflow."], "The rationale turns a preference into a mandatory transition requirement.", overrides={"direction_support": "unsupported", "supported_decision_effect": "indeterminate", "rationale_grounding": "unsupported", "unsupported_rationale_claims": ["preference is mandatory"], "unsupported_escalations": ["preference_to_requirement"]}, evidence="invalid", rationale_validity="invalid", precheck=("reject", "preference_to_requirement"), difficulty="hard"),
    _scenario("pain_unlinked", "reject", "driver", ["I work constantly.", "The administration is exhausting.", "Our current process is frustrating.", "I spend every evening catching up.", "The workload is difficult."], "The rationale invents motivation to change firms from unlinked current pain.", overrides={"decision_relevance": "weak", "supported_decision_effect": "indeterminate", "direction_support": "unsupported", "validated_evidence_strength": "unsupported", "rationale_grounding": "unsupported", "unsupported_rationale_claims": ["pain motivates moving"], "unsupported_escalations": ["pain_to_move"]}, evidence="invalid", rationale_validity="invalid", precheck=("reject", "pain_to_move"), difficulty="hard"),
    _scenario("representative_led", "reject", "driver", ["That sounds reasonable.", "Okay, I understand.", "Sure, that makes sense.", "I see what you mean.", "That is interesting."], "The rationale attributes the representative's claimed value to the advisor.", overrides={"decision_relevance": "none", "supported_decision_effect": "neutral", "direction_support": "unsupported", "validated_evidence_strength": "unsupported", "rationale_grounding": "unsupported", "representative_dependency": "fully_dependent", "unsupported_escalations": ["representative_claim_to_advisor_claim"]}, evidence="invalid", rationale_validity="invalid", precheck=("reject", "polite_or_procedural")),
    _scenario("acknowledgement", "reject", "driver", ["Okay.", "Sure.", "Got it.", "Understood.", "Makes sense."], "The rationale treats acknowledgement as recruiting intent.", overrides={"decision_relevance": "none", "supported_decision_effect": "neutral", "direction_support": "unsupported", "validated_evidence_strength": "unsupported", "rationale_grounding": "unsupported"}, evidence="invalid", rationale_validity="invalid", precheck=("reject", "polite_or_procedural")),
    _scenario("interest_commitment", "reject", "driver", ["This is interesting.", "I am curious about the approach.", "The idea has my attention.", "I would like to learn more.", "This sounds worth exploring."], "The rationale escalates preliminary interest into commitment.", overrides={"decision_relevance": "weak", "supported_decision_effect": "indeterminate", "direction_support": "unsupported", "validated_evidence_strength": "weakly_implied", "rationale_grounding": "unsupported", "unsupported_escalations": ["interest_to_commitment"]}, evidence="invalid", rationale_validity="invalid", difficulty="hard"),
    _scenario("partner_discussion", "reject", "blocker", ["I will discuss it with my partner.", "My partner and I should talk about this.", "I want to run the idea by my spouse.", "We will review this together.", "I need to speak with the other owner."], "The rationale invents a required approval dependency from ordinary discussion.", overrides={"supported_decision_effect": "indeterminate", "direction_support": "unsupported", "validated_evidence_strength": "unsupported", "rationale_grounding": "unsupported", "procedural_status": "procedural_only", "unsupported_escalations": ["discussion_to_approval_dependency"]}, evidence="invalid", rationale_validity="invalid", precheck=("reject", "discussion_to_approval_dependency"), difficulty="hard"),
    _scenario("scheduling", "reject", "driver", ["I will call Monday.", "Let us reconnect next week.", "I can meet tomorrow afternoon.", "Put a follow-up on the calendar.", "I will get back to you Friday."], "The rationale invents urgency from ordinary scheduling.", overrides={"decision_relevance": "none", "supported_decision_effect": "neutral", "direction_support": "unsupported", "validated_evidence_strength": "unsupported", "rationale_grounding": "unsupported", "procedural_status": "procedural_only", "unsupported_escalations": ["scheduling_to_urgency"]}, evidence="invalid", rationale_validity="invalid", precheck=("reject", "scheduling_to_urgency")),
    _scenario("direct_contradiction", "reject", "driver", ["Actually, we are not moving forward.", "I have decided against making the change.", "We will remain with the current firm.", "I no longer intend to proceed.", "The transition is off."], "The candidate contradicts the advisor's direct refusal.", overrides={"supported_decision_effect": "decreases_move_likelihood", "direction_support": "contradicts_candidate_type", "rationale_grounding": "contradicts_evidence", "contradiction_status": "direct_contradiction"}, evidence="invalid", rationale_validity="invalid"),
    _scenario("unsupported_rationale", "reject", "blocker", ["The technology is different.", "Their process uses another system.", "The reporting format has changed.", "The platform operates differently.", "The workflow is unfamiliar."], "The rationale claims the difference prevents moving, which the advisor never states.", overrides={"decision_relevance": "none", "supported_decision_effect": "indeterminate", "direction_support": "unsupported", "validated_evidence_strength": "unsupported", "rationale_grounding": "unsupported", "unsupported_rationale_claims": ["difference prevents moving"], "unsupported_escalations": ["current_state_to_future_condition"]}, evidence="invalid", rationale_validity="invalid", difficulty="hard"),
]


def build_dataset() -> list[EvaluationExample]:
    examples: list[EvaluationExample] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for variant, quote in enumerate(scenario["quotes"], start=1):
            split = "holdout" if variant == 5 or (variant == 4 and scenario_index < 16) else "development"
            item_type = scenario["item_type"]
            expert = _expert(item_type, scenario["verdict"], **(scenario["overrides"] or {}))
            facts = {
                "quote_traceability": expert["quote_traceability"],
                "source_turn_match": expert["source_turn_match"],
                "advisor_ownership": expert["advisor_ownership"],
                "ownership_turn_id": scenario_index * 10 + variant,
                "context_turn_ids": [scenario_index * 10 + variant],
            }
            turn = {"turn_id": facts["ownership_turn_id"], "turn_index": 0, "timestamp": "00:00:00", "raw_speaker_label": "Advisor", "inferred_role": "advisor", "role_confidence": 0.99, "text": quote}
            precheck = scenario["precheck"]
            examples.append(EvaluationExample(
                example_id=f"v2-{scenario['key']}-{variant:02d}", split=split,
                item_type=item_type, advisor_quote=quote, rationale=scenario["rationale"],
                supporting_turns=(turn,), deterministic_facts=facts, expert=expert,
                evidence_factor_validity=scenario["evidence"],
                rationale_validity=scenario["rationale_validity"],
                difficulty=scenario["difficulty"], expert_explanation=scenario["explanation"],
                precheck_verdict=precheck[0] if precheck else None,
                precheck_reason=precheck[1] if precheck else None,
            ))
    return examples


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def categorical_metrics(expected: list[str], predicted: list[str | None]) -> dict[str, Any]:
    if not expected:
        return {"count": 0, "accuracy": 0.0, "macro_precision": 0.0,
                "macro_recall": 0.0, "macro_f1": 0.0, "missing_rate": 0.0,
                "per_class": {}, "confusion_matrix": {}}
    labels = sorted(set(expected) | {value for value in predicted if value is not None})
    matrix = {actual: {guess: 0 for guess in labels + ["__missing__"]} for actual in labels}
    for actual, guess in zip(expected, predicted, strict=True):
        matrix[actual][guess or "__missing__"] += 1
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in labels if actual != label)
        fn = sum(count for guess, count in matrix[label].items() if guess != label)
        precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
        per_class[label] = {"precision": precision, "recall": recall, "f1": _safe_div(2 * precision * recall, precision + recall)}
    return {
        "count": len(expected), "accuracy": _safe_div(sum(a == b for a, b in zip(expected, predicted, strict=True)), len(expected)),
        "macro_precision": mean(item["precision"] for item in per_class.values()),
        "macro_recall": mean(item["recall"] for item in per_class.values()),
        "macro_f1": mean(item["f1"] for item in per_class.values()),
        "missing_rate": _safe_div(sum(value is None for value in predicted), len(predicted)),
        "per_class": per_class, "confusion_matrix": matrix,
    }


def list_metrics(expected: list[list[str]], predicted: list[list[str]]) -> dict[str, Any]:
    if not expected:
        return {"exact_match": 0.0, "micro_precision": 0.0,
                "micro_recall": 0.0, "micro_f1": 0.0,
                "false_detections": 0, "missed_detections": 0}
    tp = fp = fn = 0
    exact = 0
    for actual, guess in zip(expected, predicted, strict=True):
        a, g = set(actual), set(guess)
        exact += a == g; tp += len(a & g); fp += len(g - a); fn += len(a - g)
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return {"exact_match": _safe_div(exact, len(expected)), "micro_precision": precision,
            "micro_recall": recall, "micro_f1": _safe_div(2 * precision * recall, precision + recall),
            "false_detections": fp, "missed_detections": fn}


def validate_dataset(examples: list[EvaluationExample]) -> dict[str, Any]:
    ids = [item.example_id for item in examples]
    required = set(FIELDS + LIST_FIELDS + ("expected_verdict",))
    assert len(examples) == 120 and len(ids) == len(set(ids))
    assert all(required <= set(item.expert) for item in examples)
    assert not ({item.example_id for item in examples if item.split == "development"} & {item.example_id for item in examples if item.split == "holdout"})
    return {"count": len(examples), "split": Counter(item.split for item in examples),
            "verdict": Counter(item.expert["expected_verdict"] for item in examples)}


def oracle_report(examples: list[EvaluationExample]) -> dict[str, Any]:
    predictions = []
    for item in examples:
        output = EvidenceValidationOutput(**item.expert)
        predictions.append(derive_validation_decision(output, EvalCandidate(item.item_type, item.advisor_quote, item.rationale)).verdict)
    return categorical_metrics([item.expert["expected_verdict"] for item in examples], predictions)


def precheck_report(examples: list[EvaluationExample]) -> dict[str, Any]:
    expected_rejects = sum(item.expert["expected_verdict"] == "reject" for item in examples)
    resolved = [item for item in examples if item.precheck_verdict is not None]
    correct = sum(item.precheck_verdict == item.expert["expected_verdict"] for item in resolved)
    false_rejects = [item.example_id for item in resolved if item.precheck_verdict == "reject" and item.expert["expected_verdict"] != "reject"]
    return {"resolved_count": len(resolved), "resolved_rate": _safe_div(len(resolved), len(examples)),
            "hard_reject_precision": _safe_div(correct, len(resolved)), "hard_reject_recall": _safe_div(correct, expected_rejects),
            "model_calls_avoided": len(resolved), "false_rejects": false_rejects,
            "reasons": Counter(item.precheck_reason for item in resolved)}


def _payload(item: EvaluationExample) -> dict[str, Any]:
    return {"evaluation_example_id": item.example_id,
            "candidate": {"item_type": item.item_type, "advisor_quote": item.advisor_quote,
                          "rationale": item.rationale, "source_turn_ids": item.deterministic_facts["context_turn_ids"]},
            "supporting_turns": list(item.supporting_turns), "deterministic_facts": item.deterministic_facts}


def run_live(examples: list[EvaluationExample], split: str, output_path: Path, repeats: int = 1,
             limit: int | None = None, prompt_file_name: str = "03_evidence_validator.txt") -> dict[str, Any]:
    if prompt_file_name != "03_evidence_validator.txt" and split != "development":
        raise ValueError("Experimental prompts are restricted to the development split")
    selected = [item for item in examples if item.split == split]
    if limit == 12:
        selected = [item for verdict in ("pass", "needs_review", "reject")
                    for item in [candidate for candidate in selected if candidate.expert["expected_verdict"] == verdict][:4]]
    elif limit is not None and len(selected) > limit:
        selected = [selected[index * len(selected) // limit] for index in range(limit)]
    results = []
    for item in selected:
        attempts = []
        for _ in range(repeats):
            try:
                output = call_llm_json(prompt_file_name=prompt_file_name, input_payload=_payload(item),
                                       response_model=EvidenceValidationOutput, model=settings.evidence_validator_model,
                                       temperature=0, transcript_id=None)
                decision = derive_validation_decision(output, EvalCandidate(item.item_type, item.advisor_quote, item.rationale))
                clamped = _compatible_scores(output, decision)
                attempts.append({"output": output.model_dump(), "derived_verdict": decision.verdict,
                                 "reason": decision.reason, "clamped_scores": clamped, "error": None})
            except LLMClientError as error:
                attempts.append({"output": None, "derived_verdict": "reject", "reason": "malformed_or_unavailable_model_output", "clamped_scores": None, "error": str(error)})
        results.append({"example_id": item.example_id, "attempts": attempts})
    document = {"dataset": validate_dataset(examples), "split": split, "model": settings.evidence_validator_model,
                "prompt_version": "validation_2.0" if prompt_file_name == "03_evidence_validator.txt" else "validation_2.0_review_experiment_v1",
                "prompt_file_name": prompt_file_name, "prompt_sha256": prompt_sha256(prompt_file_name),
                "store": False, "temperature": 0, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document


def analyze_live(examples: list[EvaluationExample], document: dict[str, Any]) -> dict[str, Any]:
    lookup = {item.example_id: item for item in examples}
    rows = [(lookup[row["example_id"]], row["attempts"][0]) for row in document["results"]]
    valid_rows = [(item, attempt) for item, attempt in rows if attempt["output"] is not None]
    fields = {field: categorical_metrics([item.expert[field] for item, _ in valid_rows], [attempt["output"].get(field) for _, attempt in valid_rows]) for field in FIELDS}
    lists = {field: list_metrics([item.expert[field] for item, _ in valid_rows], [attempt["output"].get(field, []) for _, attempt in valid_rows]) for field in LIST_FIELDS}
    verdict = categorical_metrics([item.expert["expected_verdict"] for item, _ in rows], [attempt["derived_verdict"] for _, attempt in rows])
    errors = [{"example_id": item.example_id, "expected": item.expert["expected_verdict"], "predicted": attempt["derived_verdict"],
               "item_type": item.item_type, "evidence_strength": item.expert["validated_evidence_strength"],
               "expert_effect": item.expert["supported_decision_effect"],
               "predicted_effect": attempt["output"].get("supported_decision_effect") if attempt["output"] else None,
               "incorrect_fields": [field for field in FIELDS if attempt["output"] and attempt["output"].get(field) != item.expert[field]],
               "precheck": item.precheck_reason, "rationale_validity": item.rationale_validity}
              for item, attempt in rows if attempt["derived_verdict"] != item.expert["expected_verdict"]]
    adjustments = []
    raw_scores: list[tuple[float, float, float]] = []
    clamped_scores: list[tuple[float, float, float]] = []
    for _, attempt in valid_rows:
        raw = attempt["output"]
        raw_tuple = (raw["support_score"], raw["advisor_side_score"], raw["false_positive_risk"])
        clamped_tuple = tuple(attempt["clamped_scores"])
        raw_scores.append(raw_tuple); clamped_scores.append(clamped_tuple)
        for before, after in zip(raw_tuple, clamped_tuple, strict=True):
            adjustments.append(abs(before - after))
    strength = {}
    for label in ("explicit", "tightly_implied", "weakly_implied", "unsupported"):
        group = [(item, attempt) for item, attempt in rows if item.expert["validated_evidence_strength"] == label]
        strength[label] = {"count": len(group), "verdict_accuracy": _safe_div(sum(item.expert["expected_verdict"] == attempt["derived_verdict"] for item, attempt in group), len(group)),
                           "pass_recall": _safe_div(sum(attempt["derived_verdict"] == "pass" for item, attempt in group if item.expert["expected_verdict"] == "pass"), sum(item.expert["expected_verdict"] == "pass" for item, _ in group))}
    escalation = {}
    for label in ESCALATIONS:
        expected_positive = sum(label in item.expert["unsupported_escalations"] for item, _ in valid_rows)
        predicted_positive = sum(label in attempt["output"].get("unsupported_escalations", []) for _, attempt in valid_rows)
        true_positive = sum(label in item.expert["unsupported_escalations"] and label in attempt["output"].get("unsupported_escalations", []) for item, attempt in valid_rows)
        escalation[label] = {"expected": expected_positive, "predicted": predicted_positive,
                             "precision": _safe_div(true_positive, predicted_positive), "recall": _safe_div(true_positive, expected_positive)}
    rationale_cases = [{"example_id": item.example_id, "item_type": item.item_type,
                        "evidence_strength": item.expert["validated_evidence_strength"],
                        "rationale_validity": item.rationale_validity,
                        "predicted_verdict": attempt["derived_verdict"],
                        "rationale_grounding": attempt["output"].get("rationale_grounding") if attempt["output"] else None,
                        "underlying_factor_would_otherwise_pass": item.evidence_factor_validity == "valid"}
                       for item, attempt in rows if item.evidence_factor_validity == "valid" and item.rationale_validity != "valid" and attempt["derived_verdict"] == "reject"]
    conflict_counts = Counter()
    for _, attempt in valid_rows:
        output = attempt["output"]
        if output["rationale_grounding"] == "fully_grounded" and output.get("unsupported_rationale_claims"): conflict_counts["fully_grounded_with_unsupported_claims"] += 1
        if output["procedural_status"] == "procedural_only" and output["decision_relevance"] == "material": conflict_counts["procedural_with_material_relevance"] += 1
        if output["advisor_ownership"] == "reliable_advisor" and output["representative_dependency"] == "fully_dependent": conflict_counts["advisor_owned_fully_representative_dependent"] += 1
        if output["supported_decision_effect"] == "neutral" and output["direction_support"] == "supports_driver": conflict_counts["neutral_with_driver_support"] += 1
    raw_order = sorted(range(len(raw_scores)), key=lambda i: (-raw_scores[i][0], -raw_scores[i][1], raw_scores[i][2]))
    clamped_order = sorted(range(len(clamped_scores)), key=lambda i: (-clamped_scores[i][0], -clamped_scores[i][1], clamped_scores[i][2]))
    return {"count": len(rows), "malformed_count": len(rows) - len(valid_rows), "fields": fields,
            "lists": lists, "verdict": verdict, "errors": errors,
            "clamping": {"values_changed": sum(value > 0 for value in adjustments),
                         "mean_adjustment": mean(adjustments) if adjustments else 0,
                         "max_adjustment": max(adjustments, default=0),
                         "raw_mean": [mean(values) for values in zip(*raw_scores)] if raw_scores else [],
                         "clamped_mean": [mean(values) for values in zip(*clamped_scores)] if clamped_scores else [],
                         "downstream_ties": len(clamped_scores) - len(set(clamped_scores)),
                         "ranking_positions_changed": sum(a != b for a, b in zip(raw_order, clamped_order, strict=True))},
            "evidence_strength": strength, "escalations": escalation,
            "rationale_evidence_rejects": rationale_cases,
            "consistency_conflicts": dict(conflict_counts),
            "invalid_candidates_passed": sum(item.evidence_factor_validity == "invalid" and attempt["derived_verdict"] == "pass" for item, attempt in rows),
            "representative_led_passed": sum("representative_claim_to_advisor_claim" in item.expert["unsupported_escalations"] and attempt["derived_verdict"] == "pass" for item, attempt in rows),
            "procedural_passed": sum(item.expert["procedural_status"] == "procedural_only" and attempt["derived_verdict"] == "pass" for item, attempt in rows)}


def write_markdown(report: dict[str, Any], path: Path) -> None:
    verdict = report.get("verdict", {})
    lines = ["# Validation 2.0 Live Evaluation", "", f"Examples: {report.get('count', 0)}",
             f"Malformed: {report.get('malformed_count', 0)}", f"Verdict accuracy: {verdict.get('accuracy', 0):.3f}",
             f"Verdict macro F1: {verdict.get('macro_f1', 0):.3f}", "", "## Field accuracy", "",
             "| Field | Accuracy | Macro F1 | Missing |", "|---|---:|---:|---:|"]
    for field, metrics in report.get("fields", {}).items():
        lines.append(f"| {field} | {metrics['accuracy']:.3f} | {metrics['macro_f1']:.3f} | {metrics['missing_rate']:.3f} |")
    lines.extend(["", "## Verdict errors", ""])
    for error in report.get("errors", []):
        lines.append(f"- {error['example_id']}: expected {error['expected']}, predicted {error['predicted']}; fields={','.join(error['incorrect_fields'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "oracle", "live-development", "live-holdout", "stability", "experiment-v1", "experiment-stability", "report"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs/validation-v2"))
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(); examples = build_dataset(); validate_dataset(examples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "validate": result = validate_dataset(examples)
    elif args.mode == "oracle": result = {"oracle": oracle_report(examples), "prechecks": precheck_report(examples)}
    elif args.mode in {"live-development", "live-holdout", "stability", "experiment-v1", "experiment-stability"}:
        split = "development" if args.mode != "live-holdout" else "holdout"
        repeats = 3 if args.mode in {"stability", "experiment-stability"} else 1
        limit = 12 if repeats == 3 else None
        prompt = "03_evidence_validator_review_experiment.txt" if args.mode.startswith("experiment") else "03_evidence_validator.txt"
        result = run_live(examples, split, args.output_dir / f"{args.mode}.json", repeats, limit, prompt)
    else:
        if args.input is None: raise SystemExit("--input is required for report")
        source = json.loads(args.input.read_text(encoding="utf-8")); result = analyze_live(examples, source)
        (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_markdown(result, args.output_dir / "analysis.md")
    print(json.dumps(result, indent=2, default=lambda value: dict(value)))


if __name__ == "__main__":
    main()
