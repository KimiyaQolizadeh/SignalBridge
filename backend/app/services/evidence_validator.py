import re
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from ..config import settings
from ..llm_schemas import EvidenceValidationOutput
from ..logging_config import get_logger
from ..models import CandidateSignal, SignalScore, Transcript, TranscriptTurn
from .llm_client import LLMClientError, call_llm_json
from .prompt_loader import load_prompt
from .run_persistence import persist_validation_diagnostics
from .speaker_classifier import MIN_ROLE_CONFIDENCE


POLITE_ACKNOWLEDGEMENTS = {
    "absolutely",
    "absolutely absolutely",
    "sure",
    "okay",
    "ok",
    "interesting",
    "that sounds interesting",
    "makes sense",
    "got it",
    "i see",
}
PROCEDURAL_PATTERNS = (
    r"\bsend me (?:more |the )?information\b",
    r"\bcircle back\b",
    r"\bconnect next week\b",
    r"\bschedule (?:a|another) call\b",
    r"\bset up (?:a|another) call\b",
    r"\bput time on(?: the calendar)?\b",
    r"\bmeet (?:again|(?:on )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    r"\bcall me next week\b",
)
PROCEDURAL_FILLER_WORDS = {
    "a", "again", "another", "absolutely", "am", "can", "could", "happy",
    "i", "i'd", "i'll", "is", "it", "let's", "me", "next", "okay", "on",
    "please", "sounds", "sure", "that", "the", "to", "we", "week", "will",
}
BACKGROUND_ONLY_PATTERNS = (
    r"^i(?:'ve| have) been (?:in )?(?:the )?(?:business|industry) for \d+ years$",
    r"^we(?:'ve| have) been (?:in )?(?:the )?(?:business|industry) for \d+ years$",
    r"^i have \d+ clients$",
    r"^we have \d+ clients$",
)
DECISION_LANGUAGE = (
    "move forward",
    "moving forward",
    "because",
    "need",
    "want",
    "prefer",
    "concern",
    "risk",
    "cost",
    "fee",
    "depend",
    "unless",
    "if ",
)
logger = get_logger(__name__)
MAX_CONTEXT_TURNS_PER_SIDE = 2
VALIDATION_DIAGNOSTICS: dict[int, list[dict]] = {}


@dataclass(frozen=True)
class ResolvedEvidenceTurn:
    turn: TranscriptTurn
    previous_turn: TranscriptTurn | None
    next_turn: TranscriptTurn | None
    resolution_method: str


@dataclass(frozen=True)
class ValidationDecision:
    verdict: str
    reason: str | None
    hard_failures: tuple[str, ...] = ()
    review_reasons: tuple[str, ...] = ()
    consistency_issues: tuple[str, ...] = ()


class TranscriptNotFoundError(Exception):
    """Raised when a requested transcript does not exist."""


class NoCandidateSignalsError(Exception):
    """Raised when evidence validation is requested before extraction."""


class EvidenceValidationError(Exception):
    """A safe error for validation or persistence failures."""


def _is_reliably_advisor_owned(turn: TranscriptTurn) -> bool:
    return (
        turn.inferred_role == "advisor"
        and turn.role_confidence is not None
        and turn.role_confidence >= MIN_ROLE_CONFIDENCE
    )


def _resolve_evidence_turn(
    candidate: CandidateSignal, turns: list[TranscriptTurn]
) -> ResolvedEvidenceTurn | None:
    quote = candidate.advisor_quote
    if not quote:
        return None

    index_by_id = {turn.id: index for index, turn in enumerate(turns)}
    source_ids = (
        candidate.source_turn_ids
        if isinstance(candidate.source_turn_ids, list)
        else []
    )
    source_matches = [
        turns[index_by_id[turn_id]]
        for turn_id in source_ids
        if turn_id in index_by_id
        and _is_reliably_advisor_owned(turns[index_by_id[turn_id]])
        and quote in turns[index_by_id[turn_id]].text
    ]
    matches = source_matches
    method = "source_turn_id"
    if not matches:
        matches = [
            turn for turn in turns
            if _is_reliably_advisor_owned(turn) and quote in turn.text
        ]
        method = "unique_global_match"

    if len(matches) > 1 and candidate.timestamp:
        timestamp_matches = [
            turn for turn in matches if turn.timestamp == candidate.timestamp
        ]
        if len(timestamp_matches) == 1:
            matches = timestamp_matches
            method = f"{method}_timestamp"

    if len(matches) != 1:
        return None

    turn = matches[0]
    index = index_by_id[turn.id]
    return ResolvedEvidenceTurn(
        turn=turn,
        previous_turn=turns[index - 1] if index > 0 else None,
        next_turn=turns[index + 1] if index + 1 < len(turns) else None,
        resolution_method=method,
    )


def _supporting_turns(
    resolved: ResolvedEvidenceTurn, turns: list[TranscriptTurn]
) -> list[TranscriptTurn]:
    """Build one bounded contiguous advisor thought; never cross ownership."""
    index_by_id = {turn.id: index for index, turn in enumerate(turns)}
    source_index = index_by_id[resolved.turn.id]
    selected = {source_index}
    for direction in (-1, 1):
        for distance in range(1, MAX_CONTEXT_TURNS_PER_SIDE + 1):
            index = source_index + direction * distance
            if not 0 <= index < len(turns):
                break
            if not _is_reliably_advisor_owned(turns[index]):
                break
            selected.add(index)
    return [turns[index] for index in sorted(selected)]


def _turn_payload(turn: TranscriptTurn) -> dict:
    return {
        "turn_id": turn.id,
        "turn_index": turn.turn_index,
        "timestamp": turn.timestamp,
        "raw_speaker_label": turn.raw_speaker_label,
        "inferred_role": turn.inferred_role,
        "role_confidence": turn.role_confidence,
        "text": turn.text,
    }


def _get_or_create_score(candidate: CandidateSignal) -> SignalScore:
    if candidate.score is None:
        candidate.score = SignalScore(signal_id=candidate.id)
    return candidate.score


def _apply_deterministic_rejection(
    candidate: CandidateSignal, rejection_reason: str
) -> None:
    score = _get_or_create_score(candidate)
    score.validator_verdict = "reject"
    score.support_score = 0.0
    score.advisor_side_score = 0.0
    score.false_positive_risk = 1.0
    score.rejection_reason = rejection_reason


def _apply_deterministic_review(
    candidate: CandidateSignal, review_reason: str
) -> None:
    """Preserve grounded evidence when source selection is ambiguous, not invalid."""
    score = _get_or_create_score(candidate)
    score.validator_verdict = "needs_review"
    score.support_score = 0.5
    score.advisor_side_score = 0.8
    score.false_positive_risk = 0.5
    score.rejection_reason = review_reason


def _driver_rationale_contradicts_evidence(candidate: CandidateSignal) -> bool:
    rationale = candidate.rationale.lower()
    if candidate.item_type == "driver":
        return bool(
            re.search(
                r"\b(?:prevents?|blocks?|delays?|cannot|can't)\b.*"
                r"\b(?:proceed|proceeding|move forward|transition)\b",
                rationale,
            )
        )
    if candidate.item_type == "blocker":
        return bool(
            re.search(
                r"\b(?:increases?|strengthens?)\b.*\b(?:willingness|intent)\b.*"
                r"\b(?:proceed|move forward)\b",
                rationale,
            )
        )
    return False


def _normalized_quote(quote: str) -> str:
    words = re.findall(r"[a-z0-9']+", quote.lower())
    collapsed = [
        word
        for index, word in enumerate(words)
        if index == 0 or word != words[index - 1]
    ]
    return " ".join(collapsed)


def _is_purely_procedural_statement(quote: str) -> bool:
    normalized = _normalized_quote(quote)
    stripped = normalized
    matched = False
    for pattern in PROCEDURAL_PATTERNS:
        stripped, count = re.subn(pattern, " ", stripped)
        matched = matched or count > 0
    if not matched:
        return False
    remaining_words = set(re.findall(r"[a-z0-9']+", stripped))
    return not (remaining_words - PROCEDURAL_FILLER_WORDS)


def _decision_relevance_rejection(candidate: CandidateSignal) -> str | None:
    """Reject only deterministic, high-confidence non-signals before the LLM."""
    quote = _normalized_quote(candidate.advisor_quote)
    if quote in POLITE_ACKNOWLEDGEMENTS:
        return "polite_or_procedural"
    if candidate.item_type == "driver" and candidate.advisor_quote.rstrip().endswith("?"):
        return "insufficient_evidence"
    if _is_purely_procedural_statement(candidate.advisor_quote):
        return "polite_or_procedural"
    if (
        ("seminar" in quote or "workshop" in quote)
        and any(term in quote for term in ("i can", "we can", "i do", "we do"))
        and not any(term in quote for term in DECISION_LANGUAGE)
    ):
        return "contextual_only"
    if (
        any(re.fullmatch(pattern, quote) for pattern in BACKGROUND_ONLY_PATTERNS)
        and not any(term in quote for term in DECISION_LANGUAGE)
    ):
        return "contextual_only"
    return None


def _unsupported_escalation_rejection(candidate: CandidateSignal) -> str | None:
    quote = _normalized_quote(candidate.advisor_quote)
    rationale = _normalized_quote(candidate.rationale)
    has_dependency = bool(re.search(r"\b(?:cannot|can't|unless|must|required?|need to|depends? on)\b", quote))
    if candidate.item_type == "blocker" and candidate.advisor_quote.rstrip().endswith("?") and not has_dependency:
        return "question_to_blocker"
    if candidate.item_type == "blocker" and re.search(r"\b(?:prefer|preference)\b", quote) and re.search(r"\b(?:must|required?|cannot|unless)\b", rationale):
        return "preference_to_requirement"
    if re.search(r"\b(?:call|meet|follow up|connect)\b.*\b(?:monday|week|tomorrow)\b", quote) and "urg" in rationale:
        return "scheduling_to_urgency"
    if re.search(r"\b(?:discuss|talk)\b.*\bpartner\b", quote) and not has_dependency and re.search(r"\b(?:approv|require|cannot|unless)\b", rationale):
        return "discussion_to_approval_dependency"
    if re.search(r"\b(?:working all the time|overworked|frustrated)\b", quote) and not re.search(r"\b(?:that's why|that is why|looking for|move|change firms?)\b", quote) and re.search(r"\b(?:move|transition|motiv)\b", rationale):
        return "pain_to_move"
    if (
        re.search(
            r"\b(?:could|might|may|consider|considering|interested|if)\b"
            r"|\bbefore moving forward\b|\bclient said\b",
            quote,
        )
        and re.search(r"\b(?:commit\w*|decided|definite|will proceed|will move)\b", rationale)
    ):
        return "interest_to_commitment"
    return None


def _legacy_findings(output: EvidenceValidationOutput, candidate: CandidateSignal) -> dict:
    """Translate pre-2.0 mocked results; production prompts return every finding."""
    passed = output.verdict == "pass"
    review = output.verdict == "needs_review"
    effect = (
        "increases_move_likelihood" if candidate.item_type == "driver"
        else "decreases_move_likelihood"
    )
    direction = "supports_driver" if candidate.item_type == "driver" else "supports_blocker"
    return {
        "quote_traceability": "exact",
        "source_turn_match": "exact",
        "advisor_ownership": "reliable_advisor",
        "context_sufficiency": "incomplete" if review else "sufficient",
        "decision_relevance": "weak" if review else "material" if passed else "none",
        "supported_decision_effect": effect if passed or review else "indeterminate",
        "direction_support": direction if passed or review else "unsupported",
        "validated_evidence_strength": "weakly_implied" if review else "explicit" if passed else "unsupported",
        "rationale_grounding": "partially_grounded" if review else "fully_grounded" if passed else "unsupported",
        "unsupported_rationale_claims": [],
        "representative_dependency": "independent",
        "procedural_status": "substantive_factor",
        "unsupported_escalations": [],
        "contradiction_status": "none",
        "semantic_consistency": "consistent",
        "consistency_issues": [],
    }


def _findings(output: EvidenceValidationOutput, candidate: CandidateSignal) -> dict:
    data = output.model_dump()
    if output.quote_traceability is None:
        return _legacy_findings(output, candidate)
    return data


def _consistency_issues(findings: dict, candidate: CandidateSignal) -> list[str]:
    issues = list(findings.get("consistency_issues") or [])
    if findings.get("rationale_grounding") == "fully_grounded" and findings.get("unsupported_rationale_claims"):
        issues.append("fully_grounded_with_unsupported_claims")
    if findings.get("rationale_grounding") == "fully_grounded" and findings.get("unsupported_escalations"):
        issues.append("fully_grounded_with_unsupported_escalation")
    if findings.get("procedural_status") == "procedural_only" and findings.get("decision_relevance") == "material":
        issues.append("procedural_only_with_material_relevance")
    if findings.get("advisor_ownership") == "reliable_advisor" and findings.get("representative_dependency") == "fully_dependent":
        issues.append("advisor_owned_with_full_representative_dependency")
    if candidate.item_type == "driver" and findings.get("supported_decision_effect") in {"decreases_move_likelihood", "creates_timing_dependency"}:
        issues.append("driver_with_blocking_effect")
    if candidate.item_type == "blocker" and findings.get("supported_decision_effect") == "increases_move_likelihood":
        issues.append("blocker_with_positive_effect")
    return list(dict.fromkeys(issues))


EXPLICIT_ADVISOR_COMMITMENT_PATTERNS = (
    r"(?:i'm|we're) moving forward",
    r"let's move forward",
    r"(?:i've|we've) decided to (?:proceed|move ahead|move forward)",
    r"(?:i'm|we're) ready to proceed",
)


def is_explicit_advisor_commitment(
    candidate: CandidateSignal, findings: dict
) -> bool:
    """Recognize a supported first-person commitment, not a loose keyword hit."""
    normalized_quote = _normalized_quote(candidate.advisor_quote)
    return (
        candidate.item_type == "driver"
        and getattr(candidate, "evidence_strength", None) == "explicit"
        and any(
            re.fullmatch(pattern, normalized_quote)
            for pattern in EXPLICIT_ADVISOR_COMMITMENT_PATTERNS
        )
        and findings.get("quote_traceability") in {"exact", "normalized_exact"}
        and findings.get("source_turn_match") == "exact"
        and findings.get("advisor_ownership") == "reliable_advisor"
        and findings.get("context_sufficiency") == "sufficient"
        and findings.get("decision_relevance") == "material"
        and findings.get("direction_support") == "supports_driver"
        and findings.get("validated_evidence_strength") == "explicit"
        and findings.get("rationale_grounding") == "fully_grounded"
        and not findings.get("unsupported_rationale_claims")
        and findings.get("representative_dependency") == "independent"
        and findings.get("procedural_status") == "substantive_factor"
        and not findings.get("unsupported_escalations")
        and findings.get("contradiction_status") == "none"
        and findings.get("semantic_consistency") == "consistent"
        and not findings.get("consistency_issues")
    )


def is_reviewable_directional_factor(
    candidate: CandidateSignal, findings: dict
) -> bool:
    """Preserve supported factors whose decision impact remains uncertain."""
    expected_direction = (
        "supports_driver" if candidate.item_type == "driver" else "supports_blocker"
    )
    return (
        candidate.item_type in {"driver", "blocker"}
        and findings.get("quote_traceability") in {"exact", "normalized_exact"}
        and findings.get("source_turn_match") == "exact"
        and findings.get("advisor_ownership") == "reliable_advisor"
        and findings.get("context_sufficiency") in {"sufficient", "incomplete"}
        and findings.get("decision_relevance") in {"weak", "material"}
        and findings.get("supported_decision_effect") == "indeterminate"
        and findings.get("direction_support") == expected_direction
        and findings.get("validated_evidence_strength")
        in {"explicit", "tightly_implied", "weakly_implied"}
        and findings.get("rationale_grounding")
        in {"fully_grounded", "partially_grounded"}
        and findings.get("representative_dependency") == "independent"
        and findings.get("procedural_status") == "substantive_factor"
        and not findings.get("unsupported_escalations")
        and findings.get("contradiction_status") == "none"
        and findings.get("semantic_consistency") == "consistent"
        and not findings.get("consistency_issues")
    )


def derive_validation_decision(
    output: EvidenceValidationOutput, candidate: CandidateSignal
) -> ValidationDecision:
    """Derive reject/review/pass with hard failure taking precedence."""
    findings = _findings(output, candidate)
    issues = _consistency_issues(findings, candidate)
    explicit_commitment = is_explicit_advisor_commitment(candidate, findings)
    reviewable_factor = is_reviewable_directional_factor(candidate, findings)
    hard: list[str] = []
    checks = {
        "quote_absent": findings.get("quote_traceability") == "absent",
        "source_turn_missing": findings.get("source_turn_match") == "missing",
        "ownership_failure": findings.get("advisor_ownership") in {"mixed", "representative", "unknown", "conflicting"},
        "context_failure": findings.get("context_sufficiency") in {"contradictory", "irrelevant"},
        "decision_relevance_none": findings.get("decision_relevance") == "none",
        "neutral_or_indeterminate_effect": (
            findings.get("supported_decision_effect") in {"neutral", "indeterminate"}
            and not explicit_commitment
            and not reviewable_factor
        ),
        "direction_failure": findings.get("direction_support") in {"contradicts_candidate_type", "unsupported"},
        "unsupported_evidence": findings.get("validated_evidence_strength") == "unsupported",
        "rationale_failure": findings.get("rationale_grounding") in {"unsupported", "contradicts_evidence"},
        "representative_dependent": findings.get("representative_dependency") == "fully_dependent",
        "procedural_only": findings.get("procedural_status") == "procedural_only",
        "unsupported_escalation": bool(findings.get("unsupported_escalations")),
        "direct_contradiction": findings.get("contradiction_status") == "direct_contradiction",
        "semantic_inconsistency": findings.get("semantic_consistency") == "internally_conflicting" or bool(issues),
    }
    hard.extend(reason for reason, failed in checks.items() if failed)
    effect = (
        "increases_move_likelihood"
        if explicit_commitment
        else (
            "increases_move_likelihood"
            if reviewable_factor and candidate.item_type == "driver"
            else "decreases_move_likelihood"
            if reviewable_factor
            else findings.get("supported_decision_effect")
        )
    )
    direction = findings.get("direction_support")
    if candidate.item_type == "driver" and (effect != "increases_move_likelihood" or direction != "supports_driver"):
        hard.append("candidate_direction_mismatch")
    if candidate.item_type == "blocker" and (
        effect not in {"decreases_move_likelihood", "creates_timing_dependency"}
        or direction not in {"supports_blocker", "supports_timing_blocker"}
    ):
        hard.append("candidate_direction_mismatch")
    if hard:
        unique = tuple(dict.fromkeys(hard))
        return ValidationDecision("reject", unique[0], unique, (), tuple(issues))

    review_checks = {
        "indeterminate_effect": (
            findings.get("supported_decision_effect") == "indeterminate"
            and not explicit_commitment
        ),
        "partial_quote": findings.get("quote_traceability") == "partial",
        "ambiguous_source": findings.get("source_turn_match") == "ambiguous",
        "incomplete_context": findings.get("context_sufficiency") == "incomplete",
        "weak_relevance": findings.get("decision_relevance") == "weak",
        "weak_evidence": findings.get("validated_evidence_strength") == "weakly_implied",
        "partially_grounded_rationale": findings.get("rationale_grounding") == "partially_grounded",
        "partial_representative_dependency": findings.get("representative_dependency") == "partially_dependent",
        "mixed_procedural_content": findings.get("procedural_status") == "mixed_procedural_and_substantive",
        "unresolved_contradiction": findings.get("contradiction_status") == "unresolved",
    }
    review_reasons = tuple(reason for reason, uncertain in review_checks.items() if uncertain)
    if review_reasons:
        return ValidationDecision("needs_review", review_reasons[0], (), review_reasons, tuple(issues))
    return ValidationDecision("pass", None, (), (), tuple(issues))


def _compatible_scores(output: EvidenceValidationOutput, decision: ValidationDecision) -> tuple[float, float, float]:
    support = output.support_score
    ownership = output.advisor_side_score
    risk = output.false_positive_risk
    if decision.verdict == "pass":
        return max(support, 0.75), max(ownership, 0.8), min(risk, 0.25)
    if decision.verdict == "reject":
        return min(support, 0.49), min(ownership, 0.79), max(risk, 0.75)
    return min(max(support, 0.5), 0.74), max(ownership, 0.5), max(risk, 0.35)


def validate_evidence_for_transcript(transcript_id: int, db: Session, *, run_id: str | None = None) -> dict:
    try:
        transcript = db.get(Transcript, transcript_id)
        if transcript is None:
            logger.warning(
                "action=validate_evidence transcript_id=%s candidate_count=0 "
                "passed=0 rejected=0 needs_review=0 success=false",
                transcript_id,
            )
            raise TranscriptNotFoundError("Transcript not found")

        candidate_query = select(CandidateSignal).options(selectinload(CandidateSignal.score)).where(CandidateSignal.transcript_id == transcript_id)
        if run_id is not None:
            candidate_query = candidate_query.where(CandidateSignal.analysis_run_id == run_id)
        candidates = list(
            db.scalars(
                candidate_query.order_by(CandidateSignal.id)
            ).all()
        )
        turns = list(
            db.scalars(
                select(TranscriptTurn)
                .where(TranscriptTurn.transcript_id == transcript_id)
                .order_by(TranscriptTurn.turn_index, TranscriptTurn.id)
            ).all()
        )
    except SQLAlchemyError:
        db.rollback()
        logger.error(
            "action=validate_evidence transcript_id=%s candidate_count=0 "
            "passed=0 rejected=0 needs_review=0 success=false",
            transcript_id,
        )
        raise EvidenceValidationError(
            "Evidence validation could not be completed"
        ) from None

    if not candidates:
        logger.warning(
            "action=validate_evidence transcript_id=%s candidate_count=0 "
            "passed=0 rejected=0 needs_review=0 success=false",
            transcript_id,
        )
        raise NoCandidateSignalsError("Transcript has no candidate signals")

    diagnostics: list[dict] = []
    prompt_hash = sha256(load_prompt("03_evidence_validator.txt").encode("utf-8")).hexdigest()
    for candidate in candidates:
        started = perf_counter()
        source_ids = candidate.source_turn_ids if isinstance(candidate.source_turn_ids, list) else []
        if not candidate.rationale or not candidate.rationale.strip() or len(source_ids) != len(set(source_ids)):
            _apply_deterministic_rejection(candidate, "malformed_candidate")
            diagnostics.append({"candidate_id": candidate.id, "derived_verdict": "reject", "hard_failure_reason": "malformed_candidate", "context_turn_ids": []})
            continue
        resolved = _resolve_evidence_turn(candidate, turns)
        if resolved is None:
            quote_matches = [
                turn
                for turn in turns
                if candidate.advisor_quote and candidate.advisor_quote in turn.text
            ]
            reliable_matches = [
                turn for turn in quote_matches if _is_reliably_advisor_owned(turn)
            ]
            reason = (
                "ambiguous_evidence"
                if len(reliable_matches) > 1
                else "not_advisor_side"
                if quote_matches
                else "quote_not_found"
            )
            if reason == "ambiguous_evidence":
                _apply_deterministic_review(candidate, reason)
                diagnostics.append({"candidate_id": candidate.id, "derived_verdict": "needs_review", "needs_review_reason": reason, "context_turn_ids": []})
            else:
                _apply_deterministic_rejection(candidate, reason)
                diagnostics.append({"candidate_id": candidate.id, "derived_verdict": "reject", "hard_failure_reason": reason, "context_turn_ids": []})
            continue

        if _driver_rationale_contradicts_evidence(candidate):
            _apply_deterministic_rejection(
                candidate, "rationale_type_contradiction"
            )
            diagnostics.append({"candidate_id": candidate.id, "derived_verdict": "reject", "hard_failure_reason": "rationale_type_contradiction", "context_turn_ids": [resolved.turn.id]})
            continue

        escalation_rejection = _unsupported_escalation_rejection(candidate)
        if escalation_rejection is not None:
            _apply_deterministic_rejection(candidate, escalation_rejection)
            diagnostics.append({"candidate_id": candidate.id, "derived_verdict": "reject", "hard_failure_reason": escalation_rejection, "context_turn_ids": [resolved.turn.id]})
            continue

        relevance_rejection = _decision_relevance_rejection(candidate)
        if relevance_rejection is not None:
            _apply_deterministic_rejection(candidate, relevance_rejection)
            diagnostics.append({"candidate_id": candidate.id, "derived_verdict": "reject", "hard_failure_reason": relevance_rejection, "context_turn_ids": [resolved.turn.id]})
            continue

        supporting_turns = _supporting_turns(resolved, turns)
        candidate.timestamp = resolved.turn.timestamp
        response_metadata: dict = {}
        try:
            output = cast(
                EvidenceValidationOutput,
                call_llm_json(
                    prompt_file_name="03_evidence_validator.txt",
                    input_payload={
                        "transcript_id": transcript_id,
                        "candidate": {
                            "signal_id": candidate.id,
                            "item_type": candidate.item_type,
                            "category": candidate.category,
                            "advisor_quote": candidate.advisor_quote,
                            "timestamp": candidate.timestamp,
                            "evidence_strength": candidate.evidence_strength,
                            "rationale": candidate.rationale,
                            "source_turn_ids": candidate.source_turn_ids,
                        },
                        "supporting_turns": [
                            _turn_payload(turn) for turn in supporting_turns
                        ],
                        "deterministic_facts": {
                            "quote_traceability": "exact",
                            "source_turn_match": "exact",
                            "advisor_ownership": "reliable_advisor",
                            "ownership_turn_id": resolved.turn.id,
                            "resolution_method": resolved.resolution_method,
                            "context_turn_ids": [turn.id for turn in supporting_turns],
                            "standalone_question": candidate.advisor_quote.rstrip().endswith("?"),
                            "procedural_only": _is_purely_procedural_statement(candidate.advisor_quote),
                        },
                    },
                    response_model=EvidenceValidationOutput,
                    model=settings.evidence_validator_model,
                    temperature=0,
                    transcript_id=transcript_id,
                    response_observer=response_metadata.update,
                ),
            )
        except LLMClientError:
            db.rollback()
            diagnostics.append({
                "candidate_id": candidate.id, "derived_verdict": "reject",
                "hard_failure_reason": "malformed_or_unavailable_model_output",
                "context_turn_ids": [turn.id for turn in supporting_turns],
                "model": settings.evidence_validator_model,
                "prompt_version": "validation_2.0", "prompt_hash": prompt_hash,
                "fallback_usage": True, "malformed_model_output": True,
                "retry_count": None,
            })
            VALIDATION_DIAGNOSTICS[transcript_id] = diagnostics
            logger.error(
                "action=validate_evidence transcript_id=%s candidate_count=%s "
                "passed=0 rejected=0 needs_review=0 success=false",
                transcript_id,
                len(candidates),
            )
            raise EvidenceValidationError(
                "Evidence validation service is unavailable"
            ) from None

        decision = derive_validation_decision(output, candidate)
        support, ownership, risk = _compatible_scores(output, decision)
        score = _get_or_create_score(candidate)
        score.validator_verdict = decision.verdict
        score.support_score = support
        score.advisor_side_score = ownership
        score.false_positive_risk = risk
        score.rejection_reason = decision.reason
        findings = _findings(output, candidate)
        diagnostics.append({
            "candidate_id": candidate.id,
            "deterministic_prechecks": {
                "quote_traceability": "exact", "source_turn_match": "exact",
                "advisor_ownership": "reliable_advisor",
            },
            "structured_findings": findings,
            "derived_verdict": decision.verdict,
            "hard_failure_reason": decision.hard_failures[0] if decision.hard_failures else None,
            "needs_review_reason": decision.review_reasons[0] if decision.review_reasons else None,
            "unsupported_rationale_claims": findings.get("unsupported_rationale_claims", []),
            "unsupported_escalations": findings.get("unsupported_escalations", []),
            "contradiction_status": findings.get("contradiction_status"),
            "consistency_issues": list(decision.consistency_issues),
            "context_turn_ids": [turn.id for turn in supporting_turns],
            "model": settings.evidence_validator_model,
            "prompt_version": "validation_2.0",
            "prompt_hash": prompt_hash,
            "validation_duration_ms": round((perf_counter() - started) * 1000, 2),
            "fallback_usage": output.quote_traceability is None,
            "malformed_model_output": False,
            "retry_count": 0,
            "raw_response": response_metadata.get("raw_response"),
            "token_usage": response_metadata.get("token_usage"),
            "response_retry_count": response_metadata.get("retry_count", 0),
        })

    transcript.status = "evidence_validated"
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.error(
            "action=validate_evidence transcript_id=%s candidate_count=%s "
            "passed=0 rejected=0 needs_review=0 success=false",
            transcript_id,
            len(candidates),
        )
        raise EvidenceValidationError("Evidence validation could not be saved") from None

    counts = {"pass": 0, "reject": 0, "needs_review": 0}
    for candidate in candidates:
        verdict = candidate.score.validator_verdict
        if verdict in counts:
            counts[verdict] += 1
    VALIDATION_DIAGNOSTICS[transcript_id] = diagnostics
    if run_id is not None:
        persist_validation_diagnostics(db, run_id, diagnostics)

    logger.info(
        "action=validate_evidence transcript_id=%s candidate_count=%s "
        "passed=%s rejected=%s needs_review=%s success=true",
        transcript_id,
        len(candidates),
        counts["pass"],
        counts["reject"],
        counts["needs_review"],
    )
    return {
        "transcript_id": transcript_id,
        "status": "evidence_validated",
        "candidate_count": len(candidates),
        "passed": counts["pass"],
        "rejected": counts["reject"],
        "needs_review": counts["needs_review"],
    }
