import re

from ..models import CandidateSignal, TranscriptTurn


MAX_CONTEXT_TURNS_PER_SIDE = 2
MAX_CONTEXT_ITEMS = 5
_MID_THOUGHT_END = re.compile(r"(?:…|\.\.\.|[,;:]|\b(?:and|but|because|so|or))\s*$", re.IGNORECASE)
_REFERENTIAL_START = re.compile(r"^(?:that|this|it|he|she|they|those|these)\b", re.IGNORECASE)
_MID_THOUGHT_START = re.compile(r"^[a-z]")
_RELATIONSHIP_RULES = (
    (re.compile(r"\b(?:external(?:ly)?|third[- ]party|provider|vendor|outsourc\w*|delegat\w*)\b", re.I),
     re.compile(r"\b(?:external(?:ly)?|third[- ]party|provider|vendor|outsourc\w*|delegat\w*|you guys (?:do|handle|provide)|(?:do|handle) .+ for us)\b", re.I)),
    (re.compile(r"\b(?:adopt\w*|switch\w*|committ?\w*|intend\w* to (?:move|proceed)|will (?:move|proceed))\b", re.I),
     re.compile(r"\b(?:adopt\w*|switch\w*|committ?\w*|intend\w* to (?:move|proceed)|will (?:move|proceed))\b", re.I)),
    (re.compile(r"\b(?:retain\w*|retention|stay(?:ing)?|remain(?:ing)?)\b", re.I),
     re.compile(r"\b(?:retain\w*|retention|stay(?:ing)?|remain(?:ing)?)\b", re.I)),
    (re.compile(r"\b(?:urgent|urgency|immediate(?:ly)?|asap|deadline|time[- ]sensitive)\b", re.I),
     re.compile(r"\b(?:urgent|urgency|immediate(?:ly)?|asap|deadline|time[- ]sensitive)\b", re.I)),
    (re.compile(r"\b(?:decision[- ]maker|decision authority|has authority|approval (?:is )?required|determines? (?:the|those) decisions?)\b", re.I),
     re.compile(r"\b(?:decision[- ]maker|authority|approv\w*|decid\w*|determines? (?:the|those) decisions?)\b", re.I)),
    (re.compile(r"\b(?:because|due to|causes?|results? in|leads? to|enables?|prevents?|from having)\b", re.I),
     re.compile(r"\b(?:because|due to|causes?|results? in|leads? to|enables?|prevents?|determines? (?:the|those) decisions?|you guys .+ we (?:do not|don't|need not|needn't))\b", re.I)),
)


def _source_positions(candidate: CandidateSignal, turns: list[TranscriptTurn]) -> list[int]:
    source_ids = (
        {item for item in candidate.source_turn_ids if isinstance(item, int)}
        if isinstance(candidate.source_turn_ids, list)
        else set()
    )
    return [
        index
        for index, turn in enumerate(turns)
        if turn.id in source_ids and candidate.advisor_quote in turn.text
    ]


def context_turns(
    candidate: CandidateSignal,
    turns: list[TranscriptTurn],
    *,
    include_source: bool,
) -> list[TranscriptTurn]:
    """Return a small, transcript-ordered context window around exact evidence."""
    positions = _source_positions(candidate, turns)
    if not positions:
        return []

    selected = set(positions if include_source else [])
    first, last = min(positions), max(positions)
    for position in (first - 1, last + 1):
        if 0 <= position < len(turns) and position not in positions:
            selected.add(position)

    # A quote may be an exact excerpt rather than the complete source turn. Keep
    # that source text visible as context without changing the verbatim quote.
    for position in positions:
        if candidate.advisor_quote.strip() != turns[position].text.strip():
            selected.add(position)

    # Expand by at most one extra turn per side only for an uninterrupted,
    # visibly incomplete statement. Immediate turns are never skipped by role.
    if first > 1:
        previous = turns[first - 1]
        extra = turns[first - 2]
        if (
            previous.raw_speaker_label == turns[first].raw_speaker_label
            and extra.raw_speaker_label == previous.raw_speaker_label
            and _MID_THOUGHT_START.search(previous.text.strip())
        ):
            selected.add(first - 2)
    if last + 2 < len(turns):
        following = turns[last + 1]
        extra = turns[last + 2]
        if (
            following.raw_speaker_label == turns[last].raw_speaker_label
            and extra.raw_speaker_label == following.raw_speaker_label
            and _MID_THOUGHT_END.search(following.text.strip())
        ):
            selected.add(last + 2)

    bounded = sorted(selected)[:MAX_CONTEXT_ITEMS]
    return [turns[position] for position in bounded]


def adjacent_advisor_completion_turns(
    candidate: CandidateSignal, turns: list[TranscriptTurn]
) -> list[TranscriptTurn]:
    """Return only adjacent same-advisor turns needed to complete an excerpt."""
    positions = _source_positions(candidate, turns)
    if len(positions) != 1:
        return []
    position = positions[0]
    source = turns[position]
    selected: list[TranscriptTurn] = []
    source_starts_mid_thought = bool(
        _MID_THOUGHT_START.search(source.text.strip())
        or _REFERENTIAL_START.search(source.text.strip())
    )
    source_ends_mid_thought = bool(_MID_THOUGHT_END.search(source.text.strip()))

    if position > 0:
        previous = turns[position - 1]
        if (
            source_starts_mid_thought
            and previous.inferred_role == "advisor"
            and source.inferred_role == "advisor"
            and previous.raw_speaker_label == source.raw_speaker_label
        ):
            selected.append(previous)
    if position + 1 < len(turns):
        following = turns[position + 1]
        if (
            source_ends_mid_thought
            and following.inferred_role == "advisor"
            and source.inferred_role == "advisor"
            and following.raw_speaker_label == source.raw_speaker_label
        ):
            selected.append(following)
    return selected


def context_payload(turn: TranscriptTurn) -> dict:
    return {
        "turn_id": turn.id,
        "speaker": turn.raw_speaker_label,
        "text": turn.text,
        "timestamp": turn.timestamp,
    }

def ground_rationale(
    candidate: CandidateSignal, supporting_turns: list[TranscriptTurn]
) -> bool:
    """Replace a rationale only when it adds a protected unsupported relation."""
    source_text = " ".join(turn.text for turn in supporting_turns)
    for rationale_pattern, support_pattern in _RELATIONSHIP_RULES:
        if rationale_pattern.search(candidate.rationale) and not support_pattern.search(source_text):
            candidate.rationale = f'The advisor states, "{candidate.advisor_quote}"'
            return False
    return True