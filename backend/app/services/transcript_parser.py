import re
from typing import Any


TIMESTAMP_PATTERN = r"\d{1,2}:\d{2}:\d{2}"
PRECISE_TIMESTAMP_PATTERN = rf"{TIMESTAMP_PATTERN}(?:\.\d+)?"
TIMESTAMP_ONLY_RE = re.compile(rf"^(?P<timestamp>{TIMESTAMP_PATTERN})$")
TIMESTAMP_SPEAKER_RE = re.compile(
    rf"^(?P<timestamp>{TIMESTAMP_PATTERN})\s+"
    r"(?P<speaker>[^:]{1,100}):\s*(?P<text>.*)$"
)
SPEAKER_TIMESTAMP_RE = re.compile(
    rf"^(?P<speaker>.+?)\s+(?P<timestamp>{TIMESTAMP_PATTERN})\s*:?[ \t]*$"
)
SPEAKER_RE = re.compile(r"^(?P<speaker>[^:]{1,100}):\s*(?P<text>.*)$")
FLATTENED_BOUNDARY_RE = re.compile(
    rf"(?P<start>{PRECISE_TIMESTAMP_PATTERN})\s*-->\s*"
    rf"(?P<end>{PRECISE_TIMESTAMP_PATTERN})\s*"
    r"\[(?P<speaker>[^\]]+)\](?P<colon>\s*:)?[ \t]*",
    re.IGNORECASE,
)
METADATA_LABELS = {
    "meeting id",
    "meeting topic",
    "host email",
    "start time",
    "start time (eastern)",
}


def _looks_like_speaker(label: str) -> bool:
    """Avoid treating ordinary prose containing a colon as a speaker header."""

    stripped = label.strip()
    return (
        bool(stripped)
        and len(stripped.split()) <= 8
        and not any(character in stripped for character in ".!?")
    )


def _is_metadata_label(label: str) -> bool:
    return label.strip().lower() in METADATA_LABELS


def _authoritative_role(speaker: str) -> str | None:
    if speaker == "OPTIMIZE_REP":
        return "optimize_rep"
    if re.fullmatch(r"ADVISOR(?:_\d+)?", speaker):
        return "advisor"
    return None


def _parse_flattened_dialogue(raw_text: str) -> list[dict] | None:
    """Parse timestamp-range/bracket-label segments embedded on one line.

    A tolerant boundary expression also detects malformed bracketed headers. A
    malformed segment is skipped rather than merged into the preceding valid
    speaker's verbatim text.
    """
    boundaries = list(FLATTENED_BOUNDARY_RE.finditer(raw_text))
    if not boundaries:
        return None

    turns: list[dict[str, Any]] = []
    for index, boundary in enumerate(boundaries):
        next_start = (
            boundaries[index + 1].start()
            if index + 1 < len(boundaries)
            else len(raw_text)
        )
        speaker = boundary.group("speaker").strip().upper()
        role = _authoritative_role(speaker)

        # Missing colons and unrecognized bracket labels are malformed segment
        # headers. Their text must not be attributed to the previous speaker.
        if boundary.group("colon") is None or role is None:
            continue

        text = raw_text[boundary.end() : next_start].strip()
        if not text:
            continue

        turns.append(
            {
                "turn_index": len(turns),
                "timestamp": boundary.group("start"),
                "raw_speaker_label": speaker,
                "inferred_role": role,
                "role_confidence": 1.0,
                "text": text,
            }
        )

    return turns


def parse_transcript_text(raw_text: str) -> list[dict]:
    """Split common Zoom transcript layouts without rewriting spoken text."""

    flattened_turns = _parse_flattened_dialogue(raw_text)
    if flattened_turns is not None:
        return flattened_turns

    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_timestamp: str | None = None

    def finish_current() -> None:
        nonlocal current
        if current is None or not current["text_parts"]:
            current = None
            return

        turns.append(
            {
                "turn_index": len(turns),
                "timestamp": current["timestamp"],
                "raw_speaker_label": current["raw_speaker_label"] or "unknown",
                # Newlines preserve the original boundary between spoken lines.
                "text": "\n".join(current["text_parts"]),
            }
        )
        current = None

    def start_turn(
        speaker: str | None, timestamp: str | None, first_text: str = ""
    ) -> None:
        nonlocal current, pending_timestamp
        finish_current()
        current = {
            "timestamp": timestamp,
            "raw_speaker_label": speaker.strip() if speaker else "unknown",
            "text_parts": [first_text] if first_text else [],
        }
        pending_timestamp = None

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = TIMESTAMP_SPEAKER_RE.fullmatch(line)
        if match and _looks_like_speaker(match.group("speaker")):
            if _is_metadata_label(match.group("speaker")):
                finish_current()
                pending_timestamp = None
                continue
            start_turn(
                match.group("speaker"), match.group("timestamp"), match.group("text")
            )
            continue

        match = SPEAKER_TIMESTAMP_RE.fullmatch(line)
        if match and _looks_like_speaker(match.group("speaker")):
            start_turn(match.group("speaker"), match.group("timestamp"))
            continue

        match = TIMESTAMP_ONLY_RE.fullmatch(line)
        if match:
            finish_current()
            pending_timestamp = match.group("timestamp")
            continue

        match = SPEAKER_RE.fullmatch(line)
        if match and _looks_like_speaker(match.group("speaker")):
            if _is_metadata_label(match.group("speaker")):
                finish_current()
                pending_timestamp = None
                continue
            start_turn(
                match.group("speaker"), pending_timestamp, match.group("text")
            )
            continue

        if current is None:
            start_turn("unknown", pending_timestamp)
        current["text_parts"].append(line)

    finish_current()
    return turns
