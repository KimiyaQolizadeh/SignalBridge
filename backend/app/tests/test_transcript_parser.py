from backend.app.services.transcript_parser import parse_transcript_text


def test_timestamp_and_speaker_on_same_line() -> None:
    turns = parse_transcript_text("00:01:23 Advisor Name: text here")

    assert turns == [
        {
            "turn_index": 0,
            "timestamp": "00:01:23",
            "raw_speaker_label": "Advisor Name",
            "text": "text here",
        }
    ]


def test_speaker_then_timestamp() -> None:
    turns = parse_transcript_text(
        "Speaker 1 00:01:23\nHello, thanks for joining."
    )

    assert turns[0]["timestamp"] == "00:01:23"
    assert turns[0]["raw_speaker_label"] == "Speaker 1"
    assert turns[0]["text"] == "Hello, thanks for joining."


def test_continuation_lines_are_combined_without_rewriting() -> None:
    turns = parse_transcript_text(
        "Speaker 1:\nFirst spoken line.\nSecond spoken line.\nSpeaker 2:\nReply."
    )

    assert turns[0]["text"] == "First spoken line.\nSecond spoken line."
    assert turns[1]["text"] == "Reply."


def test_missing_timestamp_is_none() -> None:
    turns = parse_transcript_text("Advisor Name:\nI am interested.")

    assert turns[0]["timestamp"] is None
    assert turns[0]["raw_speaker_label"] == "Advisor Name"


def test_empty_lines_are_ignored() -> None:
    turns = parse_transcript_text(
        "\n\n00:01:23\n\nSpeaker 1: Hello.\n\n\nSpeaker 2: Goodbye.\n"
    )

    assert len(turns) == 2
    assert turns[0]["timestamp"] == "00:01:23"
    assert turns[0]["text"] == "Hello."
    assert turns[1]["turn_index"] == 1


def test_flattened_transcript_splits_multiple_bracketed_turns() -> None:
    turns = parse_transcript_text(
        "00:02:59.290 --> 00:03:00.510 [OPTIMIZE_REP]: First exact text. "
        "00:03:08.930 --> 00:03:10.289 [ADVISOR]: Second exact text."
    )

    assert turns == [
        {
            "turn_index": 0,
            "timestamp": "00:02:59.290",
            "raw_speaker_label": "OPTIMIZE_REP",
            "inferred_role": "optimize_rep",
            "role_confidence": 1.0,
            "text": "First exact text.",
        },
        {
            "turn_index": 1,
            "timestamp": "00:03:08.930",
            "raw_speaker_label": "ADVISOR",
            "inferred_role": "advisor",
            "role_confidence": 1.0,
            "text": "Second exact text.",
        },
    ]


def test_flattened_numbered_advisors_are_preserved_as_advisor_turns() -> None:
    turns = parse_transcript_text(
        "00:00:01.000 --> 00:00:02.000 [ADVISOR_1]: First advisor. "
        "00:00:03.000 --> 00:00:04.000 [OPTIMIZE_REP]: Representative. "
        "00:00:05.000 --> 00:00:06.000 [ADVISOR_2]: Second advisor."
    )

    assert [turn["raw_speaker_label"] for turn in turns] == [
        "ADVISOR_1",
        "OPTIMIZE_REP",
        "ADVISOR_2",
    ]
    assert [turn["inferred_role"] for turn in turns] == [
        "advisor",
        "optimize_rep",
        "advisor",
    ]


def test_normal_line_oriented_transcript_still_parses() -> None:
    turns = parse_transcript_text(
        "00:01:23 Advisor Name: Line-oriented text.\n"
        "00:01:24 Optimize Name: Reply."
    )

    assert [turn["raw_speaker_label"] for turn in turns] == [
        "Advisor Name",
        "Optimize Name",
    ]
    assert [turn["text"] for turn in turns] == [
        "Line-oriented text.",
        "Reply.",
    ]


def test_metadata_is_ignored_before_flattened_dialogue() -> None:
    turns = parse_transcript_text(
        "Meeting ID: 123\n"
        "Meeting Topic: Synthetic topic\n"
        "Host Email: synthetic@example.test\n"
        "Start Time (Eastern): 2026-01-01\n"
        "00:00:01.000 --> 00:00:02.000 [ADVISOR]: Advisor text. "
        "00:00:03.000 --> 00:00:04.000 [OPTIMIZE_REP]: Rep text."
    )

    assert len(turns) == 2
    assert all(
        turn["raw_speaker_label"] in {"ADVISOR", "OPTIMIZE_REP"}
        for turn in turns
    )


def test_malformed_flattened_segment_is_skipped_without_merging_text() -> None:
    turns = parse_transcript_text(
        "00:00:01.000 --> 00:00:02.000 [ADVISOR]: Valid advisor text. "
        "00:00:03.000 --> 00:00:04.000 [ADVISOR] malformed text. "
        "00:00:05.000 --> 00:00:06.000 [OPTIMIZE_REP]: Valid rep text."
    )

    assert len(turns) == 2
    assert turns[0]["text"] == "Valid advisor text."
    assert turns[1]["text"] == "Valid rep text."
