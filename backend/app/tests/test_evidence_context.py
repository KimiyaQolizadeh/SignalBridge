from backend.app.models import CandidateSignal, TranscriptTurn
from backend.app.services.evidence_context import (
    adjacent_advisor_completion_turns,
    context_turns,
)


def turn(turn_id: int, index: int, speaker: str, text: str) -> TranscriptTurn:
    return TranscriptTurn(
        id=turn_id,
        transcript_id=1,
        turn_index=index,
        raw_speaker_label=speaker,
        inferred_role="advisor" if speaker == "Advisor" else "representative",
        text=text,
    )


def candidate(source: TranscriptTurn, quote: str | None = None) -> CandidateSignal:
    return CandidateSignal(
        transcript_id=1,
        item_type="driver",
        category="Time Savings",
        advisor_quote=quote or source.text,
        rationale="Grounded rationale.",
        source_turn_ids=[source.id],
    )


def test_immediate_preceding_and_following_turns_are_selected() -> None:
    turns = [turn(1, 0, "Advisor", "Before."), turn(2, 1, "Advisor", "Evidence."), turn(3, 2, "Advisor", "After.")]

    result = context_turns(candidate(turns[1]), turns, include_source=False)

    assert [item.id for item in result] == [1, 3]


def test_immediate_representative_turn_is_not_skipped() -> None:
    turns = [
        turn(1, 0, "Advisor", "Earlier advisor context."),
        turn(2, 1, "Representative", "Immediate representative question."),
        turn(3, 2, "Advisor", "Evidence."),
        turn(4, 3, "Representative", "Immediate representative response."),
        turn(5, 4, "Advisor", "Later advisor context."),
    ]

    result = context_turns(candidate(turns[2]), turns, include_source=False)

    assert [item.id for item in result] == [2, 4]


def test_contiguous_mid_thought_may_expand_with_a_hard_bound() -> None:
    turns = [
        turn(1, 0, "Advisor", "Unrelated distant turn."),
        turn(2, 1, "Advisor", "setup for the thought"),
        turn(3, 2, "Advisor", "Evidence."),
        turn(4, 3, "Advisor", "The thought continues, and"),
        turn(5, 4, "Advisor", "completes here."),
        turn(6, 5, "Advisor", "Unrelated distant ending."),
    ]

    result = context_turns(candidate(turns[2]), turns, include_source=False)

    assert [item.id for item in result] == [1, 2, 4, 5]
    assert 6 not in [item.id for item in result]


def test_source_turn_excerpt_is_visible_without_changing_quote() -> None:
    source = turn(2, 1, "Advisor", "You handle CE credits for us. We save time.")
    item = candidate(source, "We save time.")

    result = context_turns(item, [turn(1, 0, "Representative", "Question."), source], include_source=False)

    assert [context.id for context in result] == [1, 2]
    assert item.advisor_quote == "We save time."


def test_previous_same_advisor_completes_mid_sentence_driver() -> None:
    previous = turn(1, 0, "Advisor", "I joined this business to be with clients, and my time is")
    source = turn(2, 1, "Advisor", "not with clients, so I want that time back.")
    result = adjacent_advisor_completion_turns(candidate(source), [previous, source])
    assert [item.id for item in result] == [1]


def test_next_same_advisor_completes_mid_sentence_driver() -> None:
    source = turn(1, 0, "Advisor", "I want to spend more time with clients, and...")
    following = turn(2, 1, "Advisor", "that would give me my energy back.")
    result = adjacent_advisor_completion_turns(candidate(source), [source, following])
    assert [item.id for item in result] == [2]


def test_representative_turn_cannot_complete_advisor_evidence() -> None:
    representative = turn(1, 0, "Representative", "This would give you time back, and")
    source = turn(2, 1, "Advisor", "help clients more.")
    assert adjacent_advisor_completion_turns(
        candidate(source), [representative, source]
    ) == []


def test_completion_is_bounded_to_immediate_turn() -> None:
    distant = turn(1, 0, "Advisor", "Distant context, and")
    previous = turn(2, 1, "Advisor", "I joined to work with clients, and")
    source = turn(3, 2, "Advisor", "not spend my time elsewhere.")
    result = adjacent_advisor_completion_turns(
        candidate(source), [distant, previous, source]
    )
    assert [item.id for item in result] == [2]


def test_unresolved_pronoun_uses_immediate_same_advisor_antecedent() -> None:
    previous = turn(
        1, 0, "Advisor", "The ability to reduce manual work would help my team."
    )
    source = turn(2, 1, "Advisor", "That would make a move more attractive.")

    result = adjacent_advisor_completion_turns(candidate(source), [previous, source])

    assert [item.id for item in result] == [1]


def test_referential_context_does_not_expand_more_than_one_turn() -> None:
    turns = [turn(1, 0, "Advisor", "Distant."), turn(2, 1, "Advisor", "The platform saves time."), turn(3, 2, "Advisor", "That would help my decision.")]
    result = adjacent_advisor_completion_turns(candidate(turns[2]), turns)

    assert [item.id for item in result] == [2]
