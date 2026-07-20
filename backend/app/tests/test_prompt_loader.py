import pytest

from backend.app.services.prompt_loader import load_prompt, prompt_sha256


def test_load_existing_prompt() -> None:
    prompt = load_prompt("02_candidate_signal_extractor.txt")

    assert "Could this statement help explain a business decision?" in prompt
    assert "CandidateSignalBatchOutput" in prompt


def test_missing_prompt_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist.txt")


def test_prompt_sha256_is_stable_and_content_based() -> None:
    digest = prompt_sha256("02_candidate_signal_extractor.txt")

    assert digest == prompt_sha256("02_candidate_signal_extractor.txt")
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_signal_prompts_define_question_condition_and_rationale_rules() -> None:
    extractor = load_prompt("02_candidate_signal_extractor.txt")
    validator = load_prompt("03_evidence_validator.txt")

    assert "Extract a question when its topic reveals a plausible business need" in extractor
    assert "Use no_signal only for greetings" in extractor
    assert "emit separate driver and blocker candidates" in extractor
    assert '"We\'re moving forward" is an explicit driver' in extractor
    assert "rationale and item_type must semantically agree" in validator
    assert "very short or ambiguous quotes" in validator
    assert "exact verbatim excerpt traceable" in validator
