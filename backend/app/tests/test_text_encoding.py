import pytest

from backend.app.services.text_encoding import repair_common_utf8_mojibake


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Already correct … ellipsis.", "Already correct … ellipsis."),
        ("Latin-1 mojibake â\x80¦ ellipsis.", "Latin-1 mojibake … ellipsis."),
        ("Windows-1252 mojibake â€¦ ellipsis.", "Windows-1252 mojibake … ellipsis."),
    ],
)
def test_valid_and_mojibake_ellipsis(source: str, expected: str) -> None:
    assert repair_common_utf8_mojibake(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Itâ€™s the advisorâ€™s choice.", "It’s the advisor’s choice."),
        ("â€œQuoted textâ\x80\x9d", "“Quoted text”"),
        ("‘Already correct’ and “valid quotes”", "‘Already correct’ and “valid quotes”"),
    ],
)
def test_curly_quotes_and_apostrophes(source: str, expected: str) -> None:
    assert repair_common_utf8_mojibake(source) == expected


def test_ordinary_ascii_is_unchanged() -> None:
    text = "Ordinary ASCII transcript text 123."

    assert repair_common_utf8_mojibake(text) == text


def test_already_correct_unicode_is_unchanged() -> None:
    text = "Correct Unicode: café, résumé, …, “quoted”, and advisor’s."

    assert repair_common_utf8_mojibake(text) == text
