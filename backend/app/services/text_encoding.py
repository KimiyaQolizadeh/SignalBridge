"""Conservative text normalization for common UTF-8 mojibake."""


COMMON_UNICODE_PUNCTUATION = "…‘’“”–—"


def _build_mojibake_replacements() -> dict[str, str]:
    replacements: dict[str, str] = {}
    for character in COMMON_UNICODE_PUNCTUATION:
        encoded = character.encode("utf-8")
        replacements[encoded.decode("latin-1")] = character
        try:
            replacements[encoded.decode("cp1252")] = character
        except UnicodeDecodeError:
            # Some UTF-8 continuation bytes are undefined in Windows-1252.
            pass
    return replacements


MOJIBAKE_REPLACEMENTS = _build_mojibake_replacements()


def repair_common_utf8_mojibake(text: str) -> str:
    """Repair known double-decoded punctuation while preserving valid Unicode."""
    repaired = text
    for corrupted, correct in MOJIBAKE_REPLACEMENTS.items():
        repaired = repaired.replace(corrupted, correct)
    return repaired
