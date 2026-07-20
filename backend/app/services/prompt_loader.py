import hashlib
from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(prompt_file_name: str) -> str:
    """Load a versioned prompt without logging its contents."""

    prompt_path = (PROMPTS_DIR / prompt_file_name).resolve()
    if prompt_path.parent != PROMPTS_DIR.resolve() or not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file_name}")

    return prompt_path.read_text(encoding="utf-8")


def prompt_sha256(prompt_file_name: str) -> str:
    return hashlib.sha256(load_prompt(prompt_file_name).encode("utf-8")).hexdigest()
