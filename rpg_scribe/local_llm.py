"""The local mlx-lm instruct model shared by dictation and transcript cleanup.

Holds only the model-agnostic parts: loading, greedy generation, the word
similarity both guards use, and reading prompts from settings/prompts/.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from rpg_scribe import paths

DEFAULT_CLEANUP_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_WORD = re.compile(r"[\w']+")


def _words(text: str) -> list[str]:
    # Punctuation is ignored: adding it is the model's job, not a sign of drift.
    return _WORD.findall(text.lower())


def word_similarity(a: str, b: str) -> float:
    """Similarity (0-1) of the two texts' word sequences, ignoring case and punctuation."""
    return difflib.SequenceMatcher(None, _words(a), _words(b), autojunk=False).ratio()


def load_prompt(name: str) -> str:
    """Read settings/prompts/<name>.md. Read on every call, so edits apply without a restart."""
    path = paths.PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text().strip()


class LocalLLM:
    """Holds an mlx-lm model in memory. Not thread-safe: run all calls
    (like all MLX work) on one thread."""

    def __init__(self, model_dir: Path) -> None:
        import mlx_lm

        self._mlx_lm = mlx_lm
        self.model, self.tokenizer = mlx_lm.load(str(model_dir))

    def generate(self, messages: list[dict], max_tokens: int) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            # Qwen3-style templates think by default; others ignore this.
            enable_thinking=False,
        )
        # No sampler means greedy decoding (temperature 0).
        text = self._mlx_lm.generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens)
        return _THINK_BLOCK.sub("", text).strip()

    def warm_up(self) -> None:
        self.generate([{"role": "user", "content": "hi"}], max_tokens=1)
