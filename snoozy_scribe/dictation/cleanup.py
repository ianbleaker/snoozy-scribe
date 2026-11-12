"""LLM cleanup of dictated text, with a guard against the model answering it.

The system prompt is the user's settings/prompts/dictation-cleanup.md override
if there is one, else the built-in default (see local_llm.load_prompt). It's
read on every call, so edits apply without restarting the app.

A small instruct model sometimes *responds* to a dictated question instead of
cleaning it up. The guard falls back to the regex-cleaned raw text whenever
the output looks like it isn't a light edit of the input.
"""

from __future__ import annotations

# DEFAULT_CLEANUP_MODEL is re-exported for dictate.py's load_cleaner().
from snoozy_scribe.local_llm import DEFAULT_CLEANUP_MODEL, LocalLLM, _words, load_prompt, word_similarity


def guard_fails(raw: str, cleaned: str) -> bool:
    """True if `cleaned` doesn't look like a cleanup of `raw`."""
    if cleaned == "EMPTY":
        return len(_words(raw)) >= 3
    if len(cleaned) > 1.5 * len(raw) + 40:
        return True
    return word_similarity(raw, cleaned) < 0.5


def build_user_message(raw: str, terms: list[str], app_context: str | None) -> str:
    lines = []
    if terms:
        lines.append(f"Preferred spellings for names and terms: {', '.join(terms)}.")
    if app_context:
        lines.append(app_context)
    lines += ["<<<TRANSCRIPT", raw, "TRANSCRIPT"]
    return "\n".join(lines)


class Cleaner(LocalLLM):
    """The dictation cleanup model. Not thread-safe: run all calls (like all
    MLX work) on one thread."""

    def clean(self, raw: str, terms: list[str], app_context: str | None = None) -> tuple[str, bool]:
        """Return (text, used_fallback). `raw` is the regex-cleaned transcript."""
        if not raw.strip():
            return "", False
        messages = [
            {"role": "system", "content": load_prompt("dictation-cleanup")},
            {"role": "user", "content": build_user_message(raw, terms, app_context)},
        ]
        max_tokens = 2 * len(self.tokenizer.encode(raw)) + 32
        cleaned = self.generate(messages, max_tokens)
        if guard_fails(raw, cleaned):
            return raw, True
        if cleaned == "EMPTY":
            return "", False
        return cleaned, False
