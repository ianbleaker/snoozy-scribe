"""Local LLM cleanup pass over a finished session transcript.

Each line is either dropped or lightly edited (punctuation, fillers,
stutters) by the shared local model, guarded per line so a substantive line
can't vanish or drift. Writes `<stem>-cleaned.md` plus `<stem>-cleaned.changes.md`
listing every drop and rewrite. See docs/design/transcription-pipeline.md.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import tqdm
import yaml

from rpg_scribe.local_llm import LocalLLM, _words, load_prompt, word_similarity

DEFAULT_CHUNK_LINES = 30
DEFAULT_MIN_SIMILARITY = 0.8
DEFAULT_MAX_DROP_WORDS = 6

# Same literal as transcribe.py's LAUGHTER_MARKER; not imported, since that
# module pulls in Whisper.
LAUGHTER_MARKER = "[laughter]"
INAUDIBLE_PLACEHOLDER = "!! [inaudible]"

_LINE = re.compile(r"^\[(?P<ts>[^\]]+)\] (?P<speaker>[^:]+): (?P<text>.*)$")
_BLEED_PREFIX = re.compile(r"^\[possible bleed: [^\]]*\] ")
_REPLY_LINE = re.compile(r"^(\d+):\s?(.*)$")

STAT_KEYS = ("lines_sent", "dropped", "rewritten", "guard_fallbacks", "drop_refused")


def parse_transcript(text: str) -> tuple[dict, list[str]]:
    """Split a transcript into its YAML frontmatter and body lines."""
    if not text.startswith("---\n"):
        raise ValueError("transcript has no YAML frontmatter")
    end = text.find("\n---\n", 3)
    if end == -1:
        raise ValueError("transcript frontmatter is not closed")
    frontmatter = yaml.safe_load(text[4:end]) or {}
    if not isinstance(frontmatter, dict):
        raise ValueError("transcript frontmatter is not a mapping")
    return frontmatter, text[end + len("\n---\n") :].splitlines()


def build_chunk_message(items: list[tuple[int, str, str]]) -> str:
    lines = ["<<<TRANSCRIPT"]
    lines += [f"{local_id} | {speaker}: {text}" for local_id, speaker, text in items]
    lines.append("TRANSCRIPT")
    return "\n".join(lines)


def parse_reply(reply: str) -> dict[int, str | None]:
    """Map local IDs to rewritten text, or None for DROP. First occurrence wins."""
    result: dict[int, str | None] = {}
    for line in reply.splitlines():
        match = _REPLY_LINE.match(line.strip())
        if not match:
            continue
        local_id = int(match.group(1))
        if local_id in result:
            continue
        text = match.group(2).strip()
        result[local_id] = None if text == "DROP" else text
    return result


def only_deletes_words(original: str, new: str) -> bool:
    """True if `new`'s words are `original`'s with some removed, in order.

    Case and punctuation are ignored (as in word_similarity), so fixing them
    passes; changing, adding or reordering a word fails.
    """
    remaining = iter(_words(original))
    return all(word in remaining for word in _words(new))


def apply_reply(
    items: list[tuple[int, str, str]],
    reply: dict[int, str | None],
    min_similarity: float,
    max_drop_words: int,
    stats: Counter,
) -> list[str | None]:
    """The new text for each item (None = drop), after the per-line guards."""
    results: list[str | None] = []
    for local_id, speaker, original in items:
        if local_id not in reply:
            results.append(original)
            continue
        new = reply[local_id]
        if new is None:
            # A question is never content-free, however short (as in merge_tracks).
            if len(_words(original)) > max_drop_words or original.rstrip().endswith("?"):
                stats["drop_refused"] += 1
                results.append(original)
            else:
                stats["dropped"] += 1
                results.append(None)
            continue
        # The model sometimes echoes the speaker despite the prompt; strip it
        # so the guards judge only the text.
        echo = f"{speaker}:"
        if new.startswith(echo) and not original.startswith(echo):
            new = new[len(echo) :].lstrip()
        if (
            word_similarity(original, new) < min_similarity
            or len(new) > 1.5 * len(original) + 40
            or not only_deletes_words(original, new)
        ):
            stats["guard_fallbacks"] += 1
            results.append(original)
            continue
        if new != original:
            stats["rewritten"] += 1
        results.append(new)
    return results


def clean_transcript(
    llm: LocalLLM,
    path: Path,
    model: str,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    max_drop_words: int = DEFAULT_MAX_DROP_WORDS,
) -> dict:
    """Clean `path`, write the `-cleaned.md` and `.changes.md` files next to
    it, and return the stats plus both output paths."""
    frontmatter, body = parse_transcript(path.read_text())
    system_prompt = load_prompt("transcript-cleanup")

    # (body index, line match, bleed prefix, text sent to the model)
    sendable: list[tuple[int, re.Match, str, str]] = []
    for index, line in enumerate(body):
        match = _LINE.match(line)
        if not match:
            continue
        text = match.group("text")
        if text in (INAUDIBLE_PLACEHOLDER, LAUGHTER_MARKER):
            continue
        prefix_match = _BLEED_PREFIX.match(text)
        prefix = prefix_match.group(0) if prefix_match else ""
        sendable.append((index, match, prefix, text[len(prefix) :]))

    stats: Counter = Counter({key: 0 for key in STAT_KEYS})
    stats["lines_sent"] = len(sendable)
    new_body: list[str | None] = list(body)
    dropped: list[str] = []
    rewritten: list[tuple[str, str]] = []

    chunks = [sendable[i : i + chunk_lines] for i in range(0, len(sendable), chunk_lines)]
    for chunk in tqdm.tqdm(chunks, desc="Cleaning", unit="chunk"):
        items = [(n + 1, match.group("speaker"), text) for n, (_, match, _, text) in enumerate(chunk)]
        user_message = build_chunk_message(items)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        max_tokens = 2 * len(llm.tokenizer.encode(user_message)) + 64
        reply = parse_reply(llm.generate(messages, max_tokens))
        results = apply_reply(items, reply, min_similarity, max_drop_words, stats)
        for (index, match, prefix, text), new in zip(chunk, results):
            original_line = body[index]
            if new is None:
                dropped.append(original_line)
                new_body[index] = None
            elif new != text:
                new_line = original_line[: match.start("text")] + prefix + new
                rewritten.append((original_line, new_line))
                new_body[index] = new_line

    cleaned_path = path.with_name(f"{path.stem}-cleaned.md")
    changes_path = path.with_name(f"{path.stem}-cleaned.changes.md")

    out_frontmatter = dict(frontmatter)
    if "title" in out_frontmatter:
        out_frontmatter["title"] = f"{out_frontmatter['title']} (cleaned)"
    out_frontmatter["source"] = path.name
    out_frontmatter["llm_cleanup"] = {
        "model": model,
        "prompt": "settings/prompts/transcript-cleanup.md",
        "chunk_lines": chunk_lines,
        "min_similarity": min_similarity,
        "max_drop_words": max_drop_words,
        **{key: stats[key] for key in STAT_KEYS},
    }
    caveats = list(out_frontmatter.get("caveats") or [])
    caveats.append(
        "Lightly edited by a local LLM (punctuation, fillers, stutters) and some "
        "content-free lines dropped; every change is listed in "
        f"`{changes_path.name}`."
    )
    out_frontmatter["caveats"] = caveats

    dumped = yaml.safe_dump(out_frontmatter, sort_keys=False, allow_unicode=True)
    kept = [line for line in new_body if line is not None]
    cleaned_path.write_text(f"---\n{dumped}---\n" + "\n".join(kept) + "\n")

    changes = ["## Dropped", ""]
    changes += dropped or ["(none)"]
    changes += ["", "## Rewritten", ""]
    if rewritten:
        for original_line, new_line in rewritten:
            changes += [original_line, f"→ {new_line}", ""]
    else:
        changes += ["(none)", ""]
    changes_path.write_text("\n".join(changes).rstrip("\n") + "\n")

    return {**{key: stats[key] for key in STAT_KEYS}, "cleaned_path": cleaned_path, "changes_path": changes_path}
