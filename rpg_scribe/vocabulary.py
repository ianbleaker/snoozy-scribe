"""Vocabulary file parsing and application.

See docs/vocabulary.md for the file format.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from rpg_scribe.paths import VOCABULARY_DIR


class VocabularyParseError(Exception):
    """Raised for a structural or content error in a vocabulary.md file."""


_VARIANT_ARROW = re.compile(r"\s*(?:→|->)\s*")


def parse_vocabulary(path: Path) -> dict[str, dict]:
    """Parse a `vocabulary.md` file into `{term: {misheard_as, fields}}`.

    `misheard_as` is a list of `(variant, replacement)` pairs. A variant
    written `variant -> replacement` (or `→`) maps to that spoken form;
    a bare variant maps to the term name itself.

    See docs/vocabulary.md for the format.
    """
    vocabulary: dict[str, dict] = {}
    current_term: str | None = None

    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("### "):
            name = line[len("### "):].strip()
            if name in vocabulary:
                raise VocabularyParseError(
                    f"{path}:{lineno}: duplicate term '{name}'"
                )
            vocabulary[name] = {"misheard_as": [], "fields": {}}
            current_term = name
            continue
        if line.startswith("# ") or line.startswith("## "):
            continue
        if line.startswith("- "):
            if current_term is None:
                raise VocabularyParseError(
                    f"{path}:{lineno}: bullet found outside any term section: {raw_line!r}"
                )
            body = line[len("- "):]
            if ":" not in body:
                raise VocabularyParseError(
                    f"{path}:{lineno}: malformed bullet (expected 'Label: value'): {raw_line!r}"
                )
            label, value = body.split(":", 1)
            label = label.strip()
            value = value.strip()
            if label.lower() == "misheard as":
                pairs = []
                for item in value.split(","):
                    item = item.strip()
                    if not item:
                        continue
                    parts = [p.strip() for p in _VARIANT_ARROW.split(item, maxsplit=1)]
                    if len(parts) == 2 and not all(parts):
                        raise VocabularyParseError(
                            f"{path}:{lineno}: malformed variant (expected "
                            f"'variant -> replacement'): {item!r}"
                        )
                    pairs.append((parts[0], parts[1] if len(parts) == 2 else current_term))
                vocabulary[current_term]["misheard_as"] = pairs
            elif value:
                vocabulary[current_term]["fields"][label] = value
            continue
        raise VocabularyParseError(f"{path}:{lineno}: malformed line: {raw_line!r}")

    return vocabulary


_SENTENCE_START = re.compile(r"^\s*$|[.!?]\s*$")


def apply_vocabulary(text: str, vocabulary: dict) -> str:
    """Case-insensitively replace known `misheard_as` variants with their replacements.

    Capitalizes only the replacement's first letter, regardless of its own
    casing, when the match starts `text` or follows `.`/`!`/`?` plus
    whitespace; otherwise uses the replacement's exact casing.

    All variants are matched in a single longest-first pass, so a
    replacement is never re-scanned (no "sky-sky-ward"). Replacement forms
    are also matched (and left untouched), so a short variant like "Ward"
    can't fire inside an already-correct "sky-ward". Matches are anchored to word
    boundaries so a variant like "Oste" doesn't fire inside "hosted".
    """
    replacements: dict[str, str] = {}
    for data in vocabulary.values():
        for variant, replacement in data["misheard_as"]:
            replacements.setdefault(variant.lower(), replacement)
    if not replacements:
        return text
    # Already-correct forms map to None: matched (to shield them) but left as-is.
    targets: dict[str, str | None] = dict(replacements)
    for replacement in replacements.values():
        targets.setdefault(replacement.lower(), None)

    alternation = "|".join(
        re.escape(v) for v in sorted(targets, key=len, reverse=True)
    )
    pattern = re.compile(r"\b(?:" + alternation + r")\b", re.IGNORECASE)

    def _replace(match: re.Match) -> str:
        replacement = targets[match.group(0).lower()]
        if replacement is None:
            return match.group(0)
        if _SENTENCE_START.search(text[: match.start()]):
            return replacement[:1].upper() + replacement[1:]
        return replacement

    return pattern.sub(_replace, text)


def build_initial_prompt(vocabulary: dict, enrich_fields: set[str] | None) -> str:
    """Build Whisper's `initial_prompt` from vocabulary term names.

    `enrich_fields is None` -> bare term names only. `enrich_fields == set()`
    -> append every field found per term ("all"). Non-empty `enrich_fields`
    -> append only fields whose label case-insensitively matches an entry.
    """
    parts = []
    for term, data in vocabulary.items():
        fields = data["fields"]
        if enrich_fields is None or not fields:
            parts.append(term)
            continue
        if enrich_fields:
            wanted_lower = {f.lower() for f in enrich_fields}
            values = [v for label, v in fields.items() if label.lower() in wanted_lower]
        else:
            values = list(fields.values())
        if values:
            parts.append(f"{term} ({'; '.join(values)})")
        else:
            parts.append(term)
    return ", ".join(parts)


def list_campaigns() -> list[str]:
    """Campaign names: stems of `VOCABULARY_DIR/*.md`, minus `base` and `*.example`."""
    if not VOCABULARY_DIR.is_dir():
        return []
    return sorted(
        path.stem
        for path in VOCABULARY_DIR.glob("*.md")
        if path.stem != "base" and not path.stem.endswith(".example")
    )


def campaign_path(name: str) -> Path:
    return VOCABULARY_DIR / f"{name}.md"


INBOX_HEADING = "## Inbox"


def _check_text(kind: str, text: str, allow_comma: bool = True) -> None:
    if not text.strip():
        raise ValueError(f"{kind} is empty")
    if "\n" in text or "\r" in text:
        raise ValueError(f"{kind} {text!r} contains a newline")
    if "->" in text or "→" in text:
        raise ValueError(f"{kind} {text!r} contains '->'")
    if not allow_comma and "," in text:
        raise ValueError(f"{kind} {text!r} contains ','")


def _section_end(lines: list[str], start: int) -> int:
    """Index just past the section starting at `start` (next heading or EOF)."""
    for i in range(start + 1, len(lines)):
        if lines[i].lstrip().startswith("#"):
            return i
    return len(lines)


def add_term(
    path: Path,
    term: str,
    misheard_as: list[str] | None = None,
    fields: dict[str, str] | None = None,
) -> None:
    """Add `term` (or new variants/fields for an existing one) to a vocabulary file.

    An existing `### term` (case-sensitive) gets new `misheard_as` variants
    merged into its `- Misheard as:` line, de-duplicated case-insensitively,
    plus any fields whose label it doesn't have yet. A new term is appended
    under `## Inbox`, which is created at the end of the file if missing. A
    missing file is created. The result is written atomically, and only if it
    still parses; otherwise `VocabularyParseError` is raised and the file is
    left untouched. Raises `ValueError` for text that would break the format.
    """
    term = term.strip()
    _check_text("term", term)
    variants = [v.strip() for v in misheard_as or [] if v.strip()]
    for variant in variants:
        _check_text("variant", variant, allow_comma=False)
    fields = {k.strip(): v.strip() for k, v in (fields or {}).items()}
    for label, value in fields.items():
        _check_text("field label", label, allow_comma=False)
        if ":" in label:
            raise ValueError(f"field label {label!r} contains ':'")
        _check_text("field value", value)

    lines = path.read_text().splitlines() if path.exists() else ["# Vocabulary", "", INBOX_HEADING]
    heading = next((i for i, line in enumerate(lines) if line.strip() == f"### {term}"), None)

    if heading is not None:
        end = _section_end(lines, heading)
        misheard_line = next(
            (i for i in range(heading + 1, end) if re.match(r"\s*- misheard as:", lines[i], re.IGNORECASE)),
            None,
        )
        if variants:
            if misheard_line is None:
                lines.insert(heading + 1, "- Misheard as: ")
                misheard_line = heading + 1
                end += 1
            existing = [v.strip() for v in lines[misheard_line].split(":", 1)[1].split(",") if v.strip()]
            seen = {_VARIANT_ARROW.split(v, maxsplit=1)[0].strip().lower() for v in existing}
            for variant in variants:
                if variant.lower() not in seen:
                    existing.append(variant)
                    seen.add(variant.lower())
            lines[misheard_line] = f"- Misheard as: {', '.join(existing)}"
        labels = set()
        for i in range(heading + 1, end):
            match = re.match(r"\s*- ([^:]+):", lines[i])
            if match:
                labels.add(match.group(1).strip().lower())
        new_bullets = [f"- {label}: {value}" for label, value in fields.items() if label.lower() not in labels]
        while end > heading + 1 and not lines[end - 1].strip():
            end -= 1
        lines[end:end] = new_bullets
    else:
        entry = [f"### {term}"]
        if variants:
            entry.append(f"- Misheard as: {', '.join(variants)}")
        entry += [f"- {label}: {value}" for label, value in fields.items()]
        inbox = next((i for i, line in enumerate(lines) if line.strip() == INBOX_HEADING), None)
        if inbox is None:
            while lines and not lines[-1].strip():
                lines.pop()
            lines += ["", INBOX_HEADING]
            inbox = len(lines) - 1
        end = _section_end(lines, inbox)
        # Terms inside the inbox are ### headings, so skip past them to the next ## or #.
        while end < len(lines) and lines[end].lstrip().startswith("### "):
            end = _section_end(lines, end)
        insert_at = end
        while insert_at > inbox + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        block = ["", *entry]
        if insert_at < len(lines) and lines[insert_at].strip():
            block.append("")
        lines[insert_at:insert_at] = block

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines) + "\n")
        try:
            parse_vocabulary(tmp)
        except VocabularyParseError as e:
            raise VocabularyParseError(str(e).replace(str(tmp), str(path))) from None
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
