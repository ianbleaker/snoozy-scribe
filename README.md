# rpg-scribe

A local CLI tool that transcribes a Craig-recorded Discord TTRPG session
(one audio file per speaker + an `info.txt` manifest) into a single
Markdown transcript, chronologically merged across speakers with
per-line timestamps.

Runs fully offline via [`mlx-whisper`](https://github.com/ml-explore/mlx-examples)
on Apple Silicon's Metal GPU — no cloud API calls.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Edit `config.yaml` with your Discord usernames → display name mapping
(`speaker_map`). `config.yaml` is gitignored — it's local to your machine.

Optionally, `cp vocabulary.example.md vocabulary.md` and fill in your
campaign's terms — see [Vocabulary file](#vocabulary-file) below. If you
previously kept a `vocabulary` key in `config.yaml`, move its entries into
`vocabulary.md` in the new format; there's no automatic migration.

## Usage

```sh
python3 transcribe.py [craig_folder]
```

`<craig_folder>` is optional, and accepts either:

- The folder Craig's Discord bot exports for a session (containing
  `info.txt` and the per-speaker audio files) — used as-is.
- A path to a Craig `.zip` export — extracted automatically before
  transcription.

If omitted, the oldest `.zip` file in `input_dir` (from `config.yaml`) is
picked up automatically — drop Craig `.zip` exports there and run without
arguments. After a successful run, a zip that was extracted (whether passed
explicitly or auto-picked) is moved into `<input_dir>/processed/`.

The transcript is written to `output_dir` (from `config.yaml`) as
`<session-title>.md`. Lines Whisper's decode couldn't be trusted for show
`!! [inaudible]` in the transcript; if the run had any, the original
garbled text is written alongside it to `<session-title>.errors.md` for
manual review against the audio.

Options:

- `--config <path>` — use a config file other than `config.yaml`.
- `--model <repo-id>` — override the configured `whisper_model`.
- `--title <title>` — override the session title (defaults to the input's
  basename, without the `.zip` extension if applicable).
- `--vocabulary <path>` — use a vocabulary file other than the configured
  `vocabulary_file` / default `./vocabulary.md`.
- `--enrich-vocabulary [FIELD,...]` — seed Whisper's `initial_prompt` with
  vocabulary enrichment fields in addition to term names. Bare flag includes
  every field found on any term; a comma-separated list (e.g.
  `--enrich-vocabulary Tagline,Description`) includes only those fields.
  Omit the flag entirely to seed only term names, with no enrichment.

On first run, the configured Whisper model is downloaded into `model_dir`
(from `config.yaml`) and reused on subsequent runs.

Config keys:

- `output_dir` — directory transcript Markdown files are written to.
- `input_dir` — directory to drop Craig `.zip` exports into for automatic
  pickup. Defaults to `./input`.
- `vad_min_silence_ms` — minimum pause (in milliseconds) that splits speech
  into separate regions before transcription. Defaults to `800`.
- `timestamp_format` — `strftime()` format for each line's timestamp.
  Defaults to `"%H:%M:%S"`.
- `line_format` — `str.format()` template for each transcript line, with
  `{timestamp}`, `{speaker}`, and `{text}` placeholders. Use Markdown here to
  style the output, e.g. `"[{timestamp}] **{speaker}**: {text}"` to bold
  speaker names. Defaults to `"[{timestamp}] {speaker}: {text}"`.
- `vocabulary_file` — path to a `vocabulary.md` file. Defaults to
  `./vocabulary.md` if it exists; otherwise transcription runs without a
  vocabulary.

## Vocabulary file

`vocabulary.md` (gitignored, like `config.yaml`) holds campaign-specific
terms: post-fix corrections for known Whisper mis-transcriptions, and
optional enrichment fields for prompt seeding. Copy `vocabulary.example.md`
to get started.

Format: `#`/`##` headings are free-form organizational grouping (e.g.
"Factions", "Locations") and are ignored by the parser. `###` marks a term
— term names must be unique across the whole file. Bullets under a term are
`Label: value` pairs. `Misheard as` is the one reserved label — a
comma-separated list of Whisper's known wrong transcriptions of the term,
used for post-fix replacement only (never seeded into the prompt). Any
other label (`Tagline`, `Description`, etc.) is a free-form enrichment
field. A term may have zero bullets.

```markdown
# Campaign Vocabulary

## Factions
### Aldric
- Misheard as: Aldrick, All Drick
- Tagline: Captain of the town guard.

## Locations
### Millhaven
- Misheard as: Mill Haven, Mil Haven
```

At startup, the tool reports how many terms were loaded and which
enrichment field labels were found (with counts), so you know what's
available for `--enrich-vocabulary` without opening the file. A malformed
`vocabulary.md` (duplicate term, unparseable bullet, etc.) is a hard error
— transcription won't start until it's fixed.

Every term name is seeded into Whisper's `initial_prompt` to improve
first-pass transcription accuracy for unfamiliar words. `Misheard as`
variants are additionally corrected post-transcription via case-insensitive
find/replace, regardless of whether `--enrich-vocabulary` is used.
