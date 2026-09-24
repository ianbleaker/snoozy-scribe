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
pip install -r requirements.lock
cp config.example.yaml config.yaml
```

`requirements.lock` pins the exact tested set. `requirements.txt` lists the
direct dependencies with version ranges. If you change it, rebuild the venv
from scratch, install from `requirements.txt`, and regenerate the lock with
`pip freeze > requirements.lock`.

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
- `--start HH:MM:SS` / `--end HH:MM:SS` — transcribe only part of the
  session. Either bound can be omitted to mean the start or end of the
  recording. Timestamps stay relative to the whole session. The transcript
  is written as `<session-title>-<start>-<end>.md` (e.g.
  `my-session-01h30m-02h15m.md`) unless `--output-name` is given. A windowed
  run doesn't save resumable progress and doesn't move the zip into
  `processed/`, so you can come back for other windows or a full run later.
- `--test [SECONDS]` — a quick windowed run of SECONDS (default 120),
  starting at `--start` if given or at a random point otherwise, written to
  `<session-title>-test.md`.

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
- `timestamp_mode` — `"elapsed"` stamps each line with time since the start
  of the recording, so the first segment reads `00:00:00`; `"clock"` stamps
  it with the real wall-clock time it was recorded. Defaults to `"elapsed"`.
  Overridden by `--elapsed-timestamps` / `--clock-timestamps`.
- `line_format` — `str.format()` template for each transcript line, with
  `{timestamp}`, `{speaker}`, and `{text}` placeholders. Use Markdown here to
  style the output, e.g. `"[{timestamp}] **{speaker}**: {text}"` to bold
  speaker names. Defaults to `"[{timestamp}] {speaker}: {text}"`.
- `vocabulary_file` — path to a `vocabulary.md` file. Defaults to
  `./vocabulary.md` if it exists; otherwise transcription runs without a
  vocabulary.

## GUI

```sh
python3 gui.py
```

A basic desktop window you can use instead of the CLI. It runs
`transcribe.py` for you and shows a progress bar, the elapsed time, and its
log output (collapsed by default; it opens by itself if a run fails). It exposes only a subset of the CLI's options:

- Craig export — a `.zip` or an export folder. Leave it blank to pick up the
  oldest `.zip` in `input_dir`, as the CLI does.
- Output name (`--output-name`)
- Time window — start and end as `HH:MM:SS` (`--start` / `--end`). Leave
  both blank to transcribe the whole session.
- Model — q4 / turbo / large-v3 (`--model`)
- Timestamp mode — elapsed or wall-clock (`--elapsed-timestamps` /
  `--clock-timestamps`)
- Show `[inaudible]` placeholders (`--show-inaudible` / `--hide-inaudible`)

Everything else still comes from `config.yaml`. Advanced flags (`--language`,
`--test`, `--fresh`, `--vocabulary`, thresholds, etc.) need the CLI. Cancel
stops the run after a confirmation, and progress on completed tracks is
saved (except in windowed runs), just like the CLI's Escape-then-y. When a
run finishes, the window offers buttons to open the transcript or reveal it
in Finder, and its log is saved next to the transcript as
`<session-title>.log` (uncheck "Save log next to the transcript", or set
`save_gui_log: false` in `config.yaml`, to skip it).

To launch it with a double-click instead of from a terminal, build a macOS
app once:

```sh
./make-app.sh
```

That creates `rpg-scribe.app` (gitignored) in the repo. Drag it to the Dock
or Applications. It runs `gui.py` with this repo's `.venv`, so there's no need
to activate anything. The repo's path is baked in, so rerun `./make-app.sh` if
you move the repo. If the window doesn't appear, check
`~/Library/Logs/rpg-scribe.log`.

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
