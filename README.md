# rpg-scribe

A local CLI tool that transcribes a Craig-recorded Discord TTRPG session
(one audio file per speaker + an `info.txt` manifest) into a single
Markdown transcript, chronologically merged across speakers with
per-line timestamps.

Runs fully offline via [`mlx-whisper`](https://github.com/ml-explore/mlx-examples)
on Apple Silicon's Metal GPU — no cloud API calls.

## Setup

The pipeline decodes audio with ffmpeg, so install it first:

```sh
brew install ffmpeg
```

Then set up the Python environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --no-deps -r requirements.lock
cp config.example.yaml config.yaml
```

`--no-deps` matters: mlx-whisper declares torch but doesn't need it for
transcription, and a normal install would pull in ~700MB of torch.

Upgrading an existing install from before torch was dropped: delete `.venv`
and rerun the steps above to reclaim about 700MB.

`requirements.lock` pins the exact tested set. `requirements.txt` lists the
direct dependencies with version ranges. If you change it, regenerate the
lock in a throwaway venv, removing torch and the packages only it pulls in:

```sh
python3 -m venv /tmp/lockvenv
/tmp/lockvenv/bin/pip install -r requirements.txt
/tmp/lockvenv/bin/pip uninstall -y torch sympy mpmath networkx Jinja2 MarkupSafe setuptools
/tmp/lockvenv/bin/pip freeze > requirements.lock
```

Then rebuild `.venv` from the lock:
`rm -rf .venv && python3 -m venv .venv && .venv/bin/pip install --no-deps -r requirements.lock`.
`pip check` will report
`mlx-whisper ... requires torch, which is not installed` — that's expected.

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
Each track's transcribed segments are cached under
`<output_dir>/.partial/<slug>/` and kept after the run. Rerunning the same
zip, including one already in `input/processed/`, reuses them and only
re-runs VAD and merging. Pass `--fresh` to retranscribe from scratch.
Caches unused for `partial_max_age_days` (default 30; `0` turns it off)
are deleted at the start of each full run. `--prune-cache` deletes every
cache and exits, and `--prune-cache DAYS` deletes only the ones unused for
DAYS or more.

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
  `<session-title>-<start>-<end>.md` like `--start`/`--end`.
- `--shared-mic USERNAME@START[-END]=NAME1,NAME2` — label one track's lines
  in a span with several players' names; see
  [Players sharing a mic](#players-sharing-a-mic).
- `--prune-cache [DAYS]` — delete cached segments under
  `<output_dir>/.partial/` (all of them, or only those unused for DAYS) and
  exit.

When two players share a room on separate headsets, each mic picks up the
other, and the transcript can show the same line under both speakers.
Matching cross-track lines are deduplicated before merging, but only when
the audio confirms bleed. The quieter track has to carry an attenuated,
correlated copy of the louder one, so a player echoing someone else is
never removed. The thresholds are config keys (`bleed_level_gap_db`,
`bleed_min_correlation`, `bleed_text_similarity`, `bleed_max_offset_s`;
see `config.example.yaml`). Pairs that match on text but miss the audio
thresholds are kept. `show_bleed_markers: true` (or `--show-bleed-markers`)
prefixes the quieter copy with `[possible bleed: <speaker>]` for review.
`--verbose` prints the level gap and correlation of each pair, so you can
tune the thresholds and rerun the cached zip cheaply.

### Players sharing a mic

If players switch to one mic mid-session (say a headset dies), declare the
span for that run:

```sh
.venv/bin/python transcribe.py session.zip --shared-mic bob42@01:44:00=Bob,Carol
```

The syntax is `--shared-mic USERNAME@START[-END]=NAME1,NAME2[,...]`, with times
in recording time (`HH:MM:SS`, the same clock as `--start`/`--end`). Without
`-END` the span runs to the end of the recording. Repeat the flag for more
spans. Every line from that track in the span is labelled `Bob/Carol`, and
each Whisper segment stays its own line, so you can attribute lines by hand
afterwards. The label is used as-is and bypasses `speaker_map`. Because the
segment cache is kept, adding an override after a run only re-runs VAD and
merging. In the GUI, use the "Shared mic" field, with several entries
separated by `;`.

Whisper's stock silence phrases ("Thanks for watching!", "Subtitles by the
Amara.org community", `www.` URLs) are stripped from each segment before
merging; edit the list with `hallucination_phrases` in `config.yaml`.

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
- Mark possible mic bleed (`--show-bleed-markers` / `--hide-bleed-markers`)
- Shared mic (`--shared-mic`) — several entries separated by `;`

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
used for post-fix replacement only (never seeded into the prompt). A bare
variant is replaced with the term name; `variant -> replacement` (or `→`)
replaces it with that spoken form instead, so a nickname or a short form
doesn't get expanded to the full heading (`Mir -> Mira` under
`### Captain Mira Holt`). Any
other label (`Tagline`, `Description`, etc.) is a free-form enrichment
field. A term may have zero bullets. By convention, `Player: <speaker>`
marks a term as a player character and names who plays it (using the
`speaker_map` name) — `/analyze-narrative` uses this to map players to
characters.

```markdown
# Campaign Vocabulary

## Factions
### Aldric
- Misheard as: Aldrick, All Drick
- Tagline: Captain of the town guard.

### Captain Bertrand Vell
- Misheard as: Bertram -> Bertrand, Captain Bell -> Captain Vell

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
find/replace, regardless of whether `--enrich-vocabulary` is used. All
variants are matched in one longest-first pass, and text already in a
replacement's correct form is left alone — so a short variant like `Ward`
won't fire inside a correct `sky-ward`.
