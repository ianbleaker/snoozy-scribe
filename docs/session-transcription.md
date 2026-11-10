# Session transcription

A local CLI tool that transcribes a Craig-recorded Discord TTRPG session (one audio file per speaker + an `info.txt` manifest) into a single Markdown transcript, chronologically merged across speakers with per-line timestamps.

Runs fully offline via [`mlx-whisper`](https://github.com/ml-explore/mlx-examples) on Apple Silicon's Metal GPU — no cloud API calls.

The dictation app is covered in the [README](../README.md).

## Setup

The pipeline decodes audio with ffmpeg, so install it first:

```sh
brew install ffmpeg
```

Edit `settings/config.yaml` with your Discord usernames → display name mapping (`speaker_map`). It's gitignored — local to your machine.

Optionally, `cp settings/vocabulary/campaign.example.md settings/vocabulary/<campaign>.md`, fill in your campaign's terms and set `campaign: <campaign>` in the config — see [vocabulary.md](vocabulary.md).

**Upgrading from the old layout** (`config.yaml`, `vocabulary.md`, `models/`, `input/` and `transcripts/` at the repo root): `transcribe.py`, `gui.py` and the dictation app check for it at startup. They exit and print the exact `mv` commands and config keys to fix. Nothing is moved automatically.

## Usage

```sh
python3 transcribe.py [craig_folder]
```

`<craig_folder>` is optional, and accepts either:

- The folder Craig's Discord bot exports for a session (containing `info.txt` and the per-speaker audio files) — used as-is.
- A path to a Craig `.zip` export — extracted automatically before transcription.

If omitted, the oldest `.zip` file in `input_dir` (`data/input/` by default) is picked up automatically — drop Craig `.zip` exports there and run without arguments. After a successful run, a zip that was extracted (whether passed explicitly or auto-picked) is moved into `<input_dir>/processed/`. Each track's transcribed segments are cached under `<output_dir>/.partial/<slug>/` and kept after the run. Rerunning the same zip, including one already in `data/input/processed/`, reuses them and only re-runs VAD and merging. Pass `--fresh` to retranscribe from scratch. If the cache was built with different settings (for example another `--model`), the run stops and lists what differs instead of discarding it. Caches unused for `partial_max_age_days` (default 30; `0` turns it off) are deleted at the start of each full run. `--prune-cache` deletes every cache and exits, and `--prune-cache DAYS` deletes only the ones unused for DAYS or more.

The transcript is written to `output_dir` (`data/transcripts/` by default) as `<session-title>.md`. Lines Whisper's decode couldn't be trusted for show `!! [inaudible]` in the transcript; if the run had any, the original garbled text is written alongside it to `<session-title>.errors.md` for manual review against the audio. Lines deliberately dropped as filler, crosstalk backchannels or laughter are logged, with the reason, to `<session-title>.pruned.md`.

Options:

- `--config <path>` — use a config file other than `settings/config.yaml`.
- `--model <repo-id>` — override the configured `whisper_model`.
- `--title <title>` — override the session title (defaults to the input's basename, without the `.zip` extension if applicable).
- `--campaign <name>` — use `settings/vocabulary/<name>.md` as the vocabulary, overriding the config `campaign` key. An unknown name exits with the list of available campaigns.
- `--vocabulary <path>` — use this vocabulary file instead (overrides `--campaign` and `campaign`).
- `--enrich-vocabulary [FIELD,...]` — seed Whisper's `initial_prompt` with vocabulary enrichment fields in addition to term names. Bare flag includes every field found on any term; a comma-separated list (e.g. `--enrich-vocabulary Tagline,Description`) includes only those fields. Omit the flag entirely to seed only term names, with no enrichment.
- `--start HH:MM:SS` / `--end HH:MM:SS` — transcribe only part of the session. Either bound can be omitted to mean the start or end of the recording. Timestamps stay relative to the whole session. The transcript is written as `<session-title>-<start>-<end>.md` (e.g. `my-session-01h30m-02h15m.md`) unless `--output-name` is given. A windowed run doesn't save resumable progress and doesn't move the zip into `processed/`, so you can come back for other windows or a full run later.
- `--test [SECONDS]` — a quick windowed run of SECONDS (default 120), starting at `--start` if given or at a random point otherwise, written to `<session-title>-<start>-<end>.md` like `--start`/`--end`.
- `--shared-mic USERNAME@START[-END]=NAME1,NAME2` — label one track's lines in a span with several players' names; see [Players sharing a mic](#players-sharing-a-mic).
- `--prune-cache [DAYS]` — delete cached segments under `<output_dir>/.partial/` (all of them, or only those unused for DAYS) and exit.

When two players share a room on separate headsets, each mic picks up the other, and the transcript can show the same line under both speakers. Matching cross-track lines are deduplicated before merging, but only when the audio confirms bleed. The quieter track has to carry an attenuated, correlated copy of the louder one, so a player echoing someone else is never removed. The thresholds are config keys (`bleed_level_gap_db`, `bleed_min_correlation`, `bleed_text_similarity`, `bleed_max_offset_s`; see `settings/config.example.yaml`). Pairs that match on text but miss the audio thresholds are kept. `show_bleed_markers: true` (or `--show-bleed-markers`) prefixes the quieter copy with `[possible bleed: <speaker>]` for review. `--verbose` prints the level gap and correlation of each pair, so you can tune the thresholds and rerun the cached zip cheaply.


### Players sharing a mic

If players switch to one mic mid-session (say a headset dies), declare the span for that run:

```sh
.venv/bin/python transcribe.py session.zip --shared-mic bob42@01:44:00=Bob,Carol
```

The syntax is `--shared-mic USERNAME@START[-END]=NAME1,NAME2[,...]`, with times in recording time (`HH:MM:SS`, the same clock as `--start`/`--end`). Without `-END` the span runs to the end of the recording. Repeat the flag for more spans. Every line from that track in the span is labelled `Bob/Carol`, and each Whisper segment stays its own line, so you can attribute lines by hand afterwards. The label is used as-is and bypasses `speaker_map`. Because the segment cache is kept, adding an override after a run only re-runs VAD and merging. In the GUI, use the "Shared mic" field, with several entries separated by `;`.

Whisper's stock silence phrases ("Thanks for watching!", "Subtitles by the Amara.org community", `www.` URLs) are stripped from each segment before merging; edit the list with `hallucination_phrases` in `settings/config.yaml`.

On first run, the configured Whisper model is downloaded into `model_dir` (`data/models/` by default) and reused on subsequent runs.

Config keys:

- `output_dir` — directory transcript Markdown files are written to. Defaults to `./data/transcripts`.
- `input_dir` — directory to drop Craig `.zip` exports into for automatic pickup. Defaults to `./data/input`.
- `model_dir` — where model weights are downloaded. Defaults to `./data/models`.
- `vad_min_silence_ms` — minimum pause (in milliseconds) that splits speech into separate regions before transcription. Defaults to `800`.
- `timestamp_format` — `strftime()` format for each line's timestamp. Defaults to `"%H:%M:%S"`.
- `timestamp_mode` — `"elapsed"` stamps each line with time since the start of the recording, so the first segment reads `00:00:00`; `"clock"` stamps it with the real wall-clock time it was recorded. Defaults to `"elapsed"`. Overridden by `--elapsed-timestamps` / `--clock-timestamps`.
- `line_format` — `str.format()` template for each transcript line, with `{timestamp}`, `{speaker}`, and `{text}` placeholders. Use Markdown here to style the output, e.g. `"[{timestamp}] **{speaker}**: {text}"` to bold speaker names. Defaults to `"[{timestamp}] {speaker}: {text}"`.
- `campaign` — the campaign whose vocabulary (`settings/vocabulary/<campaign>.md`) session transcription uses. Omit it to run without a vocabulary.


## LLM cleanup

```sh
.venv/bin/python clean_transcript.py data/transcripts/<session-title>.md
```

An optional second pass over a finished transcript, using the same local model as dictation cleanup (no network calls once it's downloaded). For each line the model either drops it (content-free lines and false starts the speaker restarts) or lightly edits it: punctuation, capitalization, um/uh, stutters. It never rephrases. Per-line guards keep the original whenever a rewrite drifts or a long line would be dropped. Design: [transcription-pipeline.md](design/transcription-pipeline.md#llm-cleanup-pass-transcript_cleanuppy).

It writes two files next to the input, overwriting any from an earlier run:

- `<session-title>-cleaned.md` — the cleaned transcript. Its frontmatter adds `source`, an `llm_cleanup` block (model, settings and counts) and a caveat. This is the input `/analyze-narrative` expects.
- `<session-title>-cleaned.changes.md` — every dropped line, and every rewrite as the original line followed by `→ <new line>`.

A full session (about 2,000 lines) takes several minutes. `--config <path>` uses a config file other than `settings/config.yaml`.

Config keys, all optional, under `transcript_cleanup:`:

- `model` — mlx-lm instruct model. Defaults to dictation's default cleanup model.
- `chunk_lines` — lines per model call. Defaults to `30`.
- `min_similarity` — a rewrite less similar than this to the original is discarded. Defaults to `0.8`.
- `max_drop_words` — a line with more words than this is never dropped. Defaults to `6`. A line ending in `?` is never dropped either.

The prompt is `settings/prompts/transcript-cleanup.md`; edit it to tune what gets dropped or edited.

## GUI

```sh
python3 gui.py
```

A basic desktop window you can use instead of the CLI. It runs `transcribe.py` for you and shows a progress bar, the elapsed time, and its log output (collapsed by default; it opens by itself if a run fails). It exposes only a subset of the CLI's options:

- Craig export — a `.zip` or an export folder. Leave it blank to pick up the oldest `.zip` in `input_dir`, as the CLI does.
- Output name (`--output-name`)
- Time window — start and end as `HH:MM:SS` (`--start` / `--end`). Leave both blank to transcribe the whole session.
- Model — q4 / turbo / large-v3 (`--model`)
- Timestamp mode — elapsed or wall-clock (`--elapsed-timestamps` / `--clock-timestamps`)
- Show `[inaudible]` placeholders (`--show-inaudible` / `--hide-inaudible`)
- Mark possible mic bleed (`--show-bleed-markers` / `--hide-bleed-markers`)
- Shared mic (`--shared-mic`) — several entries separated by `;`

Everything else still comes from `settings/config.yaml`, including the `campaign`. Advanced flags (`--language`, `--campaign`, `--test`, `--fresh`, `--vocabulary`, thresholds, etc.) need the CLI. Cancel stops the run after a confirmation, and progress on completed tracks is saved (except in windowed runs), just like the CLI's Escape-then-y. When a run finishes, the window offers buttons to open the transcript or reveal it in Finder, and its log is saved next to the transcript as `<session-title>.log` (uncheck "Save log next to the transcript", or set `save_gui_log: false` in the config, to skip it).

The GUI also opens from the dictation app's menu ("Transcribe session…").
