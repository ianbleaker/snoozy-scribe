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
(`speaker_map`) and any recurring mis-transcriptions to fix
(`vocabulary`). `config.yaml` is gitignored — it's local to your machine.

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
`<session-title>.md`.

Options:

- `--config <path>` — use a config file other than `config.yaml`.
- `--model <repo-id>` — override the configured `whisper_model`.
- `--title <title>` — override the session title (defaults to the input's
  basename, without the `.zip` extension if applicable).

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
