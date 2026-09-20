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
python3 transcribe.py <craig_folder>
```

`<craig_folder>` is the folder Craig's Discord bot exports for a session
(containing `info.txt` and the per-speaker audio files). The transcript is
written to `output_dir` (from `config.yaml`) as `<session-title>.md`.

Options:

- `--config <path>` — use a config file other than `config.yaml`.
- `--model <repo-id>` — override the configured `whisper_model`.
- `--title <title>` — override the session title (defaults to the Craig
  folder's basename).

On first run, the configured Whisper model is downloaded into `model_dir`
(from `config.yaml`) and reused on subsequent runs.
