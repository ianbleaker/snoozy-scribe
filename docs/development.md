# Development

## Repo layout

```
transcribe.py  gui.py  dictate.py               entry points
setup.sh       make-app.sh                       install, app bundle build
settings/      config.yaml, vocabulary/          your settings (gitignored, *.example.* tracked)
data/          input/, transcripts/, models/     your data (gitignored)
rpg_scribe/    internal Python
```

Relative paths in `settings/config.yaml` resolve against the repo root, not
the current directory.

## Manual setup

`./setup.sh` is the supported install. If it can't run, these are the steps
it takes, in order, from the repo root:

1. Check for Apple Silicon, Homebrew and the Xcode command-line tools.
2. Create `.venv` with Homebrew's `python3` (Apple's `/usr/bin/python3` lacks
   the `python3-config --embed` that `make-app.sh` needs), or recreate it if
   it no longer starts:
   `"$(brew --prefix)/bin/python3" -m venv .venv`
3. `.venv/bin/pip install --no-deps -r requirements.lock`
4. `cp settings/config.example.yaml settings/config.yaml`, if it doesn't exist.
5. Warn if `ffmpeg` is missing (`brew install ffmpeg`; only session
   transcription needs it).
6. If `~/Applications/rpg-scribe.app` is missing, the venv was recreated, or
   `--rebuild-app` was passed: refuse while the app is running, run
   `./make-app.sh`, and copy the bundle with
   `ditto rpg-scribe.app ~/Applications/rpg-scribe.app`, then clear the
   old grants with `tccutil reset <service> local.rpg-scribe.gui` for
   Microphone, ListenEvent, Accessibility and PostEvent.
7. Warn about a stale copy in `/Applications`.

The app's name, bundle ID, LaunchAgent label and log path all come from
`rpg_scribe/identity.py`; `make-app.sh` and `setup.sh` read them with
`.venv/bin/python -m rpg_scribe.identity --shell`.

`--no-deps` matters: mlx-whisper declares torch but doesn't need it for
transcription, and a normal install would pull in ~700MB of torch.

Upgrading an existing install from before torch was dropped: delete `.venv`
and rerun the steps above to reclaim about 700MB.

## The app bundle

The app bundle's executable is a small compiled launcher that runs
`dictate.py` on this repo's `.venv`, with the repo path baked in. It needs
Xcode's command-line tools (`xcode-select --install`) for clang. Output goes to
`~/Library/Logs/rpg-scribe.log`.

**Rerun `./setup.sh --rebuild-app` after moving the repo or upgrading Homebrew's
Python.** Each rebuild changes the app's signature, so grant the permissions
again afterwards.

## Dependencies and the lock

`requirements.lock` pins the exact tested set. `requirements.txt` lists the
direct dependencies with version ranges. If you change it, regenerate the
lock in a throwaway venv, removing torch and the packages only it pulls in.
Passing the current lock as a constraint (`-c`) keeps already-pinned packages
at their tested versions and only adds what's new:

```sh
python3 -m venv /tmp/lockvenv
/tmp/lockvenv/bin/pip install -r requirements.txt -c requirements.lock
/tmp/lockvenv/bin/pip uninstall -y torch sympy mpmath networkx setuptools
/tmp/lockvenv/bin/pip freeze > requirements.lock
```

Jinja2 and MarkupSafe stay in the lock: transformers (pulled in by mlx-lm)
needs jinja2 to render the dictation cleanup model's chat template.

Then rebuild `.venv` from the lock:
`rm -rf .venv && python3 -m venv .venv && .venv/bin/pip install --no-deps -r requirements.lock`.
`pip check` will report
`mlx-whisper ... requires torch, which is not installed` — that's expected.

## Testing dictation without the app

Test the pipeline without the app:
`.venv/bin/python dictate.py --once clip.wav` prints the `RAW:` and `CLEAN:`
text. `--print-context` prints the frontmost app and window after 3 s.
