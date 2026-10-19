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
6. Create the `rpg-scribe Code Signing` certificate in the login keychain
   if `security find-identity -v -p codesigning` doesn't list it (see
   [Code signing](#code-signing)). If that fails, warn and carry on; the app
   is then signed ad-hoc.
7. If `~/Applications/rpg-scribe.app` is missing, the venv was recreated,
   `--rebuild-app` was passed, or the installed copy is ad-hoc signed while
   the certificate exists: refuse while the app is running, run
   `./make-app.sh`, and copy the bundle with
   `ditto rpg-scribe.app ~/Applications/rpg-scribe.app`. If the designated
   requirement changed (`codesign -d -r-`), or the build is ad-hoc, clear the
   old grants with `tccutil reset <service> local.rpg-scribe.gui` for
   Microphone, ListenEvent, Accessibility and PostEvent.
8. Warn about a stale copy in `/Applications`.

The app's name, bundle ID, LaunchAgent label, log path and certificate name
all come from
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
Python.** With the certificate, the permission grants survive the rebuild.

### Code signing

`make-app.sh` signs with the `rpg-scribe Code Signing` certificate when
`security find-identity -v -p codesigning` lists it, and ad-hoc otherwise.
macOS keys the permission grants to the signature's designated requirement.
For a certificate signature that names the certificate, so grants survive
rebuilds. For an ad-hoc one it's the build's code hash, so every rebuild
loses them. Switching an install from ad-hoc to the certificate means
granting once more.

`setup.sh` creates the certificate once: an RSA key and self-signed cert with
the codeSigning extended key usage from Apple's `/usr/bin/openssl`
(Homebrew's OpenSSL 3 writes `.p12` files `security import` can't read),
imported with `security import -T /usr/bin/codesign`, then trusted with
`security add-trusted-cert -r trustRoot -p codeSign`, which asks for your
password. It never regenerates an existing certificate, since a new one
would lose the grants again.

If scripting it fails, create it by hand: Keychain Access → Certificate
Assistant → Create a Certificate…, name `rpg-scribe Code Signing`, Identity
Type *Self-Signed Root*, Certificate Type *Code Signing*. Then open the
certificate, set Trust → Code Signing to *Always Trust*, and rerun
`./setup.sh`. Check it with `security find-identity -v -p codesigning`.

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
