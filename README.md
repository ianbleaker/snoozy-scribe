# rpg-scribe

A menu-bar app for local push-to-talk dictation on the Mac: hold a key,
speak, release, and cleaned-up text is pasted into the frontmost app.
Everything runs on-device: Whisper transcribes, a small mlx-lm model
(`dictation.cleanup_model`) fixes punctuation, fillers, false starts and
self-corrections ("send it to Bob, no, Alice" → "Send it to Alice."), and
your vocabulary fixes spellings. It also transcribes recorded Discord
sessions (Craig exports) into Markdown transcripts, offline too. How the
dictation app works: [docs/design/dictation.md](docs/design/dictation.md).

## Requirements

- A Mac with Apple Silicon.
- [Homebrew](https://brew.sh).
- Xcode's command-line tools: `xcode-select --install`.
- ffmpeg, only for session transcription: `brew install ffmpeg`.

## Install

```sh
git clone https://github.com/ianbleaker/rpg-scribe.git
cd rpg-scribe
./setup.sh
```

`setup.sh` builds `.venv`, creates `settings/config.yaml` if it doesn't
exist, and builds and installs the app to `~/Applications/rpg-scribe.app`.
Open it from Spotlight; ✎ appears in the menu bar. Rerun `./setup.sh` any
time (after `git pull`, say); it only rebuilds the app when it has to. Pass
`--rebuild-app` to force a rebuild.

## Permissions

On first launch, grant rpg-scribe these permissions when asked (System
Settings → Privacy & Security):

- **Microphone**, to record.
- **Input Monitoring**, for the global hotkeys.
- **Accessibility**, to paste (synthesized ⌘V) and read the focused window
  title.

Input Monitoring and Accessibility are asked for at launch; Microphone is
asked for on your first dictation, so that one records nothing.

Each rebuild changes the app's signature, so grant the permissions again
afterwards. If the hotkey does nothing after granting, quit from the ✎ menu
and reopen.

## Using it

The models load in the background at launch (about 30 s); ✎… shows while
they do. They stay in memory (roughly 3–4 GB) so dictation is fast.

### Hotkeys

- **Hold right Option** (`dictation.hotkey`) to record; release to
  transcribe and paste. A hold counts only if it lasts 200 ms with no other
  key pressed, so ⌥←/⌥→ and Option-typed characters never trigger it.
- **ctrl+Option+D** (`dictation.toggle_hotkey`) starts a hands-free
  recording; press it again to stop. The hold key does nothing meanwhile.
- **Add selected text** can also get a hotkey (`dictation.add_selection_hotkey`).

A pill near the bottom of the screen shows "● Recording" with a level meter,
then "Transcribing…". It never takes focus, so the paste lands where you were
typing. Confirmations ("Added X to mycampaign.md") show there too. If the
focused field is a password field, the text is left on the clipboard instead.

### Menu

- **Campaign ▸**: Off or one of your campaigns. Dictation always uses
  `base.md`, plus the active campaign's terms. The choice is saved in
  `data/dictation-state.json`.
- **History ▸**: the last `dictation.history_size` results (memory only;
  dictated text is never written to disk). Each has "Copy cleaned" and
  "Copy raw", and "(fallback used)" when the cleanup model's output was
  rejected and the raw text was pasted instead.
- **Fix last dictation…**: enter `wrong -> right`. Adds `wrong` as a
  `Misheard as` variant of `right`, then copies the corrected last
  dictation to the clipboard.
- **Add term…**: `Term`, or `Term | misheard1, misheard2`.
- **Add selected text**: copies the selection from the frontmost app and
  opens Add term with it.
- **Transcribe session…** opens the session GUI. **Open settings folder**.
  **Launch at login** adds a LaunchAgent that opens the app at login.

## Config

All keys sit under `dictation:` in `settings/config.yaml` and are optional.
`settings/config.example.yaml` lists each with its default: `hotkey`,
`toggle_hotkey`, `add_selection_hotkey`, `cleanup`, `cleanup_model`,
`inline_fillers`, `app_context`, `overlay` and `history_size`. Dictation
also uses the top-level `whisper_model`, `language`, `model_dir` and
`hallucination_phrases`.

## Vocabulary

Vocabulary files in `settings/vocabulary/` hold terms Whisper gets wrong:
`base.md` is always on for dictation, and a campaign file adds one
campaign's names and places. Format, tiers and capture:
[docs/vocabulary.md](docs/vocabulary.md).

## Transcribing recorded sessions

`transcribe.py` (or the GUI, from **Transcribe session…**) turns a Craig
export of a Discord session into one Markdown transcript, with speakers
merged by time. Setup and usage:
[docs/session-transcription.md](docs/session-transcription.md).

## Troubleshooting

- **Logs**: the app writes to `~/Library/Logs/rpg-scribe.log`.
- **App won't start after a Homebrew Python upgrade, or after moving the
  repo**: rerun `./setup.sh --rebuild-app`, then grant the permissions
  again.
- **Old copy in `/Applications`**: earlier installs had you drag the app
  there, and that copy goes stale. Delete it and use
  `~/Applications/rpg-scribe.app`; if **Launch at login** is on, toggle it
  off and on. `setup.sh` warns when it finds one.

## Development

Lock regeneration, manual setup, the app launcher and testing without the
app: [docs/development.md](docs/development.md).
