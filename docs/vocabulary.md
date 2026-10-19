# Vocabulary

Vocabulary files live in `settings/vocabulary/` (gitignored, except the
`*.example.md` files). They hold terms Whisper gets wrong: post-fix
corrections for known mis-transcriptions, and optional enrichment fields for
prompt seeding. There are two tiers:

- **Campaign files**, `settings/vocabulary/<campaign>.md`: names and places
  for one campaign. Any `*.md` there except `base.md` and `*.example.md` is a
  campaign. Session transcription uses **only** the campaign file (config
  `campaign` or `--campaign`), so its terms get Whisper's whole ~224-token
  prompt. Start from `campaign.example.md`.
- **`base.md`**: terms that are always on for dictation (tool names, LLM
  jargon). Session transcription doesn't use it. Start from
  `base.example.md`.

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
field; one with a blank value is ignored. A term may have zero bullets. By convention, `Player: <speaker>`
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
vocabulary file (duplicate term, unparseable bullet, etc.) is a hard error
— transcription won't start until it's fixed.

Every term name is seeded into Whisper's `initial_prompt` to improve
first-pass transcription accuracy for unfamiliar words. `Misheard as`
variants are additionally corrected post-transcription via case-insensitive
find/replace, regardless of whether `--enrich-vocabulary` is used. All
variants are matched in one longest-first pass, and text already in a
replacement's correct form is left alone — so a short variant like `Ward`
won't fire inside a correct `sky-ward`.

## Capturing terms from the menu

The dictation app's menu has **Fix last dictation…**, **Add term…** and
**Add selected text** (see the [README](../README.md#menu)).

The capture dialogs ask which file to write: **Base** or the active
campaign. New terms go under a `## Inbox` heading at the end of that file
for you to re-file later. Edits to vocabulary files take effect on the next
dictation; a file that fails to parse keeps its last good version and shows
the error.
