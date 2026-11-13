"""Session transcription window (AppKit).

Runs transcribe.py as a subprocess with `--progress-events`, streaming its
stdout into a log pane and its `@@PROGRESS@@` JSON events into a progress
bar. Exposes only a subset of the CLI's flags; see
docs/session-transcription.md.

Hosted in-process by the menu-bar app (dictate.py), or standalone by gui.py.
Main thread only, except the reader threads, which hand everything to the
main thread with `callAfter`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import AppKit
import Foundation
import yaml
from PyObjCTools.AppHelper import callAfter

from snoozy_scribe import cocoa_ui, identity, paths
from snoozy_scribe.audio_input import AUDIO_EXTENSIONS, InputError, InputKind, detect_input

CODE_DIR = paths.CODE_DIR
PROGRESS_EVENT_PREFIX = "@@PROGRESS@@ "
# Tail of the child's stderr kept for a failure dump; tqdm redraws land
# there too, so an uncapped buffer would grow for the whole run.
STDERR_TAIL_LINES = 500

# (label, --model repo).
MODELS = [
    ("q4", "mlx-community/whisper-large-v3-turbo-q4"),
    ("turbo", "mlx-community/whisper-large-v3-turbo"),
    ("large-v3", "mlx-community/whisper-large-v3-mlx"),
]

WINDOW_WIDTH = 760
MIN_WIDTH = 560
# Minimum window height while the log is expanded.
MIN_EXPANDED_HEIGHT = 520
MARGIN = 16
PHASE_LABELS = {"vad": "Analyzing speech", "transcribe": "Transcribing"}
INPUT_FILETYPES = ["zip", *(ext.lstrip(".") for ext in sorted(AUDIO_EXTENSIONS))]

_ON = AppKit.NSControlStateValueOn
_OFF = AppKit.NSControlStateValueOff


def _load_config() -> dict:
    config_path = paths.DEFAULT_CONFIG_PATH
    try:
        config = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return config if isinstance(config, dict) else {}


def _hstack(views, fill: bool = False):
    stack = AppKit.NSStackView.stackViewWithViews_(views)
    stack.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationHorizontal)
    stack.setSpacing_(6)
    if fill:
        stack.setDistribution_(AppKit.NSStackViewDistributionFill)
    stack.setContentHuggingPriority_forOrientation_(
        AppKit.NSLayoutPriorityDefaultHigh, AppKit.NSLayoutConstraintOrientationVertical
    )
    return stack


def _stretch(view) -> None:
    """Let `view` take the spare width in a filling horizontal stack."""
    view.setContentHuggingPriority_forOrientation_(1, AppKit.NSLayoutConstraintOrientationHorizontal)


def _fixed_width(view, width: float) -> None:
    view.widthAnchor().constraintEqualToConstant_(width).setActive_(True)


def _form_label(text: str):
    """A label for the form's first column, which takes no spare width."""
    field = cocoa_ui.label(text)
    field.setContentHuggingPriority_forOrientation_(
        AppKit.NSLayoutPriorityDefaultHigh, AppKit.NSLayoutConstraintOrientationHorizontal
    )
    return field


class TranscribeWindow:
    """Controller for the transcription window.

    `standalone` is True under gui.py: closing the window quits. In the
    menu-bar app, closing during a run only hides the window; the run goes
    on and "Transcribe audio…" brings it back.
    """

    def __init__(self, standalone: bool) -> None:
        self._standalone = standalone
        self._window = None
        self._proc: subprocess.Popen | None = None
        self._elapsed_timer = cocoa_ui.Repeater(1, self._tick_elapsed)
        # Quitting the host app mustn't leave transcribe.py running. The
        # notification center holds its observer weakly, like the targets.
        self._terminate_observer = cocoa_ui.ActionTarget(lambda _n: self._terminate_child())
        Foundation.NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
            self._terminate_observer, "perform:", AppKit.NSApplicationWillTerminateNotification, None
        )

    # --- show / build -----------------------------------------------------

    def show(self) -> None:
        AppKit.NSApplication.sharedApplication()
        # A window closed while idle is rebuilt, so it re-reads config.yaml
        # like a fresh process would; one hidden mid-run keeps its progress.
        if self._window is None or (not self._window.isVisible() and self._proc is None):
            self._build()
        cocoa_ui.ensure_main_menu(identity.APP_NAME)
        if not self._standalone:
            cocoa_ui.set_regular(True)
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        self._window.makeKeyAndOrderFront_(None)

    def _build(self) -> None:
        paths.ensure_user_dirs()
        config = _load_config()
        self._input_dir = paths.resolve(config.get("input_dir", paths.DEFAULT_INPUT_DIR))
        default_output_dir = paths.resolve(config.get("output_dir", paths.DEFAULT_OUTPUT_DIR))
        model_repos = [repo for _, repo in MODELS]
        default_model = config.get("whisper_model")

        # AppKit holds targets and delegates weakly; this keeps them alive.
        self._targets: list = []
        self._cancelling = False
        self._result: dict | None = None
        self._output_path: Path | None = None
        # Set when a --separate run finished with several transcripts.
        self._batch_output_dir: Path | None = None
        self._run_output_dir: Path | None = None
        # The input's detected kind, or None when blank or unusable.
        self._input_kind: InputKind | None = None
        self._run_start = 0.0
        # Windowed runs don't save partial output, so cancelling one loses it.
        self._windowed = False
        # Window height to restore when the log is expanded again.
        self._expanded_height = 860
        self._model = default_model if default_model in model_repos else model_repos[0]

        style = (
            AppKit.NSWindowStyleMaskTitled
            | AppKit.NSWindowStyleMaskClosable
            | AppKit.NSWindowStyleMaskMiniaturizable
            | AppKit.NSWindowStyleMaskResizable
        )
        window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(0, 0, WINDOW_WIDTH, 600), style, AppKit.NSBackingStoreBuffered, False
        )
        window.setTitle_(f"{identity.APP_NAME} — Transcribe audio")
        window.setReleasedWhenClosed_(False)
        self._delegate = cocoa_ui.Delegate(should_close=self._should_close, will_close=self._will_close)
        window.setDelegate_(self._delegate)
        self._window = window

        def checkbox(title: str, on: bool, fn=lambda _s: None):
            box = cocoa_ui.checkbox(title, fn, self._targets)
            box.setState_(_ON if on else _OFF)
            return box

        # --- form ---
        self._input_delegate = cocoa_ui.Delegate(text_changed=lambda _n: self._refresh_detected())
        self.input_entry = cocoa_ui.text_field("Blank = oldest .zip in input_dir", self._input_delegate)
        _stretch(self.input_entry)
        self.file_button = cocoa_ui.button("File…", lambda _s: self._pick_file(), self._targets)
        self.folder_button = cocoa_ui.button("Folder…", lambda _s: self._pick_folder(), self._targets)
        self.detected_label = cocoa_ui.label("", secondary=True)
        # Input errors can be long; wrap them rather than widen the window.
        self.detected_label.setLineBreakMode_(AppKit.NSLineBreakByWordWrapping)
        self.detected_label.setMaximumNumberOfLines_(0)
        self.detected_label.setContentCompressionResistancePriority_forOrientation_(
            AppKit.NSLayoutPriorityDefaultLow, AppKit.NSLayoutConstraintOrientationHorizontal
        )

        self.output_dir_entry = cocoa_ui.text_field()
        self.output_dir_entry.setStringValue_(str(default_output_dir))
        _stretch(self.output_dir_entry)
        self.output_dir_button = cocoa_ui.button("Browse…", lambda _s: self._pick_output_dir(), self._targets)

        self.output_entry = cocoa_ui.text_field("Blank = slugified session title")

        self.start_entry = cocoa_ui.text_field("Start (HH:MM:SS)")
        _fixed_width(self.start_entry, 140)
        self.end_entry = cocoa_ui.text_field("End (HH:MM:SS)")
        _fixed_width(self.end_entry, 140)
        window_row = _hstack(
            [self.start_entry, cocoa_ui.label("to"), self.end_entry, cocoa_ui.label("Blank = whole session", secondary=True)]
        )

        self.shared_mic_entry = cocoa_ui.text_field("e.g. bob42@01:44:00=Bob,Carol — separate several with ;")

        self.separate_check = checkbox(
            "Transcribe each file separately", False, lambda _s: self._apply_input_rules()
        )

        self.speaker_entry = cocoa_ui.text_field("Blank = Speaker")

        self.model_radios = []
        for label, repo in MODELS:
            radio = cocoa_ui.radio(label, lambda _s, r=repo: self._set_model(r), self._targets)
            radio.setState_(_ON if repo == self._model else _OFF)
            self.model_radios.append(radio)
        model_row = _hstack(self.model_radios)
        model_row.setSpacing_(16)

        self.elapsed_check = checkbox(
            "Elapsed timestamps (uncheck for wall-clock time)",
            config.get("timestamp_mode", "elapsed") == "elapsed",
        )
        self.inaudible_check = checkbox(
            "Show [inaudible] placeholders", bool(config.get("show_inaudible_placeholders", False))
        )
        self.bleed_check = checkbox("Mark possible mic bleed", bool(config.get("show_bleed_markers", False)))
        self.save_log_check = checkbox("Save log next to the transcript", bool(config.get("save_gui_log", True)))

        empty = AppKit.NSGridCell.emptyContentView
        rows = [
            (_form_label("Audio"), _hstack([self.input_entry, self.file_button, self.folder_button], fill=True)),
            (empty(), self.detected_label),
            (_form_label("Output folder"), _hstack([self.output_dir_entry, self.output_dir_button], fill=True)),
            (_form_label("Output name"), self.output_entry),
            (_form_label("Time window"), window_row),
            (_form_label("Shared mic"), self.shared_mic_entry),
            (empty(), self.separate_check),
            (_form_label("Speaker name"), self.speaker_entry),
            (_form_label("Model"), model_row),
            (empty(), self.elapsed_check),
            (empty(), self.inaudible_check),
            (empty(), self.bleed_check),
            (empty(), self.save_log_check),
        ]
        form = AppKit.NSGridView.gridViewWithViews_([list(row) for row in rows])
        form.setRowSpacing_(8)
        form.setColumnSpacing_(8)
        form.setYPlacement_(AppKit.NSGridCellPlacementCenter)
        label_column = form.columnAtIndex_(0)
        label_column.setXPlacement_(AppKit.NSGridCellPlacementTrailing)
        # Otherwise the grid hands the label column the spare width.
        label_column.setWidth_(max(label.fittingSize().width for label, _ in rows))
        form.columnAtIndex_(1).setXPlacement_(AppKit.NSGridCellPlacementFill)
        # Rows that shouldn't stretch across the window.
        for view in (
            window_row,
            self.separate_check,
            model_row,
            self.elapsed_check,
            self.inaudible_check,
            self.bleed_check,
            self.save_log_check,
        ):
            form.cellForView_(view).setXPlacement_(AppKit.NSGridCellPlacementLeading)
        form.setContentHuggingPriority_forOrientation_(
            AppKit.NSLayoutPriorityDefaultHigh, AppKit.NSLayoutConstraintOrientationVertical
        )

        # --- controls ---
        self.start_button = cocoa_ui.button("Start", lambda _s: self._start(), self._targets)
        self.cancel_button = cocoa_ui.button("Cancel", lambda _s: self._cancel(), self._targets)
        self.open_transcript_button = cocoa_ui.button(
            "Open transcript", lambda _s: self._open_transcript(), self._targets
        )
        self.open_folder_button = cocoa_ui.button("Open folder", lambda _s: self._open_folder(), self._targets)
        controls = _hstack(
            [self.start_button, self.cancel_button, self.open_transcript_button, self.open_folder_button]
        )
        controls.setSpacing_(8)

        # --- status ---
        self.progress_bar = AppKit.NSProgressIndicator.alloc().init()
        self.progress_bar.setStyle_(AppKit.NSProgressIndicatorStyleBar)
        self.progress_bar.setIndeterminate_(False)
        self.progress_bar.setMinValue_(0)
        self.progress_bar.setMaxValue_(100)
        self.progress_bar.setDoubleValue_(0)
        self.status_label = cocoa_ui.label("Idle")
        self.elapsed_label = cocoa_ui.label("")
        status_row = _hstack([])
        status_row.addView_inGravity_(self.status_label, AppKit.NSStackViewGravityLeading)
        status_row.addView_inGravity_(self.elapsed_label, AppKit.NSStackViewGravityTrailing)

        # --- log ---
        self.log_toggle = cocoa_ui.button(
            "", lambda _s: self._set_log_visible(self._log_scroll.isHidden()), self._targets
        )
        self.log_toggle.setBordered_(False)
        self._log_scroll = AppKit.NSTextView.scrollableTextView()
        self._log_scroll.setBorderType_(AppKit.NSBezelBorder)
        self._log_scroll.setContentHuggingPriority_forOrientation_(1, AppKit.NSLayoutConstraintOrientationVertical)
        self._log_scroll.setContentCompressionResistancePriority_forOrientation_(
            AppKit.NSLayoutPriorityDefaultLow, AppKit.NSLayoutConstraintOrientationVertical
        )
        self._log_scroll.heightAnchor().constraintGreaterThanOrEqualToConstant_(120).setActive_(True)
        self.log = self._log_scroll.documentView()
        self.log.setEditable_(False)
        self.log.setSelectable_(True)
        self._log_font = AppKit.NSFont.fontWithName_size_("Menlo", 12) or AppKit.NSFont.userFixedPitchFontOfSize_(12)
        self.log.setFont_(self._log_font)

        root = AppKit.NSStackView.stackViewWithViews_(
            [form, controls, self.progress_bar, status_row, self.log_toggle, self._log_scroll]
        )
        root.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationVertical)
        root.setAlignment_(AppKit.NSLayoutAttributeLeading)
        root.setDistribution_(AppKit.NSStackViewDistributionFill)
        root.setSpacing_(8)
        root.setEdgeInsets_(Foundation.NSEdgeInsets(MARGIN, MARGIN, MARGIN, MARGIN))
        root.setCustomSpacing_afterView_(16, form)
        window.setContentView_(root)
        for view in (form, self.progress_bar, status_row, self._log_scroll):
            view.widthAnchor().constraintEqualToAnchor_constant_(root.widthAnchor(), -2 * MARGIN).setActive_(True)
        self._root = root

        window.setMinSize_(AppKit.NSMakeSize(MIN_WIDTH, 0))
        self._set_log_visible(False)
        window.center()
        self._set_idle()

    # --- form state -------------------------------------------------------

    def _set_form_enabled(self, enabled: bool) -> None:
        for widget in (
            self.input_entry,
            self.file_button,
            self.folder_button,
            self.output_dir_entry,
            self.output_dir_button,
            self.output_entry,
            self.start_entry,
            self.end_entry,
            self.shared_mic_entry,
            self.separate_check,
            self.speaker_entry,
            self.elapsed_check,
            self.inaudible_check,
            self.bleed_check,
            self.save_log_check,
            self.start_button,
            *self.model_radios,
        ):
            widget.setEnabled_(enabled)
        if enabled:
            self._apply_input_rules()

    def _set_model(self, repo: str) -> None:
        self._model = repo

    @staticmethod
    def _checked(box) -> bool:
        return box.state() == _ON

    def _input_path(self) -> Path | None:
        text = self.input_entry.stringValue().strip()
        if not text:
            return None
        path = Path(text).expanduser()
        # Relative paths mean the home folder; _build_argv passes this
        # resolved path, so transcribe.py never depends on the cwd.
        return path if path.is_absolute() else Path.home() / path

    def _refresh_detected(self) -> None:
        path = self._input_path()
        self._input_kind = None
        if path is None:
            text = ""
        else:
            try:
                detected = detect_input(path)
            except InputError as e:
                text = str(e)
            else:
                self._input_kind = detected.kind
                text = {
                    InputKind.CRAIG_ZIP: "Craig export",
                    InputKind.CRAIG_FOLDER: "Craig export",
                    InputKind.SINGLE_FILE: "Single audio file",
                    InputKind.LOOSE_FOLDER: f"Folder: {len(detected.audio_files)} audio files",
                }[detected.kind]
        self.detected_label.setStringValue_(f"Detected: {text}" if text else "")
        if self._log_scroll.isHidden():
            # A wrapped input error changes the form's height.
            self._fit_collapsed()
        self._apply_input_rules()

    def _separate(self) -> bool:
        return self._input_kind is InputKind.LOOSE_FOLDER and self._checked(self.separate_check)

    def _apply_input_rules(self) -> None:
        """Enable only the fields that apply to the detected input."""
        if self._proc is not None:
            return
        if self._input_kind is not InputKind.LOOSE_FOLDER:
            self.separate_check.setState_(_OFF)
        separate = self._separate()
        self.separate_check.setEnabled_(self._input_kind is InputKind.LOOSE_FOLDER)
        self.speaker_entry.setEnabled_(self._input_kind is InputKind.SINGLE_FILE or separate)
        for widget in (self.output_entry, self.start_entry, self.end_entry, self.shared_mic_entry):
            widget.setEnabled_(not separate)

    def _set_idle(self) -> None:
        self._set_form_enabled(True)
        self.cancel_button.setEnabled_(False)
        self.open_transcript_button.setEnabled_(False)
        self.open_folder_button.setEnabled_(False)

    def _set_running(self) -> None:
        self._set_form_enabled(False)
        self.cancel_button.setEnabled_(True)
        self.open_transcript_button.setEnabled_(False)
        self.open_folder_button.setEnabled_(False)
        self.progress_bar.setDoubleValue_(0)
        self.status_label.setStringValue_("Starting…")

    def _set_done(self, status: str = "Done") -> None:
        self._set_form_enabled(True)
        self.cancel_button.setEnabled_(False)
        # A batch has no single transcript to open.
        self.open_transcript_button.setEnabled_(self._batch_output_dir is None)
        self.open_folder_button.setEnabled_(True)
        self.progress_bar.setDoubleValue_(100)
        self.status_label.setStringValue_(status)

    # --- pickers ----------------------------------------------------------

    def _pick_file(self) -> None:
        path = cocoa_ui.choose_path(False, str(self._input_dir), INPUT_FILETYPES)
        if path:
            self._set_input(path)

    def _pick_folder(self) -> None:
        path = cocoa_ui.choose_path(True, str(self._input_dir))
        if path:
            self._set_input(path)

    def _pick_output_dir(self) -> None:
        current = self.output_dir_entry.stringValue().strip()
        path = cocoa_ui.choose_path(True, current or str(Path.home()))
        if path:
            self.output_dir_entry.setStringValue_(path)

    def _set_input(self, path: str) -> None:
        self.input_entry.setStringValue_(path)
        self._refresh_detected()

    # --- run --------------------------------------------------------------

    def _build_argv(self) -> list[str]:
        argv = [sys.executable, "-u", str(CODE_DIR / "transcribe.py")]
        input_path = self._input_path()
        if input_path is not None:
            argv.append(str(input_path))
        argv.append("--progress-events")
        output_dir = self.output_dir_entry.stringValue().strip()
        if output_dir:
            argv += ["--output-dir", output_dir]
        separate = self._separate()
        if separate:
            argv.append("--separate")
        speaker = self.speaker_entry.stringValue().strip()
        if speaker and (separate or self._input_kind is InputKind.SINGLE_FILE):
            argv += ["--speaker", speaker]
        if not separate:
            # --separate rejects these; their entries are disabled meanwhile.
            output_name = self.output_entry.stringValue().strip()
            if output_name:
                argv += ["--output-name", output_name]
            # Validation (format, start < end, within the recording) is left to
            # transcribe.py, whose errors come back as an `error` event.
            window_start = self.start_entry.stringValue().strip()
            if window_start:
                argv += ["--start", window_start]
            window_end = self.end_entry.stringValue().strip()
            if window_end:
                argv += ["--end", window_end]
            for shared_mic in self.shared_mic_entry.stringValue().split(";"):
                if shared_mic.strip():
                    argv += ["--shared-mic", shared_mic.strip()]
        argv += ["--model", self._model]
        argv.append("--elapsed-timestamps" if self._checked(self.elapsed_check) else "--clock-timestamps")
        argv.append("--show-inaudible" if self._checked(self.inaudible_check) else "--hide-inaudible")
        argv.append("--show-bleed-markers" if self._checked(self.bleed_check) else "--hide-bleed-markers")
        return argv

    def _start(self) -> None:
        if self._proc is not None:
            return
        self._log_clear()
        self._cancelling = False
        self._result = None
        self._output_path = None
        self._batch_output_dir = None
        output_dir = self.output_dir_entry.stringValue().strip()
        self._run_output_dir = Path(output_dir).expanduser() if output_dir else None
        self._windowed = not self._separate() and bool(
            self.start_entry.stringValue().strip() or self.end_entry.stringValue().strip()
        )
        argv = self._build_argv()
        self._set_running()
        # stdin=DEVNULL keeps transcribe.py's _CancelWatcher off (it only arms
        # on a TTY stdin); otherwise launching the app or gui.py from a
        # terminal would put that terminal into cbreak mode. Config paths
        # resolve against the Application Support folder and the input path
        # is absolute, so the cwd doesn't matter.
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_output, args=(self._proc,), daemon=True).start()
        self._run_start = time.monotonic()
        self._tick_elapsed()
        self._elapsed_timer.start()

    def _read_output(self, proc: subprocess.Popen) -> None:
        # Runs on a worker thread; everything reaches the UI via callAfter,
        # in order, so the exit always lands after the last line.
        assert proc.stdout is not None and proc.stderr is not None
        # stderr is drained on its own thread so a chatty tqdm can't fill the
        # pipe and stall the child. Text mode splits tqdm's `\r` redraws into
        # separate lines, which the bounded tail then ages out.
        stderr_tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        stderr_reader = threading.Thread(
            target=lambda: stderr_tail.extend(line.rstrip("\n") for line in proc.stderr),
            daemon=True,
        )
        stderr_reader.start()
        for line in proc.stdout:
            callAfter(self._on_line, line.rstrip("\n"))
        returncode = proc.wait()
        stderr_reader.join()
        callAfter(self._on_exit, returncode, list(stderr_tail))

    def _on_line(self, line: str) -> None:
        if line.startswith(PROGRESS_EVENT_PREFIX):
            self._on_event(json.loads(line[len(PROGRESS_EVENT_PREFIX):]))
        else:
            self._log_append(line)

    def _on_event(self, event: dict) -> None:
        phase = event.get("phase")
        if phase in PHASE_LABELS:
            pct = event["pct"]
            self.progress_bar.setDoubleValue_(pct)
            track = f" ({event['track']})" if event.get("track") else ""
            self.status_label.setStringValue_(f"{PHASE_LABELS[phase]} — {pct:.0f}%{track}")
        elif phase == "step":
            self.status_label.setStringValue_(f"{event['label']}…")
        elif phase == "done":
            self._result = event
            self._output_path = Path(event["output_path"]) if event.get("output_path") else None
            if len(event.get("output_paths", [])) > 1:
                self._batch_output_dir = self._run_output_dir or self._output_path.parent
        elif phase == "error":
            self._result = event
            # A user cancel also ends in an error event; that one isn't a failure.
            if not self._cancelling:
                self._log_append(event["message"], "error")

    def _on_exit(self, returncode: int, stderr_tail: list[str]) -> None:
        self._proc = None
        if self._cancelling:
            self._log_append("Cancelled")
            self.status_label.setStringValue_("Cancelled")
            self._set_idle()
        elif self._result is not None and self._result["phase"] == "done":
            failed = self._result.get("failed", [])
            for failure in failed:
                self._log_append(f"{failure['file']} failed: {failure['message']}", "error")
            if self._checked(self.save_log_check):
                self._save_log()
            total = len(self._result.get("output_paths", [])) + len(failed)
            self._set_done(f"Done — {len(failed)} of {total} failed" if failed else "Done")
        else:
            if self._result is None:
                self._log_append(f"transcribe.py exited with code {returncode}", "error")
            self._log_traceback(stderr_tail)
            self.status_label.setStringValue_("Failed")
            self._set_idle()

    def _log_traceback(self, stderr_tail: list[str]) -> None:
        # The error event only carries `Type: message`; the traceback itself
        # is on stderr, after any tqdm noise. sys.exit("error: ...") failures
        # have no traceback and are already explained by their error event.
        # The header can share a line with an unterminated bar redraw, so it's
        # matched mid-line rather than with startswith.
        header = "Traceback (most recent call last):"
        for i, line in enumerate(stderr_tail):
            start = line.find(header)
            if start != -1:
                lines = [line[start:], *stderr_tail[i + 1:]]
                self._log_append("\n".join(lines).rstrip(), "error")
                return

    def _tick_elapsed(self) -> None:
        elapsed = timedelta(seconds=round(time.monotonic() - self._run_start))
        self.elapsed_label.setStringValue_(f"Elapsed {elapsed}")
        if self._proc is None:
            self._elapsed_timer.stop()

    # --- cancel / close ---------------------------------------------------

    def _cancel(self) -> None:
        if self._proc is None:
            return
        detail = (
            "Windowed runs don't save progress."
            if self._windowed
            else "Progress so far is saved."
        )
        if not cocoa_ui.ask_yes_no("Cancel", f"Stop transcription? {detail}"):
            return
        # The run may have ended while the alert was up.
        if self._proc is not None:
            self._cancelling = True
            self.cancel_button.setEnabled_(False)
            self.status_label.setStringValue_("Cancelling…")
            # The reader thread waits for the exit and reports it via
            # callAfter, so the UI never blocks on the child shutting down.
            self._proc.terminate()

    def _should_close(self) -> bool:
        # In the menu-bar app the window just hides; a run keeps going and
        # "Transcribe audio…" reopens it.
        if not self._standalone:
            return True
        if self._proc is not None:
            if not cocoa_ui.ask_yes_no("Quit", "A transcription is running. Stop it and quit?"):
                return False
            self._terminate_child()
        # Nothing reopens a standalone window, so closing it quits.
        callAfter(AppKit.NSApp.terminate_, None)
        return True

    def _will_close(self) -> None:
        if not self._standalone:
            cocoa_ui.set_regular(False)

    def _terminate_child(self) -> None:
        if self._proc is not None:
            self._proc.terminate()

    # --- done actions -----------------------------------------------------

    def _open_transcript(self) -> None:
        if self._output_path is not None:
            subprocess.run(["open", str(self._output_path)])

    def _open_folder(self) -> None:
        if self._batch_output_dir is not None:
            subprocess.run(["open", str(self._batch_output_dir)])
        elif self._output_path is not None:
            subprocess.run(["open", "-R", str(self._output_path)])

    # --- log pane ---------------------------------------------------------

    def _set_frame_height(self, height: float) -> None:
        frame = self._window.frame()
        top = frame.origin.y + frame.size.height
        self._window.setFrame_display_(
            AppKit.NSMakeRect(frame.origin.x, top - height, frame.size.width, height), True
        )

    def _set_log_visible(self, visible: bool) -> None:
        # Collapsing shrinks the window to fit the form; expanding restores
        # the height it had before, so the log gets its space back.
        unlimited = AppKit.NSMakeSize(Foundation.CGFLOAT_MAX, Foundation.CGFLOAT_MAX)
        if visible:
            self._log_scroll.setHidden_(False)
            self.log_toggle.setTitle_("▾ Hide log")
            self._window.setMaxSize_(unlimited)
            self._window.setMinSize_(AppKit.NSMakeSize(MIN_WIDTH, MIN_EXPANDED_HEIGHT))
            self._set_frame_height(max(self._expanded_height, MIN_EXPANDED_HEIGHT))
        else:
            if self._window.isVisible():
                self._expanded_height = self._window.frame().size.height
            self._log_scroll.setHidden_(True)
            self.log_toggle.setTitle_("▸ Show log")
            self._window.setMinSize_(AppKit.NSMakeSize(MIN_WIDTH, 0))
            self._fit_collapsed()

    def _fit_collapsed(self) -> None:
        """Size the window to its content while the log is hidden."""
        self._window.setMaxSize_(AppKit.NSMakeSize(Foundation.CGFLOAT_MAX, Foundation.CGFLOAT_MAX))
        self._root.layoutSubtreeIfNeeded()
        content_height = self._root.fittingSize().height
        height = self._window.frameRectForContentRect_(
            AppKit.NSMakeRect(0, 0, WINDOW_WIDTH, content_height)
        ).size.height
        self._set_frame_height(height)
        # Nothing to give extra height to while the log is hidden.
        self._window.setMaxSize_(AppKit.NSMakeSize(Foundation.CGFLOAT_MAX, height))

    def _log_append(self, text: str, tag: str | None = None) -> None:
        if tag == "error" and self._log_scroll.isHidden():
            # Failures are only explained in the log, so don't leave it hidden.
            self._set_log_visible(True)
        color = AppKit.NSColor.systemRedColor() if tag == "error" else AppKit.NSColor.textColor()
        attributed = AppKit.NSAttributedString.alloc().initWithString_attributes_(
            text + "\n",
            {AppKit.NSFontAttributeName: self._log_font, AppKit.NSForegroundColorAttributeName: color},
        )
        storage = self.log.textStorage()
        storage.appendAttributedString_(attributed)
        self.log.scrollRangeToVisible_((storage.length(), 0))

    def _save_log(self) -> None:
        # Kept next to the transcript (and its .errors.md), as `<name>.log`;
        # a batch's goes in its output folder as `batch-<timestamp>.log`.
        if self._batch_output_dir is not None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_path = self._batch_output_dir / f"batch-{stamp}.log"
        else:
            assert self._output_path is not None
            log_path = self._output_path.with_suffix(".log")
        try:
            log_path.write_text(self.log.string() + "\n")
        except OSError as e:
            self._log_append(f"Couldn't save the log to {log_path}: {e}", "error")

    def _log_clear(self) -> None:
        self.log.setString_("")
