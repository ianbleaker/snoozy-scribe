"""Basic desktop GUI for rpg-scribe.

Runs transcribe.py as a subprocess with `--progress-events`, streaming its
stdout into a log pane and its `@@PROGRESS@@` JSON events into a progress
bar. Exposes only a subset of the CLI's flags; see README.md.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from collections import deque
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
from datetime import timedelta
from pathlib import Path

import customtkinter as ctk
import yaml

REPO_DIR = Path(__file__).resolve().parent
PROGRESS_EVENT_PREFIX = "@@PROGRESS@@ "
# Tail of the child's stderr kept for a failure dump; tqdm redraws land
# there too, so an uncapped buffer would grow for the whole run.
STDERR_TAIL_LINES = 500

# (label, --model repo); see docs/analysis/model-weight-analysis.md.
MODELS = [
    ("q4", "mlx-community/whisper-large-v3-turbo-q4"),
    ("turbo", "mlx-community/whisper-large-v3-turbo"),
    ("large-v3", "mlx-community/whisper-large-v3-mlx"),
]

WINDOW_WIDTH = 760
PHASE_LABELS = {"vad": "Analyzing speech", "transcribe": "Transcribing"}


def _load_config() -> dict:
    config_path = REPO_DIR / "config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return config if isinstance(config, dict) else {}


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("rpg-scribe")
        self.geometry(f"{WINDOW_WIDTH}x720")

        config = _load_config()
        self._input_dir = REPO_DIR / config.get("input_dir", "./input")
        model_repos = [repo for _, repo in MODELS]
        default_model = config.get("whisper_model")

        self._proc: subprocess.Popen | None = None
        self._events: queue.Queue = queue.Queue()
        self._cancelling = False
        self._result: dict | None = None
        self._output_path: Path | None = None
        self._run_start = 0.0
        # Windowed runs don't save partial output, so cancelling one loses it.
        self._windowed = False
        # Window height to restore when the log is expanded again.
        self._expanded_height = 720

        self.model_var = ctk.StringVar(
            value=default_model if default_model in model_repos else model_repos[0]
        )
        self.elapsed_var = ctk.BooleanVar(value=config.get("timestamp_mode", "elapsed") == "elapsed")
        self.inaudible_var = ctk.BooleanVar(value=bool(config.get("show_inaudible_placeholders", False)))
        self.bleed_var = ctk.BooleanVar(value=bool(config.get("show_bleed_markers", False)))
        self.save_log_var = ctk.BooleanVar(value=bool(config.get("save_gui_log", True)))

        self.grid_columnconfigure(0, weight=1)

        form = ctk.CTkFrame(self)
        form.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        form.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(form, text="Craig export").grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))
        self.input_entry = ctk.CTkEntry(form, placeholder_text="Blank = oldest .zip in input_dir")
        self.input_entry.grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=(12, 6))
        self.zip_button = ctk.CTkButton(form, text="Zip…", width=70, command=self._pick_zip)
        self.zip_button.grid(row=0, column=2, padx=(0, 6), pady=(12, 6))
        self.folder_button = ctk.CTkButton(form, text="Folder…", width=70, command=self._pick_folder)
        self.folder_button.grid(row=0, column=3, padx=(0, 12), pady=(12, 6))

        ctk.CTkLabel(form, text="Output name").grid(row=1, column=0, sticky="w", padx=12, pady=6)
        self.output_entry = ctk.CTkEntry(form, placeholder_text="Blank = slugified session title")
        self.output_entry.grid(row=1, column=1, columnspan=3, sticky="ew", padx=(0, 12), pady=6)

        ctk.CTkLabel(form, text="Time window").grid(row=2, column=0, sticky="w", padx=12, pady=6)
        window_row = ctk.CTkFrame(form, fg_color="transparent")
        window_row.grid(row=2, column=1, columnspan=3, sticky="w", pady=6)
        self.start_entry = ctk.CTkEntry(window_row, width=140, placeholder_text="Start (HH:MM:SS)")
        self.start_entry.pack(side="left")
        ctk.CTkLabel(window_row, text="to").pack(side="left", padx=8)
        self.end_entry = ctk.CTkEntry(window_row, width=140, placeholder_text="End (HH:MM:SS)")
        self.end_entry.pack(side="left")
        ctk.CTkLabel(window_row, text="Blank = whole session", text_color="gray").pack(
            side="left", padx=(12, 0)
        )

        ctk.CTkLabel(form, text="Model").grid(row=3, column=0, sticky="w", padx=12, pady=6)
        model_row = ctk.CTkFrame(form, fg_color="transparent")
        model_row.grid(row=3, column=1, columnspan=3, sticky="w", pady=6)
        self.model_radios = []
        for label, repo in MODELS:
            radio = ctk.CTkRadioButton(model_row, text=label, variable=self.model_var, value=repo)
            radio.pack(side="left", padx=(0, 16))
            self.model_radios.append(radio)

        self.elapsed_check = ctk.CTkCheckBox(
            form, text="Elapsed timestamps (uncheck for wall-clock time)", variable=self.elapsed_var
        )
        self.elapsed_check.grid(row=4, column=1, columnspan=3, sticky="w", pady=6)
        self.inaudible_check = ctk.CTkCheckBox(
            form, text="Show [inaudible] placeholders", variable=self.inaudible_var
        )
        self.inaudible_check.grid(row=5, column=1, columnspan=3, sticky="w", pady=6)
        self.bleed_check = ctk.CTkCheckBox(
            form, text="Mark possible mic bleed", variable=self.bleed_var
        )
        self.bleed_check.grid(row=6, column=1, columnspan=3, sticky="w", pady=6)
        self.save_log_check = ctk.CTkCheckBox(
            form, text="Save log next to the transcript", variable=self.save_log_var
        )
        self.save_log_check.grid(row=7, column=1, columnspan=3, sticky="w", pady=(6, 12))

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.grid(row=1, column=0, sticky="ew", padx=16, pady=8)
        controls.grid_columnconfigure(4, weight=1)
        self.start_button = ctk.CTkButton(controls, text="Start", width=90, command=self._start)
        self.start_button.grid(row=0, column=0, padx=(0, 8))
        self.cancel_button = ctk.CTkButton(controls, text="Cancel", width=90, command=self._cancel)
        self.cancel_button.grid(row=0, column=1, padx=(0, 8))
        self.open_transcript_button = ctk.CTkButton(
            controls, text="Open transcript", command=self._open_transcript
        )
        self.open_transcript_button.grid(row=0, column=2, padx=(0, 8))
        self.open_folder_button = ctk.CTkButton(controls, text="Open folder", command=self._open_folder)
        self.open_folder_button.grid(row=0, column=3)

        status = ctk.CTkFrame(self, fg_color="transparent")
        status.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 8))
        status.grid_columnconfigure(0, weight=1)
        self.progress_bar = ctk.CTkProgressBar(status)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        self.status_label = ctk.CTkLabel(status, text="Idle", anchor="w")
        self.status_label.grid(row=1, column=0, sticky="w")
        self.elapsed_label = ctk.CTkLabel(status, text="", anchor="e")
        self.elapsed_label.grid(row=1, column=0, sticky="e")

        self.log_toggle = ctk.CTkButton(
            self,
            text="",
            width=0,
            anchor="w",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover=False,
            command=lambda: self._set_log_visible(not self.log.winfo_manager()),
        )
        self.log_toggle.grid(row=3, column=0, sticky="w", padx=16)
        self.log = ctk.CTkTextbox(self, wrap="word", font=ctk.CTkFont(family="Menlo", size=12))
        self.log.grid(row=4, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.log.tag_config("error", foreground="#e5484d")
        self.log.configure(state="disabled")
        self._set_log_visible(False)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._set_idle()

    # --- form state -------------------------------------------------------

    def _set_form_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (
            self.input_entry,
            self.zip_button,
            self.folder_button,
            self.output_entry,
            self.start_entry,
            self.end_entry,
            self.elapsed_check,
            self.inaudible_check,
            self.bleed_check,
            self.save_log_check,
            self.start_button,
            *self.model_radios,
        ):
            widget.configure(state=state)

    def _set_idle(self) -> None:
        self._set_form_enabled(True)
        self.cancel_button.configure(state="disabled")
        self.open_transcript_button.configure(state="disabled")
        self.open_folder_button.configure(state="disabled")

    def _set_running(self) -> None:
        self._set_form_enabled(False)
        self.cancel_button.configure(state="normal")
        self.open_transcript_button.configure(state="disabled")
        self.open_folder_button.configure(state="disabled")
        self.progress_bar.set(0)
        self.status_label.configure(text="Starting…")

    def _set_done(self) -> None:
        self._set_form_enabled(True)
        self.cancel_button.configure(state="disabled")
        self.open_transcript_button.configure(state="normal")
        self.open_folder_button.configure(state="normal")
        self.progress_bar.set(1)
        self.status_label.configure(text="Done")

    # --- pickers ----------------------------------------------------------

    def _pick_zip(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=self._input_dir, filetypes=[("Craig export", "*.zip"), ("All files", "*")]
        )
        if path:
            self._set_input(path)

    def _pick_folder(self) -> None:
        path = filedialog.askdirectory(initialdir=self._input_dir, mustexist=True)
        if path:
            self._set_input(path)

    def _set_input(self, path: str) -> None:
        self.input_entry.delete(0, "end")
        self.input_entry.insert(0, path)

    # --- run --------------------------------------------------------------

    def _build_argv(self) -> list[str]:
        argv = [sys.executable, "-u", str(REPO_DIR / "transcribe.py")]
        input_path = self.input_entry.get().strip()
        if input_path:
            argv.append(input_path)
        argv.append("--progress-events")
        output_name = self.output_entry.get().strip()
        if output_name:
            argv += ["--output-name", output_name]
        # Validation (format, start < end, within the recording) is left to
        # transcribe.py, whose errors come back as an `error` event.
        window_start = self.start_entry.get().strip()
        if window_start:
            argv += ["--start", window_start]
        window_end = self.end_entry.get().strip()
        if window_end:
            argv += ["--end", window_end]
        argv += ["--model", self.model_var.get()]
        argv.append("--elapsed-timestamps" if self.elapsed_var.get() else "--clock-timestamps")
        argv.append("--show-inaudible" if self.inaudible_var.get() else "--hide-inaudible")
        argv.append("--show-bleed-markers" if self.bleed_var.get() else "--hide-bleed-markers")
        return argv

    def _start(self) -> None:
        self._log_clear()
        self._cancelling = False
        self._result = None
        self._output_path = None
        self._windowed = bool(self.start_entry.get().strip() or self.end_entry.get().strip())
        self._set_running()
        # stdin=DEVNULL keeps transcribe.py's _CancelWatcher off (it only arms
        # on a TTY stdin); otherwise launching gui.py from a terminal would put
        # that terminal into cbreak mode. cwd=REPO_DIR because config.yaml,
        # vocabulary.md and input_dir all resolve relative to it.
        self._proc = subprocess.Popen(
            self._build_argv(),
            cwd=REPO_DIR,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_output, args=(self._proc,), daemon=True).start()
        self._run_start = time.monotonic()
        self._tick_elapsed()
        self.after(100, self._poll)

    def _read_output(self, proc: subprocess.Popen) -> None:
        # Runs on a worker thread; everything reaches the UI via the queue.
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
            self._events.put(("line", line.rstrip("\n")))
        returncode = proc.wait()
        stderr_reader.join()
        self._events.put(("exit", (returncode, list(stderr_tail))))

    def _poll(self) -> None:
        while True:
            try:
                kind, value = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "exit":
                self._on_exit(*value)
                return
            if value.startswith(PROGRESS_EVENT_PREFIX):
                self._on_event(json.loads(value[len(PROGRESS_EVENT_PREFIX):]))
            else:
                self._log_append(value)
        self.after(100, self._poll)

    def _on_event(self, event: dict) -> None:
        phase = event.get("phase")
        if phase in PHASE_LABELS:
            pct = event["pct"]
            self.progress_bar.set(pct / 100)
            track = f" ({event['track']})" if event.get("track") else ""
            self.status_label.configure(text=f"{PHASE_LABELS[phase]} — {pct:.0f}%{track}")
        elif phase == "step":
            self.status_label.configure(text=f"{event['label']}…")
        elif phase == "done":
            self._result = event
            self._output_path = Path(event["output_path"])
        elif phase == "error":
            self._result = event
            # A user cancel also ends in an error event; that one isn't a failure.
            if not self._cancelling:
                self._log_append(event["message"], "error")

    def _on_exit(self, returncode: int, stderr_tail: list[str]) -> None:
        self._proc = None
        if self._cancelling:
            self._log_append("Cancelled")
            self.status_label.configure(text="Cancelled")
            self._set_idle()
        elif self._result is not None and self._result["phase"] == "done":
            if self.save_log_var.get():
                self._save_log()
            self._set_done()
        else:
            if self._result is None:
                self._log_append(f"transcribe.py exited with code {returncode}", "error")
            self._log_traceback(stderr_tail)
            self.status_label.configure(text="Failed")
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
        self.elapsed_label.configure(text=f"Elapsed {elapsed}")
        if self._proc is not None:
            self.after(1000, self._tick_elapsed)

    # --- cancel / close ---------------------------------------------------

    def _cancel(self) -> None:
        if self._proc is None:
            return
        detail = (
            "Windowed runs don't save progress."
            if self._windowed
            else "Progress on completed tracks is saved."
        )
        if not messagebox.askyesno("Cancel", f"Stop transcription? {detail}", parent=self):
            return
        if self._proc is not None:
            self._cancelling = True
            self.cancel_button.configure(state="disabled")
            self.status_label.configure(text="Cancelling…")
            # The reader thread waits for the exit and reports it via the
            # queue, so the UI never blocks on the child shutting down.
            self._proc.terminate()

    def _on_close(self) -> None:
        if self._proc is not None:
            if not messagebox.askyesno(
                "Quit", "A transcription is running. Stop it and quit?", parent=self
            ):
                return
            self._proc.terminate()
        self.destroy()

    # --- done actions -----------------------------------------------------

    def _open_transcript(self) -> None:
        if self._output_path is not None:
            subprocess.run(["open", str(self._output_path)])

    def _open_folder(self) -> None:
        if self._output_path is not None:
            subprocess.run(["open", "-R", str(self._output_path)])

    # --- log pane ---------------------------------------------------------

    def _set_log_visible(self, visible: bool) -> None:
        # Collapsing shrinks the window to fit the form; expanding restores
        # the height it had before, so the log gets its space back.
        width = self.winfo_width() if self.winfo_ismapped() else WINDOW_WIDTH
        if visible:
            self.log.grid()
            self.grid_rowconfigure(4, weight=1)
            self.log_toggle.configure(text="▾ Hide log")
            self.log_toggle.grid_configure(pady=0)
            self.minsize(560, 520)
            self.geometry(f"{width}x{max(self._expanded_height, 520)}")
        else:
            if self.winfo_ismapped():
                self._expanded_height = self.winfo_height()
            self.log.grid_remove()
            self.grid_rowconfigure(4, weight=0)
            self.log_toggle.configure(text="▸ Show log")
            self.log_toggle.grid_configure(pady=(0, 12))
            self.minsize(560, 0)
            self.update_idletasks()
            self.geometry(f"{width}x{self.winfo_reqheight()}")

    def _log_append(self, text: str, tag: str | None = None) -> None:
        if tag == "error" and not self.log.winfo_manager():
            # Failures are only explained in the log, so don't leave it hidden.
            self._set_log_visible(True)
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _save_log(self) -> None:
        # Kept next to the transcript (and its .errors.md), as `<name>.log`.
        assert self._output_path is not None
        log_path = self._output_path.with_suffix(".log")
        try:
            log_path.write_text(self.log.get("1.0", "end-1c") + "\n")
        except OSError as e:
            self._log_append(f"Couldn't save the log to {log_path}: {e}", "error")

    def _log_clear(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()
