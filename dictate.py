#!/usr/bin/env python3
"""snoozy-scribe menu-bar dictation app.

Hold the hotkey, speak, release: the clip is transcribed with mlx-whisper,
cleaned up by a small mlx-lm model, and pasted into the frontmost app. See
README.md ("Using it").

Threads: rumps owns the main thread. The hotkey tap runs on its own thread
and only enqueues events. A control thread opens and closes the mic. One
worker thread runs all MLX work (warm-up, Whisper, cleanup) so it's
serialized. UI changes from other threads go through `callAfter`.
"""

from __future__ import annotations

import argparse
import logging
import plistlib
import queue
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

import rumps
from PyObjCTools.AppHelper import callAfter, callLater

from snoozy_scribe import identity, paths
from snoozy_scribe.config import check_legacy_layout, load_config
from snoozy_scribe.dictation.context import context_line, frontmost_context
from snoozy_scribe.dictation.clipboard import copy_selection, paste_text, secure_input_active, set_clipboard_text
from snoozy_scribe.dictation.hotkeys import (
    DELETE_KEYCODE,
    Binding,
    HotkeyTap,
    binding_problem,
    display_binding,
    format_binding,
    parse_binding,
)
from snoozy_scribe.dictation.permissions import PermissionsWindow, all_granted, running_as_bundle
from snoozy_scribe.dictation.recorder import SAMPLE_RATE, Recorder
from snoozy_scribe.dictation.state import load_state, update_state
from snoozy_scribe.dictation.vocab_state import VocabState, get_selected_vocabularies, set_vocabulary_selected
from snoozy_scribe.vocabulary import (
    VocabularyParseError,
    add_term,
    apply_vocabulary,
    list_vocabularies,
    merge_vocabularies,
    vocabulary_path,
)

log = logging.getLogger("dictate")

DEFAULT_HOLD_KEY = "right_option"
DEFAULT_TOGGLE_HOTKEY = "ctrl+option+d"
# Config key -> (menu label, is the hold key, default). Menu-bound overrides
# live under "hotkeys" in data/dictation-state.json, keyed the same way.
HOTKEY_ROLES = {
    "hotkey": ("Hold to talk", True, DEFAULT_HOLD_KEY),
    "toggle_hotkey": ("Toggle recording", False, DEFAULT_TOGGLE_HOTKEY),
    "add_selection_hotkey": ("Add selected text", False, None),
}
CAPTURE_SECONDS = 10
LAUNCH_AGENT_PATH = Path.home() / "Library/LaunchAgents" / f"{identity.LAUNCH_AGENT_LABEL}.plist"
LEGACY_LAUNCH_AGENT_PATH = LAUNCH_AGENT_PATH.with_name(f"{identity.LEGACY_LAUNCH_AGENT_LABEL}.plist")
APP_BUNDLE_ID = identity.BUNDLE_ID
# Shown in messages, with ~ for the home folder.
LOG_PATH_DISPLAY = f"~/{identity.LOG_PATH.relative_to(Path.home())}"

TITLES = {"loading": "✎…", "ready": "✎", "recording": "●", "transcribing": "✎⋯"}
STATUS_LABELS = {
    "loading": "Loading models…",
    "ready": "Ready",
    "recording": "Recording…",
    "transcribing": "Transcribing…",
}


# Set by the app when the overlay is on: messages go to the overlay as a toast.
_toast = None
# Set by init_notifications() when the overlay is off and this is the .app.
_center = None
# Held strongly here: the notification center only keeps a weak reference.
_center_delegate = None


def init_notifications() -> None:
    """Send messages to Notification Center, asking permission on first run.

    rumps.notification (the deprecated NSUserNotificationCenter) is never shown
    for this app. Only snoozy-scribe.app uses UNUserNotificationCenter: a terminal
    run is Homebrew's Python.app, which would ask permission as "Python", so it
    keeps the rumps fallback.
    """
    global _center, _center_delegate
    import AppKit

    if AppKit.NSBundle.mainBundle().bundleIdentifier() != APP_BUNDLE_ID:
        log.info("not running as %s.app; Notification Center messages are off", identity.APP_NAME)
        return
    import UserNotifications as UN

    class _Delegate(AppKit.NSObject):
        # Without this, banners are dropped while the app is frontmost, which
        # it is right after an "Add term…" or "Fix last dictation…" dialog.
        def userNotificationCenter_willPresentNotification_withCompletionHandler_(self, _c, _n, handler):
            handler(UN.UNNotificationPresentationOptionBanner | UN.UNNotificationPresentationOptionList)

    def authorized(granted, error):
        # Called off the main thread; logging only.
        log.info("notification authorization: granted=%s error=%s", bool(granted), error)

    center = UN.UNUserNotificationCenter.currentNotificationCenter()
    _center_delegate = _Delegate.alloc().init()
    center.setDelegate_(_center_delegate)
    center.requestAuthorizationWithOptions_completionHandler_(
        UN.UNAuthorizationOptionAlert | UN.UNAuthorizationOptionSound, authorized
    )
    _center = center


def _post_notification(title: str, message: str) -> None:
    import uuid

    import UserNotifications as UN

    content = UN.UNMutableNotificationContent.alloc().init()
    content.setTitle_(title)
    content.setBody_(message)
    request = UN.UNNotificationRequest.requestWithIdentifier_content_trigger_(str(uuid.uuid4()), content, None)

    def done(error):
        if error is not None:
            log.warning("notification not shown: %s", error)

    _center.addNotificationRequest_withCompletionHandler_(request, done)


def notify(message: str, title: str = identity.APP_NAME) -> None:
    """Show a short message. Main thread only."""
    log.info("notify: %s", message)
    if _toast is not None:
        _toast(message)
        return
    try:
        if _center is not None:
            _post_notification(title, message)
        else:
            rumps.notification(title, "", message)
    except Exception:
        log.exception("notification failed")


def app_bundle_path() -> Path:
    """The running .app, or the repo's snoozy-scribe.app when run from a terminal."""
    import AppKit

    bundle = Path(AppKit.NSBundle.mainBundle().bundlePath())
    return bundle if bundle.suffix == ".app" else paths.REPO_DIR / f"{identity.APP_NAME}.app"


class DictateApp(rumps.App):
    def __init__(self, config: dict) -> None:
        super().__init__(identity.APP_NAME, title=TITLES["loading"], quit_button="Quit")
        self._config = config
        dictation = config.get("dictation") or {}
        self._recorder = Recorder()
        self._vocab = VocabState(notify=lambda m: callAfter(notify, m))
        self._ready = False
        self._app_context = bool(dictation.get("app_context", True))
        self._context: dict | None = None
        # Newest first: dicts with raw, clean, used_fallback. Memory only.
        self._history: deque[dict] = deque(maxlen=int(dictation.get("history_size", 20)))
        self._history_serial = 0
        # Control thread only: None, "hold" or "toggle".
        self._mode: str | None = None
        self._vocab_names: list[str] | None = None

        self._status_item = rumps.MenuItem(STATUS_LABELS["loading"])
        self._vocab_menu = rumps.MenuItem("Vocabulary")
        self._history_menu = rumps.MenuItem("History")
        self._login_item = rumps.MenuItem("Launch at login", callback=self._toggle_login)
        self._login_item.state = LAUNCH_AGENT_PATH.exists()
        self._hotkeys_menu = rumps.MenuItem("Hotkeys")
        self._hotkey_items: dict[str, rumps.MenuItem] = {}
        for role in HOTKEY_ROLES:
            # After the menu closes, so the key press doesn't go to the menu.
            item = rumps.MenuItem(role, callback=lambda _s, r=role: self._after_menu(lambda: self._capture_hotkey(r)))
            self._hotkey_items[role] = item
            self._hotkeys_menu.add(item)
        self._hotkeys_menu.add(None)
        self._hotkeys_menu.add(rumps.MenuItem("Reset to config defaults", callback=self._reset_hotkeys))
        # Bumped per capture, so a superseded capture's result is ignored.
        self._capture_serial = 0
        self.menu = [
            self._status_item,
            None,
            self._vocab_menu,
            self._history_menu,
            None,
            rumps.MenuItem("Fix last dictation…", callback=self._fix_last),
            rumps.MenuItem("Add term…", callback=lambda _s: self._add_term_dialog("")),
            # After the menu closes, so ⌘C reaches the app with the selection.
            rumps.MenuItem("Add selected text", callback=lambda _s: self._after_menu(self._add_selection)),
            None,
            rumps.MenuItem("Transcribe session…", callback=self._open_gui),
            rumps.MenuItem("Open settings folder", callback=self._open_settings),
            rumps.MenuItem("Edit cleanup prompt…", callback=self._open_cleanup_prompt),
            self._hotkeys_menu,
            rumps.MenuItem("Permissions…", callback=lambda _s: self._permissions.show()),
            self._login_item,
            None,
        ]
        self._refresh_vocab_menu()
        self._rebuild_history()
        self._overlay = None
        if dictation.get("overlay", True):
            from snoozy_scribe.dictation.overlay import Overlay

            self._overlay = Overlay(level=lambda: self._recorder.level)
            global _toast
            _toast = self._overlay.show_message
        else:
            init_notifications()
        # Picks up vocabulary files added or removed while running.
        self._vocab_timer = rumps.Timer(lambda _t: self._refresh_vocab_menu(), 5)
        self._vocab_timer.start()

        self._hotkeys = self._config_hotkeys()
        self._hotkeys.update(self._saved_hotkeys())
        self._refresh_hotkey_items()

        # Hotkey events -> control thread (mic); clips -> worker thread (MLX).
        self._control: queue.Queue = queue.Queue()
        self._clips: queue.Queue = queue.Queue()
        threading.Thread(target=self._control_loop, name="dictate-control", daemon=True).start()
        threading.Thread(target=self._worker_loop, name="dictate-worker", daemon=True).start()
        self._tap = HotkeyTap(
            self._hotkeys["hotkey"],
            on_hold_start=lambda: self._control.put(("start", None)),
            on_hold_end=lambda valid: self._control.put(("stop", valid)),
            presses=self._presses(self._hotkeys),
        )
        self._tap.start()
        self._permissions = PermissionsWindow(tap_active=lambda: self._tap.active, can_relaunch=running_as_bundle())
        # Terminal runs skip this: their grants belong to Python, not the app.
        if running_as_bundle() and not all_granted():
            # NSApp isn't running until app.run(), so show on the first tick.
            self._after_menu(self._permissions.show)

    # --- UI (main thread) --------------------------------------------------

    def _set_status(self, status: str) -> None:
        self.title = TITLES[status]
        self._status_item.title = STATUS_LABELS[status]
        if self._overlay is not None:
            if status == "recording":
                self._overlay.show_recording()
            elif status == "transcribing":
                self._overlay.show_transcribing()
            else:
                self._overlay.hide_status()

    def _refresh_vocab_menu(self) -> None:
        names = list_vocabularies()
        if names != self._vocab_names:
            if self._vocab_names is not None:
                self._vocab_menu.clear()
            self._vocab_names = names
            for name in names:
                self._vocab_menu.add(rumps.MenuItem(name, callback=self._toggle_vocabulary))
            if not names:
                # No callback, so it shows disabled.
                self._vocab_menu.add(rumps.MenuItem("(none)"))
        if not names:
            return
        selected = get_selected_vocabularies()
        for item in self._vocab_menu.values():
            item.state = item.title in selected

    def _toggle_vocabulary(self, sender) -> None:
        set_vocabulary_selected(sender.title, not sender.state)
        self._refresh_vocab_menu()

    def _add_history(self, raw: str, clean: str, used_fallback: bool) -> None:
        self._history.appendleft({"raw": raw, "clean": clean, "used_fallback": used_fallback})
        self._rebuild_history()

    def _rebuild_history(self) -> None:
        # Before the first add the item has no submenu to clear.
        if self._history_serial or self._history_menu.get("(empty)") is not None:
            self._history_menu.clear()
        if not self._history:
            self._history_menu.add(rumps.MenuItem("(empty)"))
            return
        for entry in self._history:
            label = entry["clean"] if len(entry["clean"]) <= 40 else entry["clean"][:40] + "…"
            item = rumps.MenuItem(label)
            item.add(rumps.MenuItem("Copy cleaned", callback=lambda _s, t=entry["clean"]: set_clipboard_text(t)))
            item.add(rumps.MenuItem("Copy raw", callback=lambda _s, t=entry["raw"]: set_clipboard_text(t)))
            if entry["used_fallback"]:
                item.add(rumps.MenuItem("(fallback used)"))
            # Keys must be unique; identical labels would otherwise collide.
            self._history_serial += 1
            self._history_menu[f"history-{self._history_serial}"] = item

    # --- hotkey binding (main thread) -----------------------------------------

    def _config_hotkeys(self) -> dict[str, Binding | None]:
        """Bindings from config.yaml (or the defaults). A bad one stops startup."""
        dictation = self._config.get("dictation") or {}
        hotkeys = {}
        for role, (_label, _hold, default) in HOTKEY_ROLES.items():
            spec = dictation.get(role, default)
            if not spec and default is not None:
                raise ValueError(f"dictation.{role} in config.yaml is empty")
            hotkeys[role] = parse_binding(spec) if spec else None
        problem = self._hotkey_conflict(hotkeys)
        if problem:
            raise ValueError(f"dictation hotkeys in config.yaml: {problem}")
        return hotkeys

    def _saved_hotkeys(self) -> dict[str, Binding | None]:
        """Menu-bound overrides. Bad ones are dropped with a message, not fatal."""
        saved = load_state().get("hotkeys")
        if not isinstance(saved, dict):
            return {}
        overrides = {}
        for role, spec in saved.items():
            if role not in HOTKEY_ROLES:
                continue
            label, hold, default = HOTKEY_ROLES[role]
            if spec is None and default is None:  # an optional hotkey, cleared
                overrides[role] = None
                continue
            try:
                binding = parse_binding(spec)
                problem = binding_problem(binding, hold)
            except (ValueError, TypeError, AttributeError) as e:
                problem = str(e)
            if problem:
                log.warning("ignoring saved %s %r: %s", role, spec, problem)
                callAfter(notify, f"Ignored the saved {label} hotkey ({spec}); using config.yaml.")
                continue
            overrides[role] = binding
        problem = self._hotkey_conflict({**self._hotkeys, **overrides})
        if problem:
            log.warning("ignoring saved hotkeys: %s", problem)
            callAfter(notify, f"Ignored saved hotkeys ({problem}); using config.yaml.")
            return {}
        return overrides

    @staticmethod
    def _hotkey_conflict(hotkeys: dict[str, Binding | None]) -> str | None:
        seen: dict[Binding, str] = {}
        for role, binding in hotkeys.items():
            if binding is None:
                continue
            if binding in seen:
                first, second = HOTKEY_ROLES[seen[binding]][0], HOTKEY_ROLES[role][0]
                return f"{display_binding(binding)} is bound to both {first} and {second}"
            seen[binding] = role
        return None

    def _presses(self, hotkeys: dict[str, Binding | None]) -> dict[Binding, Callable[[], None]]:
        handlers = {
            "toggle_hotkey": lambda: self._control.put(("toggle", None)),
            "add_selection_hotkey": lambda: callAfter(self._add_selection),
        }
        return {hotkeys[role]: handler for role, handler in handlers.items() if hotkeys[role] is not None}

    def _set_hotkeys(self, hotkeys: dict[str, Binding | None]) -> None:
        self._tap.set_bindings(hotkeys["hotkey"], self._presses(hotkeys))
        self._hotkeys = hotkeys
        self._refresh_hotkey_items()

    def _refresh_hotkey_items(self) -> None:
        for role, item in self._hotkey_items.items():
            item.title = f"{HOTKEY_ROLES[role][0]}: {display_binding(self._hotkeys[role])}…"

    def _capture_hotkey(self, role: str) -> None:
        label, hold, _default = HOTKEY_ROLES[role]
        if not self._tap.active:
            notify("Hotkeys need Accessibility first; see Permissions….")
            return
        if self._mode is not None or self.title not in (TITLES["ready"], TITLES["loading"]):
            notify("Wait until dictation is idle before changing a hotkey.")
            return
        self._capture_serial += 1
        serial = self._capture_serial
        if self._tap.capturing:
            self._tap.cancel_capture()
        prompt = f"Press a key or mouse button for {label}. Esc cancels"
        if role == "add_selection_hotkey":
            prompt += ", ⌫ clears"
        if self._overlay is not None:
            self._overlay.show_prompt(prompt + ".")
        else:
            notify(prompt + ".")
        # Not rumps.Timer: its first fire is immediate, not after the interval.
        callLater(CAPTURE_SECONDS, self._capture_timed_out, serial)
        self._tap.capture(hold, lambda b: callAfter(self._hotkey_captured, serial, role, b))

    def _hotkey_captured(self, serial: int, role: str, binding: Binding | None) -> None:
        if serial != self._capture_serial:
            return  # replaced by a newer capture
        self._capture_serial += 1  # disarm the timeout
        label, hold, _default = HOTKEY_ROLES[role]
        if binding is None:
            notify(f"{label} hotkey unchanged.")
            return
        if role == "add_selection_hotkey" and binding == Binding("key", DELETE_KEYCODE):
            binding = None
        else:
            problem = binding_problem(binding, hold)
            if problem:
                notify(f"Can't use {display_binding(binding)}: {problem}.")
                return
        hotkeys = {**self._hotkeys, role: binding}
        problem = self._hotkey_conflict(hotkeys)
        if problem:
            notify(f"Not changed: {problem}.")
            return
        self._set_hotkeys(hotkeys)
        saved = load_state().get("hotkeys")
        saved = saved if isinstance(saved, dict) else {}
        saved[role] = format_binding(binding) if binding else None
        update_state(hotkeys=saved)
        log.info("bound %s to %s", role, saved[role])
        notify(f"{label}: {display_binding(binding)}")

    def _capture_timed_out(self, serial: int) -> None:
        if serial == self._capture_serial and self._tap.capturing:
            self._tap.cancel_capture()

    def _reset_hotkeys(self, _sender) -> None:
        update_state(hotkeys=None)
        self._set_hotkeys(self._config_hotkeys())
        notify("Hotkeys reset to config.yaml.")

    # --- vocabulary capture (main thread) -------------------------------------

    def _after_menu(self, action) -> None:
        def fire(timer) -> None:
            timer.stop()
            action()

        self._one_shot = rumps.Timer(fire, 0.2)
        self._one_shot.start()

    def _ask(self, title: str, message: str, default: str = "") -> tuple[str, Path] | None:
        """Ask for text and a target file ("Base" or the active campaign)."""
        import AppKit

        campaign = get_active_campaign()
        window = rumps.Window(
            message=message, title=title, default_text=default, ok="Base", cancel="Cancel", dimensions=(320, 24)
        )
        if campaign:
            window.add_button(campaign)
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        response = window.run()
        if response.clicked == 0:
            return None
        if not response.text.strip():
            notify("Nothing entered, so no changes were made.")
            return None
        target = BASE_PATH if response.clicked == 1 else campaign_path(campaign)
        return response.text.strip(), target

    def _write_term(self, target: Path, term: str, variants: list[str]) -> bool:
        try:
            add_term(target, term, variants)
        except (ValueError, VocabularyParseError) as e:
            rumps.alert(title="Couldn't add term", message=str(e))
            return False
        notify(f"Added {term} to {target.name}")
        return True

    def _fix_last(self, _sender) -> None:
        if not self._history:
            notify("Nothing dictated yet.")
            return
        last = self._history[0]
        answer = self._ask(
            "Fix last dictation",
            f"Last dictation:\n{last['clean']}\n\nType the correction as  wrong -> right",
        )
        if answer is None:
            return
        text, target = answer
        if "->" not in text:
            rumps.alert(title="Couldn't add term", message="Expected  wrong -> right")
            return
        wrong, right = (part.strip() for part in text.split("->", 1))
        if self._write_term(target, right, [wrong]):
            self._clips.put(("fix", last["raw"], None))

    def _add_term_dialog(self, default: str) -> None:
        answer = self._ask("Add term", "Term, or  Term | misheard1, misheard2", default)
        if answer is None:
            return
        text, target = answer
        term, _, variants = text.partition("|")
        self._write_term(target, term.strip(), [v.strip() for v in variants.split(",") if v.strip()])

    def _add_selection(self) -> None:
        selection = copy_selection()
        if not selection or not selection.strip():
            notify("No text selected.")
            return
        self._add_term_dialog(" ".join(selection.split()))

    def _open_gui(self, _sender) -> None:
        subprocess.Popen([sys.executable, str(paths.REPO_DIR / "gui.py")], cwd=paths.REPO_DIR)

    def _open_settings(self, _sender) -> None:
        subprocess.run(["open", str(paths.SETTINGS_DIR)], check=False)

    def _open_cleanup_prompt(self, _sender) -> None:
        subprocess.run(["open", str(paths.PROMPTS_DIR / "dictation-cleanup.md")], check=False)

    def _toggle_login(self, sender) -> None:
        if LAUNCH_AGENT_PATH.exists():
            LAUNCH_AGENT_PATH.unlink()
            sender.state = False
            return
        write_launch_agent()
        sender.state = True

    # --- control thread: mic on/off ----------------------------------------

    def _control_loop(self) -> None:
        while True:
            kind, valid = self._control.get()
            try:
                if kind == "start":
                    # Pressing the hold key during a toggle recording does nothing.
                    if self._ready and self._mode is None:
                        self._start_recording()
                        self._mode = "hold"
                        callAfter(self._set_status, "recording")
                elif kind == "stop":
                    if self._mode == "hold":
                        self._mode = None
                        self._finish_recording(valid)
                    elif self._mode is None and valid and not self._ready:
                        callAfter(notify, "Models are still loading. Try again in a moment.")
                elif kind == "toggle":
                    if not self._ready:
                        callAfter(notify, "Models are still loading. Try again in a moment.")
                    elif self._mode == "toggle":
                        self._mode = None
                        self._finish_recording(True)
                    else:
                        # The chord can arrive mid-hold when it uses the hold key
                        # (ctrl + right Option + d). Restart as a toggle recording.
                        if self._mode == "hold":
                            self._recorder.stop()
                        self._start_recording()
                        self._mode = "toggle"
                        callAfter(self._set_status, "recording")
            except Exception:
                log.exception("recording %s failed", kind)
                self._mode = None
                callAfter(notify, f"Recording failed; see {LOG_PATH_DISPLAY}.")
                callAfter(self._set_status, "ready")

    def _start_recording(self) -> None:
        self._recorder.start()
        self._context = frontmost_context() if self._app_context else None

    def _finish_recording(self, valid: bool) -> None:
        audio = self._recorder.stop()
        if not valid:
            log.info("hold cancelled; discarded %.1fs", len(audio) / SAMPLE_RATE)
            callAfter(self._set_status, "ready")
            return
        callAfter(self._set_status, "transcribing")
        self._clips.put(("clip", audio, self._context))

    # --- worker thread: all MLX work ---------------------------------------

    def _worker_loop(self) -> None:
        from snoozy_scribe.dictation import pipeline

        try:
            self._settings = pipeline.load_settings(self._config)
            pipeline.warm_up(self._settings)
            self._cleaner = load_cleaner(self._config)
            if self._cleaner is not None:
                self._cleaner.warm_up()
        except Exception:
            log.exception("model loading failed")
            callAfter(notify, f"Couldn't load the models; see {LOG_PATH_DISPLAY}.")
            return
        log.info("models ready")
        self._ready = True
        callAfter(self._set_status, "ready")

        while True:
            kind, payload, context = self._clips.get()
            try:
                if kind == "fix":
                    self._refix(payload)
                else:
                    self._process(payload, context)
            except Exception:
                log.exception("dictation failed")
                callAfter(notify, f"Dictation failed; see {LOG_PATH_DISPLAY}.")
            finally:
                if not self._recorder.recording:
                    callAfter(self._set_status, "ready")

    def _refix(self, raw: str) -> None:
        """Re-apply the (just updated) vocabulary and cleanup to an earlier raw text."""
        merged = merge_vocabularies(self._vocab.current_vocabs())
        raw = apply_vocabulary(raw, merged)
        clean = raw
        if self._cleaner is not None and raw:
            clean, _ = self._cleaner.clean(raw, list(merged))
        set_clipboard_text(clean)
        callAfter(notify, "Corrected text copied")

    def _process(self, audio, context: dict | None) -> None:
        from snoozy_scribe.dictation import pipeline

        vocabs = self._vocab.current_vocabs()
        raw = pipeline.transcribe_clip(audio, vocabs, self._settings)
        clean, used_fallback = raw, False
        if self._cleaner is not None and raw:
            clean, used_fallback = self._cleaner.clean(
                raw, list(merge_vocabularies(vocabs)), context_line(context)
            )
        log.info("dictated %.1fs into %s, fallback=%s", len(audio) / SAMPLE_RATE, context, used_fallback)
        if not clean:
            return
        callAfter(self._add_history, raw, clean, used_fallback)
        if secure_input_active():
            set_clipboard_text(clean)
            callAfter(notify, "Couldn't paste into a secure field. The text is on the clipboard.")
            return
        paste_text(clean)


def run_once(wav_path: str) -> None:
    """Run the dictation pipeline on a file with no UI, for testing."""
    import numpy as np
    from mlx_whisper.audio import load_audio

    from snoozy_scribe.dictation import pipeline

    config = load_config(paths.DEFAULT_CONFIG_PATH)
    settings = pipeline.load_settings(config)
    vocabs = VocabState(notify=print).current_vocabs()
    audio = np.array(load_audio(wav_path), dtype=np.float32)
    raw = pipeline.transcribe_clip(audio, vocabs, settings)
    print(f"RAW: {raw}")
    cleaner = load_cleaner(config)
    if cleaner is not None:
        clean, used_fallback = cleaner.clean(raw, list(merge_vocabularies(vocabs)))
        print(f"CLEAN: {clean}" + ("  (fallback used)" if used_fallback else ""))


def load_cleaner(config: dict):
    """The cleanup model, or None when `dictation.cleanup` is false."""
    from snoozy_scribe.dictation.cleanup import DEFAULT_CLEANUP_MODEL, Cleaner
    from snoozy_scribe.models import resolve_model_path

    dictation = config.get("dictation") or {}
    if not dictation.get("cleanup", True):
        return None
    model_dir = paths.resolve(config.get("model_dir", "./data/models"))
    return Cleaner(resolve_model_path(dictation.get("cleanup_model", DEFAULT_CLEANUP_MODEL), model_dir))


def write_launch_agent() -> None:
    # Launched through LaunchServices like a Finder double-click, so the
    # permissions granted to the bundle still apply.
    LAUNCH_AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LAUNCH_AGENT_PATH.open("wb") as f:
        plistlib.dump(
            {
                "Label": identity.LAUNCH_AGENT_LABEL,
                "ProgramArguments": ["/usr/bin/open", "-a", str(app_bundle_path())],
                "RunAtLoad": True,
            },
            f,
        )


def migrate_legacy_launch_agent() -> None:
    """Replace the old name's login item, which would keep opening the old app."""
    if not LEGACY_LAUNCH_AGENT_PATH.exists():
        return
    try:
        LEGACY_LAUNCH_AGENT_PATH.unlink()
        write_launch_agent()
    except OSError as e:
        log.warning("couldn't move Launch at login from %s: %s", LEGACY_LAUNCH_AGENT_PATH.name, e)
        return
    log.info("moved Launch at login from %s to %s", LEGACY_LAUNCH_AGENT_PATH.name, LAUNCH_AGENT_PATH.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{identity.APP_NAME} menu-bar dictation app.")
    parser.add_argument("--once", metavar="WAV", help="Transcribe WAV with no UI and print the result")
    parser.add_argument(
        "--print-context", action="store_true", help="Wait 3 s, then print the frontmost app and window"
    )
    args = parser.parse_args()
    if args.print_context:
        import time

        time.sleep(3)
        print(frontmost_context())
        return
    if args.once:
        run_once(args.once)
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # mlx_whisper opens a tqdm bar on every transcribe (even with verbose=None),
    # and tqdm's default lock includes a multiprocessing RLock. The app quits
    # through AppKit, which skips Python finalization, so that semaphore is
    # never unlinked and resource_tracker logs it as leaked. Nothing here uses
    # multiprocessing, so a thread lock is enough.
    import tqdm

    tqdm.tqdm.set_lock(threading.RLock())

    legacy = check_legacy_layout()
    try:
        if legacy:
            raise SystemExit("\n".join(legacy))
        config = load_config(paths.DEFAULT_CONFIG_PATH)
        migrate_legacy_launch_agent()
        app = DictateApp(config)
    except (SystemExit, ValueError) as e:
        # ValueError: a bad or clashing hotkey string in config.yaml.
        log.error("%s", e)
        rumps.alert(title=f"{identity.APP_NAME} can't start", message=str(e))
        sys.exit(1)
    app.run()


if __name__ == "__main__":
    main()
