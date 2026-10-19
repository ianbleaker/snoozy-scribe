"""Global hotkeys via one active CGEventTap on its own thread.

The tap sees modifier changes (the hold-to-talk key) and key-downs (hold
cancellation and the toggle chord). It's active rather than listen-only so
the toggle chord can be swallowed before it types a character. An active tap
needs Accessibility. Input Monitoring covers reading keys.

Handlers are called on the tap thread and must return quickly. macOS
disables a tap whose callback stalls, so hand real work off to another thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import ApplicationServices
import Quartz

log = logging.getLogger(__name__)

# Hold must last this long, with no other key pressed, to count as dictation.
MIN_HOLD_SECONDS = 0.2

# key name -> (virtual keycode, device-dependent flag bit that is set while
# that specific key is down). Device bits tell right from left modifiers.
HOLD_KEYS = {
    "right_option": (61, 0x40),
    "right_command": (54, 0x10),
    "right_control": (62, 0x2000),
    "fn": (63, Quartz.kCGEventFlagMaskSecondaryFn),
}

CHORD_MODIFIERS = {
    "ctrl": Quartz.kCGEventFlagMaskControl,
    "option": Quartz.kCGEventFlagMaskAlternate,
    "cmd": Quartz.kCGEventFlagMaskCommand,
    "shift": Quartz.kCGEventFlagMaskShift,
}
_ALL_CHORD_MODIFIERS = 0
for _mask in CHORD_MODIFIERS.values():
    _ALL_CHORD_MODIFIERS |= _mask

# US ANSI virtual keycodes.
CHORD_KEYS = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8,
    "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45,
    "m": 46, "space": 49,
}


def parse_chord(spec: str) -> tuple[int, int]:
    """Parse `mod+mod+key` (e.g. `ctrl+option+d`) into (modifier mask, keycode)."""
    parts = [p.strip().lower() for p in spec.split("+")]
    *mods, key = parts
    if not mods or key not in CHORD_KEYS or any(m not in CHORD_MODIFIERS for m in mods):
        raise ValueError(
            f"bad hotkey {spec!r}: expected mod+mod+key with mods "
            f"{', '.join(CHORD_MODIFIERS)} and a single letter or 'space'"
        )
    mask = 0
    for mod in mods:
        mask |= CHORD_MODIFIERS[mod]
    return mask, CHORD_KEYS[key]


class HotkeyTap:
    """Hold-to-talk key plus toggle chords, reported through callbacks.

    `on_hold_start()` fires as the hold key goes down. Recording should start
    right away so the first word isn't lost. `on_hold_end(valid)` fires when
    the hold finishes. `valid` is False when the hold was too short or another
    key was pressed during it (⌥← word navigation, Option-typed characters),
    in which case the audio should be discarded. `chords` maps a parsed
    chord to a handler. Matching key-downs are swallowed.
    """

    def __init__(
        self,
        hold_key: str,
        on_hold_start: Callable[[], None],
        on_hold_end: Callable[[bool], None],
        chords: dict[tuple[int, int], Callable[[], None]] | None = None,
    ) -> None:
        if hold_key not in HOLD_KEYS:
            raise ValueError(f"unknown hotkey {hold_key!r}; expected one of {', '.join(HOLD_KEYS)}")
        self._keycode, self._device_mask = HOLD_KEYS[hold_key]
        self._on_hold_start = on_hold_start
        self._on_hold_end = on_hold_end
        self._chords = chords or {}
        self._tap = None
        self._held_since: float | None = None
        self._interrupted = False

    def start(self) -> None:
        threading.Thread(target=self._run, name="hotkey-tap", daemon=True).start()

    def _run(self) -> None:
        mask = Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged) | Quartz.CGEventMaskBit(
            Quartz.kCGEventKeyDown
        )
        prompted = False
        while True:
            self._tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap,
                Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionDefault,
                mask,
                self._callback,
                None,
            )
            if self._tap is not None:
                break
            if not prompted:
                # Show the system prompts once. The tap is retried until both
                # permissions are granted, so no relaunch is needed.
                log.warning("event tap refused; grant Input Monitoring and Accessibility")
                Quartz.CGRequestListenEventAccess()
                ApplicationServices.AXIsProcessTrustedWithOptions(
                    {ApplicationServices.kAXTrustedCheckOptionPrompt: True}
                )
                prompted = True
            time.sleep(2)
        log.info("event tap active")
        source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self._tap, True)
        Quartz.CFRunLoopRun()

    def _callback(self, proxy, event_type, event, refcon):
        if event_type in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
            log.warning("event tap disabled (%s); re-enabling", event_type)
            Quartz.CGEventTapEnable(self._tap, True)
            return event
        flags = Quartz.CGEventGetFlags(event)
        keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        if event_type == Quartz.kCGEventFlagsChanged:
            if keycode == self._keycode:
                self._hold_changed(bool(flags & self._device_mask))
            return event
        # Key-down.
        if self._held_since is not None and not self._interrupted:
            self._interrupted = True
        handler = self._chords.get((flags & _ALL_CHORD_MODIFIERS, keycode))
        if handler is not None:
            # Swallow key repeats too, but only act on the first press.
            if not Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat):
                handler()
            return None
        return event

    def _hold_changed(self, down: bool) -> None:
        if down and self._held_since is None:
            self._held_since = time.monotonic()
            self._interrupted = False
            self._on_hold_start()
        elif not down and self._held_since is not None:
            held = time.monotonic() - self._held_since
            valid = held >= MIN_HOLD_SECONDS and not self._interrupted
            self._held_since = None
            self._on_hold_end(valid)
