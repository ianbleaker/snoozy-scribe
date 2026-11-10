"""Global hotkeys via one active CGEventTap on its own thread.

A binding is a modifier key, a keyboard key or a mouse button (button 3 and
up), optionally with modifiers. Foot pedals and macro pads that send F13-F19
or extra mouse buttons work like any other key or button.

The tap sees modifier changes, key-downs/ups and extra mouse buttons. It's
active rather than listen-only so bound keys and buttons can be swallowed
before they reach the app. An active tap needs Accessibility, which also
covers reading keys, so no Input Monitoring.

Handlers are called on the tap thread and must return quickly. macOS
disables a tap whose callback stalls, so hand real work off to another thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import NamedTuple

import Quartz

log = logging.getLogger(__name__)

# Hold must last this long, with no other key pressed, to count as dictation.
MIN_HOLD_SECONDS = 0.2

ESCAPE_KEYCODE = 53
DELETE_KEYCODE = 51

# Modifier key name -> (virtual keycode, device-dependent flag bit that is set
# while that specific key is down). Device bits tell right from left.
MODIFIER_KEYS = {
    "right_option": (61, 0x40),
    "left_option": (58, 0x20),
    "right_command": (54, 0x10),
    "left_command": (55, 0x08),
    "right_control": (62, 0x2000),
    "left_control": (59, 0x01),
    "right_shift": (60, 0x04),
    "left_shift": (56, 0x02),
    "fn": (63, Quartz.kCGEventFlagMaskSecondaryFn),
}
_MODIFIER_BY_CODE = {code: (name, bit) for name, (code, bit) in MODIFIER_KEYS.items()}
_MODIFIER_DISPLAY = {"option": "⌥", "command": "⌘", "control": "⌃", "shift": "⇧"}

CHORD_MODIFIERS = {
    "ctrl": Quartz.kCGEventFlagMaskControl,
    "option": Quartz.kCGEventFlagMaskAlternate,
    "shift": Quartz.kCGEventFlagMaskShift,
    "cmd": Quartz.kCGEventFlagMaskCommand,
}
_CHORD_SYMBOLS = {"ctrl": "⌃", "option": "⌥", "shift": "⇧", "cmd": "⌘"}
_ALL_CHORD_MODIFIERS = 0
for _mask in CHORD_MODIFIERS.values():
    _ALL_CHORD_MODIFIERS |= _mask

# US ANSI virtual keycodes. Keys without a name here round-trip as keycode:N.
KEY_NAMES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8,
    "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26,
    "8": 28, "0": 29, "o": 31, "u": 32, "i": 34, "p": 35, "return": 36,
    "l": 37, "j": 38, "k": 40, "n": 45, "m": 46, "tab": 48, "space": 49,
    "delete": 51, "escape": 53,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98,
    "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111, "f13": 105,
    "f14": 107, "f15": 113, "f16": 106, "f17": 64, "f18": 79, "f19": 80,
    "f20": 90,
    "home": 115, "page_up": 116, "forward_delete": 117, "end": 119,
    "page_down": 121, "left": 123, "right": 124, "down": 125, "up": 126,
}
_KEY_BY_CODE = {code: name for name, code in KEY_NAMES.items()}
_FUNCTION_KEYS = {KEY_NAMES[f"f{n}"] for n in range(1, 21)}
# Keys that type or navigate (the main block 0-51, arrows, Home/End/Page,
# forward delete). They need a modifier, or binding them breaks typing.
_TYPING_KEYS = set(range(52)) | {115, 116, 117, 119, 121, 123, 124, 125, 126}
# Caps Lock toggles rather than holds, and Escape cancels capture.
_UNBINDABLE_KEYS = {57, ESCAPE_KEYCODE}

# kCGMouseEventButtonNumber: 0 left, 1 right, 2 middle, 3+ extra. Named
# mouse3 (middle) and up, so mouse4/mouse5 are the usual side buttons.
_FIRST_BINDABLE_BUTTON = 2


class Binding(NamedTuple):
    """kind is "modifier" (a lone modifier key), "key" or "mouse".

    code is the virtual keycode or the 0-based mouse button number. mods is
    the generic modifier mask held with a key or button (0 for a modifier).
    """

    kind: str
    code: int
    mods: int = 0


def parse_binding(spec: str) -> Binding:
    """Parse a config string: `right_option`, `f13`, `mouse4`, `ctrl+option+d`, `keycode:105`."""
    parts = [p.strip().lower() for p in str(spec).split("+")]
    *mod_names, key = parts
    if any(m not in CHORD_MODIFIERS for m in mod_names):
        raise ValueError(f"bad hotkey {spec!r}: modifiers are {', '.join(CHORD_MODIFIERS)}")
    mods = 0
    for name in mod_names:
        mods |= CHORD_MODIFIERS[name]
    if key in MODIFIER_KEYS:
        if mods:
            raise ValueError(f"bad hotkey {spec!r}: {key} can't take modifiers")
        return Binding("modifier", MODIFIER_KEYS[key][0])
    if key.startswith("mouse") and key[5:].isdigit():
        return Binding("mouse", int(key[5:]) - 1, mods)
    if key.startswith("keycode:") and key[8:].isdigit():
        code = int(key[8:])
        if code in _MODIFIER_BY_CODE:
            return Binding("modifier", code)
        return Binding("key", code, mods)
    if key in KEY_NAMES:
        return Binding("key", KEY_NAMES[key], mods)
    raise ValueError(
        f"bad hotkey {spec!r}: expected a modifier key ({', '.join(MODIFIER_KEYS)}), "
        "a key such as d, space or f13, mouseN, or keycode:N, optionally after mod+"
    )


def _mod_names(mods: int) -> list[str]:
    return [name for name, mask in CHORD_MODIFIERS.items() if mods & mask]


def format_binding(b: Binding) -> str:
    """The config string for `b`; parse_binding(format_binding(b)) == b."""
    if b.kind == "modifier":
        return _MODIFIER_BY_CODE[b.code][0]
    key = f"mouse{b.code + 1}" if b.kind == "mouse" else _KEY_BY_CODE.get(b.code, f"keycode:{b.code}")
    return "+".join([*_mod_names(b.mods), key])


def display_binding(b: Binding | None) -> str:
    """Short label for menus, e.g. Right ⌥, ⌃⌥D, Mouse 4, F13."""
    if b is None:
        return "none"
    if b.kind == "modifier":
        name = _MODIFIER_BY_CODE[b.code][0]
        if name == "fn":
            return "fn"
        side, mod = name.split("_")
        return f"{side.title()} {_MODIFIER_DISPLAY[mod]}"
    symbols = "".join(_CHORD_SYMBOLS[n] for n in _mod_names(b.mods))
    if b.kind == "mouse":
        return f"{symbols}{' ' if symbols else ''}Mouse {b.code + 1}"
    name = _KEY_BY_CODE.get(b.code)
    if name is None:
        label = f"Key {b.code}"
    elif len(name) == 1 or b.code in _FUNCTION_KEYS:
        label = name.upper()
    else:
        label = name.replace("_", " ").title()
    return symbols + label


def binding_problem(b: Binding, hold: bool) -> str | None:
    """Why `b` can't be used (as the hold key if `hold`, else a press hotkey), or None."""
    if b.kind == "modifier":
        if b.code not in _MODIFIER_BY_CODE:
            return "that modifier isn't supported"
        return None if hold else "a modifier alone can't toggle; add a key or mouse button"
    if b.kind == "mouse":
        if b.code < _FIRST_BINDABLE_BUTTON:
            return "left and right click can't be bound"
    elif b.code in _UNBINDABLE_KEYS:
        return "Caps Lock and Escape can't be bound"
    if hold and b.mods:
        return "the hold key is a single key or button, without modifiers"
    if not b.mods and b.kind == "key" and b.code in _TYPING_KEYS:
        return f"{display_binding(b)} on its own is needed for typing; add a modifier or use an F-key"
    return None


def _event_binding(kind: str, code: int, flags: int) -> Binding:
    return Binding(kind, code, flags & _ALL_CHORD_MODIFIERS)


class _Bindings(NamedTuple):
    hold: Binding
    hold_bit: int  # device flag bit when hold is a modifier, else 0
    presses: dict[Binding, Callable[[], None]]


class HotkeyTap:
    """Hold-to-talk binding plus press hotkeys, reported through callbacks.

    `on_hold_start()` fires as the hold key goes down. Recording should start
    right away so the first word isn't lost. `on_hold_end(valid)` fires when
    the hold finishes. `valid` is False when the hold was too short or another
    key was pressed during it (⌥← word navigation, Option-typed characters),
    in which case the audio should be discarded. `presses` maps a binding to
    a handler. Matching key-downs and button presses are swallowed.
    """

    def __init__(
        self,
        hold: Binding,
        on_hold_start: Callable[[], None],
        on_hold_end: Callable[[bool], None],
        presses: dict[Binding, Callable[[], None]] | None = None,
    ) -> None:
        self._on_hold_start = on_hold_start
        self._on_hold_end = on_hold_end
        self._tap = None
        self._held_since: float | None = None
        self._interrupted = False
        # Mouse buttons whose down was swallowed, so their up is too.
        self._swallowed_buttons: set[int] = set()
        # (hold, on_done) while capturing the next input as a binding.
        self._capture: tuple[bool, Callable[[Binding | None], None]] | None = None
        self._capture_modifier: int | None = None
        self._bindings = self._make_bindings(hold, presses or {})
        # True once the tap exists, i.e. the Accessibility grant took effect.
        self.active = False

    @staticmethod
    def _make_bindings(hold: Binding, presses: dict[Binding, Callable[[], None]]) -> _Bindings:
        for b, hold_role in [(hold, True), *((p, False) for p in presses)]:
            problem = binding_problem(b, hold=hold_role)
            if problem:
                raise ValueError(f"bad hotkey {format_binding(b)!r}: {problem}")
        bit = _MODIFIER_BY_CODE[hold.code][1] if hold.kind == "modifier" else 0
        return _Bindings(hold, bit, dict(presses))

    def set_bindings(self, hold: Binding, presses: dict[Binding, Callable[[], None]]) -> None:
        """Swap all bindings at once. A hold in progress ends as cancelled."""
        bindings = self._make_bindings(hold, presses)
        if self._held_since is not None:
            self._held_since = None
            self._on_hold_end(False)
        self._bindings = bindings

    def capture(self, hold: bool, on_done: Callable[[Binding | None], None]) -> None:
        """Report the next key or button press as a binding instead of acting on it.

        `on_done(None)` on Escape or cancel_capture(). Called on the tap thread.
        Nothing is validated here; see binding_problem().
        """
        self._capture_modifier = None
        self._capture = (hold, on_done)

    def cancel_capture(self) -> None:
        self._finish_capture(None)

    @property
    def capturing(self) -> bool:
        return self._capture is not None

    def _finish_capture(self, binding: Binding | None) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture[1](binding)

    def start(self) -> None:
        threading.Thread(target=self._run, name="hotkey-tap", daemon=True).start()

    def _run(self) -> None:
        mask = 0
        for event_type in (
            Quartz.kCGEventFlagsChanged,
            Quartz.kCGEventKeyDown,
            Quartz.kCGEventKeyUp,
            Quartz.kCGEventOtherMouseDown,
            Quartz.kCGEventOtherMouseUp,
        ):
            mask |= Quartz.CGEventMaskBit(event_type)
        warned = False
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
            if not warned:
                # Prompting is the permissions window's job; retry silently.
                log.warning("event tap refused; waiting for Accessibility")
                warned = True
            time.sleep(2)
        self.active = True
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
        if event_type in (Quartz.kCGEventOtherMouseDown, Quartz.kCGEventOtherMouseUp):
            button = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventButtonNumber)
            return self._mouse(event_type == Quartz.kCGEventOtherMouseDown, button, flags, event)
        keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        if self._capture is not None:
            return self._capture_key(event_type, keycode, flags, event)
        b = self._bindings
        if event_type == Quartz.kCGEventFlagsChanged:
            # Never swallowed: apps would be left with stuck modifier state.
            if b.hold.kind == "modifier" and keycode == b.hold.code:
                self._hold_changed(bool(flags & b.hold_bit))
            return event
        is_hold_key = b.hold.kind == "key" and keycode == b.hold.code
        if event_type == Quartz.kCGEventKeyUp:
            if is_hold_key:
                self._hold_changed(False)
                return None
            return event
        # Key-down.
        repeat = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat)
        handler = b.presses.get(_event_binding("key", keycode, flags))
        if handler is not None:
            self._interrupt_hold()
            # Swallow key repeats too, but only act on the first press.
            if not repeat:
                handler()
            return None
        if is_hold_key:
            if not repeat:
                self._hold_changed(True)
            return None
        self._interrupt_hold()
        return event

    def _mouse(self, down: bool, button: int, flags: int, event):
        if not down:
            if button in self._swallowed_buttons:
                self._swallowed_buttons.discard(button)
                b = self._bindings
                if b.hold.kind == "mouse" and button == b.hold.code:
                    self._hold_changed(False)
                return None
            return event
        if self._capture is not None:
            self._swallowed_buttons.add(button)
            self._finish_capture(_event_binding("mouse", button, flags))
            return None
        b = self._bindings
        handler = b.presses.get(_event_binding("mouse", button, flags))
        if handler is not None:
            self._swallowed_buttons.add(button)
            handler()
            return None
        if b.hold.kind == "mouse" and button == b.hold.code:
            self._swallowed_buttons.add(button)
            self._hold_changed(True)
            return None
        return event

    def _capture_key(self, event_type, keycode: int, flags: int, event):
        hold, _ = self._capture
        if event_type == Quartz.kCGEventFlagsChanged:
            # A lone modifier press-and-release binds that modifier (hold only).
            mod = _MODIFIER_BY_CODE.get(keycode)
            if hold and mod is not None:
                if flags & mod[1]:
                    self._capture_modifier = keycode
                elif self._capture_modifier == keycode:
                    self._finish_capture(Binding("modifier", keycode))
            return event
        if event_type == Quartz.kCGEventKeyUp:
            return event
        self._capture_modifier = None
        if not Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat):
            if keycode == ESCAPE_KEYCODE and not flags & _ALL_CHORD_MODIFIERS:
                self._finish_capture(None)
            else:
                self._finish_capture(_event_binding("key", keycode, flags))
        return None

    def _interrupt_hold(self) -> None:
        if self._held_since is not None:
            self._interrupted = True

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
