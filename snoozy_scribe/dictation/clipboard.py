"""Paste and copy through the general pasteboard via synthesized ⌘V/⌘C.

Both preserve the user's clipboard. They snapshot every pasteboard item and
every type's data, so an image or file on the clipboard survives, and they
restore it afterwards. Posting key events needs Accessibility.
"""

from __future__ import annotations

import time

import AppKit
import Quartz

_KEYCODE_C = 8
_KEYCODE_V = 9

# How long the target app gets to read the pasteboard before it's restored.
PASTE_RESTORE_DELAY = 0.5
COPY_WAIT = 0.15


def _snapshot(pasteboard) -> list[dict]:
    items = []
    for item in pasteboard.pasteboardItems() or []:
        data = {}
        for pb_type in item.types():
            value = item.dataForType_(pb_type)
            if value is not None:
                data[pb_type] = value
        items.append(data)
    return items


def _restore(pasteboard, snapshot: list[dict]) -> None:
    pasteboard.clearContents()
    if not snapshot:
        return
    items = []
    for data in snapshot:
        item = AppKit.NSPasteboardItem.alloc().init()
        for pb_type, value in data.items():
            item.setData_forType_(value, pb_type)
        items.append(item)
    pasteboard.writeObjects_(items)


def _post_command_key(keycode: int) -> None:
    source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateCombinedSessionState)
    for down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(source, keycode, down)
        Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def secure_input_active() -> bool:
    """True while a password field (or similar) has secure input on, which
    blocks synthesized keystrokes."""
    session = Quartz.CGSessionCopyCurrentDictionary() or {}
    return "kCGSSessionSecureInputPID" in session


def set_clipboard_text(text: str) -> None:
    pasteboard = AppKit.NSPasteboard.generalPasteboard()
    pasteboard.clearContents()
    pasteboard.setString_forType_(text, AppKit.NSPasteboardTypeString)


def paste_text(text: str) -> None:
    """Paste `text` into the frontmost app, then restore the clipboard."""
    pasteboard = AppKit.NSPasteboard.generalPasteboard()
    snapshot = _snapshot(pasteboard)
    pasteboard.clearContents()
    pasteboard.setString_forType_(text, AppKit.NSPasteboardTypeString)
    ours = pasteboard.changeCount()
    _post_command_key(_KEYCODE_V)
    time.sleep(PASTE_RESTORE_DELAY)
    # If something else wrote the clipboard meanwhile, leave it alone.
    if pasteboard.changeCount() == ours:
        _restore(pasteboard, snapshot)


def copy_selection() -> str | None:
    """Return the frontmost app's selected text via ⌘C, or None if nothing was copied."""
    pasteboard = AppKit.NSPasteboard.generalPasteboard()
    snapshot = _snapshot(pasteboard)
    before = pasteboard.changeCount()
    _post_command_key(_KEYCODE_C)
    time.sleep(COPY_WAIT)
    text = None
    if pasteboard.changeCount() != before:
        text = pasteboard.stringForType_(AppKit.NSPasteboardTypeString)
    _restore(pasteboard, snapshot)
    return text
