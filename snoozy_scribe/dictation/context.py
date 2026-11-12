"""Which app and window the user is dictating into, and any text selected there."""

from __future__ import annotations

import AppKit
import ApplicationServices


def _focused_window_title(pid: int) -> str | None:
    app = ApplicationServices.AXUIElementCreateApplication(pid)
    err, window = ApplicationServices.AXUIElementCopyAttributeValue(
        app, ApplicationServices.kAXFocusedWindowAttribute, None
    )
    if err or window is None:
        return None
    err, title = ApplicationServices.AXUIElementCopyAttributeValue(
        window, ApplicationServices.kAXTitleAttribute, None
    )
    if err or not title:
        return None
    return str(title)


def frontmost_context() -> dict:
    """`{"app": name or None, "window": title or None}` for the frontmost app."""
    app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        return {"app": None, "window": None}
    try:
        window = _focused_window_title(app.processIdentifier())
    except Exception:
        window = None
    return {"app": app.localizedName(), "window": window}


# A hung frontmost app can otherwise stall an AX call for about 6 s.
AX_TIMEOUT = 0.25


def focused_selection() -> dict | None:
    """The frontmost app's selected text via Accessibility, never a synthesized ⌘C.

    Returns `{"pid": int, "text": str, "editable": bool}`, or None when there is
    no frontmost app, no focused element, an AX error, or no non-blank selection.
    `editable` says whether the selection can be replaced (False for a web page
    paragraph, a PDF, a message being viewed).
    """
    front = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    if front is None:
        return None
    pid = front.processIdentifier()
    AX = ApplicationServices
    try:
        app = AX.AXUIElementCreateApplication(pid)
        AX.AXUIElementSetMessagingTimeout(app, AX_TIMEOUT)
        err, focused = AX.AXUIElementCopyAttributeValue(app, AX.kAXFocusedUIElementAttribute, None)
        if err or focused is None:
            return None
        AX.AXUIElementSetMessagingTimeout(focused, AX_TIMEOUT)
        err, text = AX.AXUIElementCopyAttributeValue(focused, AX.kAXSelectedTextAttribute, None)
        if err or not text or not str(text).strip():
            return None
        err, settable = AX.AXUIElementIsAttributeSettable(focused, AX.kAXSelectedTextAttribute, None)
        editable = bool(settable) and not err
    except Exception:
        return None
    return {"pid": pid, "text": str(text), "editable": editable}


def context_line(context: dict | None) -> str | None:
    """The cleanup-prompt sentence for `context`, or None if the app is unknown."""
    if not context or not context.get("app"):
        return None
    where = context["app"]
    if context.get("window"):
        where += f" (window: {context['window']})"
    return f"The user is dictating into {where}. Match that setting's conventions; do not mention it."
