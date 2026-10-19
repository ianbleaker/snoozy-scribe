"""Which app and window the user is dictating into, for the cleanup prompt."""

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


def context_line(context: dict | None) -> str | None:
    """The cleanup-prompt sentence for `context`, or None if the app is unknown."""
    if not context or not context.get("app"):
        return None
    where = context["app"]
    if context.get("window"):
        where += f" (window: {context['window']})"
    return f"The user is dictating into {where}. Match that setting's conventions; do not mention it."
