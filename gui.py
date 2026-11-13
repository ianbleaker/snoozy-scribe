"""Standalone session transcription window.

The same window the menu-bar app opens from "Transcribe audio…", for use
without the app: `python3 gui.py`. Closing it quits. The window itself lives
in snoozy_scribe/transcribe_window.py; see docs/session-transcription.md.
"""

from __future__ import annotations

import AppKit
from PyObjCTools import AppHelper

from snoozy_scribe import cocoa_ui
from snoozy_scribe.transcribe_window import TranscribeWindow

if __name__ == "__main__":
    AppKit.NSApplication.sharedApplication()
    cocoa_ui.set_regular(True)
    window = TranscribeWindow(standalone=True)
    window.show()
    AppHelper.runEventLoop()
