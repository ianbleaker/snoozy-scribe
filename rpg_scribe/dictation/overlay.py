"""Floating "Recording / Transcribing" pill near the bottom of the screen.

A borderless, non-activating panel that ignores the mouse, so it never takes
focus and the paste still lands in the app being dictated into. All methods
must be called on the main thread.
"""

from __future__ import annotations

from collections.abc import Callable

import AppKit
import rumps

WIDTH = 220
HEIGHT = 44
BOTTOM_MARGIN = 90
BAR_X = 120
BAR_WIDTH = 84
BAR_HEIGHT = 6
MESSAGE_SECONDS = 2.5
MAX_MESSAGE_WIDTH = 560
# RMS of normal speech sits around 0.02-0.15; this maps it onto the bar.
LEVEL_GAIN = 8.0


def _cg_color(r: float, g: float, b: float, a: float):
    return AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, a).CGColor()


class Overlay:
    def __init__(self, level: Callable[[], float]) -> None:
        self._level = level
        # rumps creates NSApplication only in run(); windows need it first.
        AppKit.NSApplication.sharedApplication()
        style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
        panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(0, 0, WIDTH, HEIGHT), style, AppKit.NSBackingStoreBuffered, False
        )
        panel.setLevel_(AppKit.NSFloatingWindowLevel)
        panel.setIgnoresMouseEvents_(True)
        panel.setHidesOnDeactivate_(False)
        panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        panel.setOpaque_(False)
        panel.setBackgroundColor_(AppKit.NSColor.clearColor())
        panel.setHasShadow_(True)

        content = panel.contentView()
        content.setWantsLayer_(True)
        content.layer().setBackgroundColor_(_cg_color(0.1, 0.1, 0.1, 0.88))
        content.layer().setCornerRadius_(HEIGHT / 2)

        self._label = AppKit.NSTextField.labelWithString_("")
        self._label.setTextColor_(AppKit.NSColor.whiteColor())
        self._label.setFont_(AppKit.NSFont.systemFontOfSize_weight_(14, AppKit.NSFontWeightMedium))
        self._label.setFrame_(AppKit.NSMakeRect(18, (HEIGHT - 20) / 2, BAR_X - 24, 20))
        content.addSubview_(self._label)

        self._track = AppKit.NSView.alloc().initWithFrame_(
            AppKit.NSMakeRect(BAR_X, (HEIGHT - BAR_HEIGHT) / 2, BAR_WIDTH, BAR_HEIGHT)
        )
        self._track.setWantsLayer_(True)
        self._track.layer().setBackgroundColor_(_cg_color(1, 1, 1, 0.18))
        self._track.layer().setCornerRadius_(BAR_HEIGHT / 2)
        content.addSubview_(self._track)
        self._bar = AppKit.NSView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, 0, BAR_HEIGHT))
        self._bar.setWantsLayer_(True)
        self._bar.layer().setBackgroundColor_(_cg_color(0.95, 0.3, 0.3, 1))
        self._bar.layer().setCornerRadius_(BAR_HEIGHT / 2)
        self._track.addSubview_(self._bar)

        self._panel = panel
        self._timer = rumps.Timer(lambda _t: self._update_level(), 0.05)
        self._message_timer = rumps.Timer(lambda _t: self.hide(), MESSAGE_SECONDS)

    def _place(self) -> None:
        screen = AppKit.NSScreen.mainScreen()
        if screen is None:
            return
        frame = screen.visibleFrame()
        x = frame.origin.x + (frame.size.width - self._panel.frame().size.width) / 2
        self._panel.setFrameOrigin_((x, frame.origin.y + BOTTOM_MARGIN))

    def _update_level(self) -> None:
        width = min(1.0, self._level() * LEVEL_GAIN) * BAR_WIDTH
        self._bar.setFrameSize_((width, BAR_HEIGHT))

    def _set_width(self, width: float, label_width: float) -> None:
        self._panel.setContentSize_((width, HEIGHT))
        self._label.setFrame_(AppKit.NSMakeRect(18, (HEIGHT - 20) / 2, label_width, 20))

    def _show_status(self, text: str, level_bar: bool) -> None:
        self._message_timer.stop()
        self._set_width(WIDTH, BAR_X - 24)
        self._label.setStringValue_(text)
        self._track.setHidden_(not level_bar)
        self._place()
        self._panel.orderFrontRegardless()

    def show_message(self, text: str) -> None:
        """Show `text` briefly, like a toast. Recording or transcribing replaces it."""
        if self._timer.is_alive():
            return  # don't cover the recording meter
        self._label.setStringValue_(text)
        text_width = min(self._label.intrinsicContentSize().width + 4, MAX_MESSAGE_WIDTH - 36)
        self._set_width(text_width + 36, text_width)
        self._track.setHidden_(True)
        self._place()
        self._panel.orderFrontRegardless()
        self._message_timer.stop()
        self._message_timer.start()

    def show_recording(self) -> None:
        self._show_status("● Recording", level_bar=True)
        self._place()
        self._panel.orderFrontRegardless()
        if not self._timer.is_alive():
            self._timer.start()

    def show_transcribing(self) -> None:
        self._timer.stop()
        self._show_status("Transcribing…", level_bar=False)

    def hide_status(self) -> None:
        """Hide the recording/transcribing pill, but let a toast finish."""
        if not self._message_timer.is_alive():
            self.hide()

    def hide(self) -> None:
        self._timer.stop()
        self._message_timer.stop()
        self._panel.orderOut_(None)
