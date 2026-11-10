"""Microphone and Accessibility: status, prompts, setup window.

No Input Monitoring: the hotkey tap is an active tap, which macOS gates on
Accessibility. Input Monitoring only covers listen-only taps.

The status/request functions are safe from any thread. The prompts only
appear while a permission is undetermined; after a denial the request calls
do nothing, so the window falls back to opening System Settings.
"""

from __future__ import annotations

import logging
import subprocess

import AppKit
import ApplicationServices
import objc
import rumps
from PyObjCTools.AppHelper import callAfter

from snoozy_scribe import identity

log = logging.getLogger(__name__)

# AVCaptureDevice straight from the system framework: the
# pyobjc-framework-AVFoundation wrapper (plus CoreMedia and CoreAudio) would
# only add the block signature below, which is registered by hand instead.
objc.loadBundle("AVFoundation", {}, bundle_path="/System/Library/Frameworks/AVFoundation.framework")
AVCaptureDevice = objc.lookUpClass("AVCaptureDevice")
objc.registerMetaDataForSelector(
    b"AVCaptureDevice",
    b"requestAccessForMediaType:completionHandler:",
    {"arguments": {3: {"callable": {"retval": {"type": b"v"}, "arguments": {0: {"type": b"^v"}, 1: {"type": b"Z"}}}}}},
)
MIC_MEDIA_TYPE = "soun"  # AVMediaTypeAudio

PERMISSIONS = ("microphone", "accessibility")

_SETTINGS_PANES = {
    "microphone": "Privacy_Microphone",
    "accessibility": "Privacy_Accessibility",
}


def status(name: str) -> str:
    """"granted", "undetermined" or "denied".

    Accessibility can't tell denied from undetermined, so it only ever
    reports "granted" or "undetermined".
    """
    if name == "microphone":
        value = AVCaptureDevice.authorizationStatusForMediaType_(MIC_MEDIA_TYPE)
        return {3: "granted", 0: "undetermined"}.get(value, "denied")
    if name == "accessibility":
        return "granted" if ApplicationServices.AXIsProcessTrusted() else "undetermined"
    raise ValueError(f"unknown permission {name!r}")


def all_granted() -> bool:
    return all(status(name) == "granted" for name in PERMISSIONS)


def request(name: str, on_done) -> None:
    """Show the system prompt for `name`; `on_done()` runs on the main thread after."""
    if name == "microphone":

        def handler(granted):
            log.info("microphone authorization: granted=%s", bool(granted))
            callAfter(on_done)

        AVCaptureDevice.requestAccessForMediaType_completionHandler_(MIC_MEDIA_TYPE, handler)
        return
    if name == "accessibility":
        ApplicationServices.AXIsProcessTrustedWithOptions({ApplicationServices.kAXTrustedCheckOptionPrompt: True})
    else:
        raise ValueError(f"unknown permission {name!r}")
    on_done()


def open_settings(name: str) -> None:
    url = f"x-apple.systempreferences:com.apple.preference.security?{_SETTINGS_PANES[name]}"
    AppKit.NSWorkspace.sharedWorkspace().openURL_(AppKit.NSURL.URLWithString_(url))


def running_as_bundle() -> bool:
    """True inside snoozy-scribe.app; False for a terminal run, whose grants belong to Python."""
    return AppKit.NSBundle.mainBundle().bundleIdentifier() == identity.BUNDLE_ID


# --- setup window (main thread only) ------------------------------------------

_ROWS = {
    "microphone": ("Microphone", "to record"),
    "accessibility": ("Accessibility", "for the hotkeys, pasting and window titles"),
}
WIDTH = 480
ROW_HEIGHT = 56
MARGIN = 20
FOOTER_HEIGHT = 56
HEIGHT = MARGIN + len(PERMISSIONS) * ROW_HEIGHT + FOOTER_HEIGHT
BUTTON_WIDTH = 130
# Consecutive polls with Accessibility granted but no tap before Relaunch shows.
RELAUNCH_AFTER_POLLS = 2


class PermissionsWindowTarget(AppKit.NSObject):
    """Button target and window delegate; forwards to the owning PermissionsWindow."""

    @objc.python_method
    def bind(self, owner):
        self.owner = owner
        return self

    def rowClicked_(self, sender):
        self.owner._row_clicked(PERMISSIONS[sender.tag()])

    def relaunchClicked_(self, _sender):
        self.owner.relaunch()

    def windowWillClose_(self, _notification):
        self.owner._timer.stop()


class PermissionsWindow:
    """One row per permission with a live ✓/✗ and a Grant / Open Settings button.

    `tap_active()` reports whether the hotkey tap exists. Relaunch is offered
    (only when `can_relaunch`) once Accessibility is granted but the tap
    still hasn't come up, which is the "needs a restart" case.
    """

    def __init__(self, tap_active, can_relaunch: bool) -> None:
        self._tap_active = tap_active
        self._can_relaunch = can_relaunch
        self._requested: set[str] = set()
        self._inactive_polls = 0
        # rumps creates NSApplication only in run(); windows need it first.
        AppKit.NSApplication.sharedApplication()
        self._target = PermissionsWindowTarget.alloc().init().bind(self)

        style = AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable
        window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(0, 0, WIDTH, HEIGHT), style, AppKit.NSBackingStoreBuffered, False
        )
        window.setTitle_(f"{identity.APP_NAME} permissions")
        window.setReleasedWhenClosed_(False)
        window.setDelegate_(self._target)
        window.center()
        content = window.contentView()

        self._marks = {}
        self._buttons = {}
        for index, name in enumerate(PERMISSIONS):
            title, purpose = _ROWS[name]
            y = HEIGHT - MARGIN - (index + 1) * ROW_HEIGHT
            mark = AppKit.NSTextField.labelWithString_("")
            mark.setFont_(AppKit.NSFont.systemFontOfSize_weight_(18, AppKit.NSFontWeightBold))
            mark.setFrame_(AppKit.NSMakeRect(MARGIN, y + 16, 24, 24))
            content.addSubview_(mark)
            label = AppKit.NSTextField.labelWithString_(title)
            label.setFont_(AppKit.NSFont.boldSystemFontOfSize_(13))
            label.setFrame_(AppKit.NSMakeRect(MARGIN + 30, y + 26, 260, 18))
            content.addSubview_(label)
            detail = AppKit.NSTextField.labelWithString_(purpose)
            detail.setFont_(AppKit.NSFont.systemFontOfSize_(11))
            detail.setTextColor_(AppKit.NSColor.secondaryLabelColor())
            detail.setFrame_(AppKit.NSMakeRect(MARGIN + 30, y + 8, 290, 16))
            content.addSubview_(detail)
            button = AppKit.NSButton.buttonWithTitle_target_action_("Grant", self._target, "rowClicked:")
            button.setTag_(index)
            button.setFrame_(AppKit.NSMakeRect(WIDTH - MARGIN - BUTTON_WIDTH, y + 12, BUTTON_WIDTH, 32))
            content.addSubview_(button)
            self._marks[name] = mark
            self._buttons[name] = button

        self._status = AppKit.NSTextField.labelWithString_("")
        self._status.setFrame_(AppKit.NSMakeRect(MARGIN, 22, 260, 18))
        content.addSubview_(self._status)
        self._relaunch = AppKit.NSButton.buttonWithTitle_target_action_("Relaunch", self._target, "relaunchClicked:")
        self._relaunch.setFrame_(AppKit.NSMakeRect(WIDTH - MARGIN - BUTTON_WIDTH, 14, BUTTON_WIDTH, 32))
        content.addSubview_(self._relaunch)

        self._window = window
        self._timer = rumps.Timer(lambda _t: self.refresh(), 1)

    def show(self) -> None:
        if not self._window.isVisible():
            self.refresh()
            if not self._timer.is_alive():
                self._timer.start()
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        self._window.makeKeyAndOrderFront_(None)

    def refresh(self) -> None:
        states = {name: status(name) for name in PERMISSIONS}
        for name, state in states.items():
            granted = state == "granted"
            mark = self._marks[name]
            mark.setStringValue_("✓" if granted else "✗")
            mark.setTextColor_(AppKit.NSColor.systemGreenColor() if granted else AppKit.NSColor.systemRedColor())
            button = self._buttons[name]
            button.setHidden_(granted)
            button.setTitle_("Grant" if self._grant_next(name, state) else "Open Settings")

        tap_active = self._tap_active()
        if states["accessibility"] == "granted" and not tap_active:
            self._inactive_polls += 1
        else:
            self._inactive_polls = 0
        show_relaunch = self._can_relaunch and self._inactive_polls >= RELAUNCH_AFTER_POLLS
        self._relaunch.setHidden_(not show_relaunch)
        if show_relaunch:
            text = "Relaunch to finish"
        elif all(state == "granted" for state in states.values()) and tap_active:
            text = "All set"
        else:
            text = "Grant the permissions above"
        self._status.setStringValue_(text)

    def _grant_next(self, name: str, state: str) -> bool:
        """True when the button should show the system prompt, False for Settings."""
        if name == "microphone":
            return state == "undetermined"
        return name not in self._requested

    def _row_clicked(self, name: str) -> None:
        if self._grant_next(name, status(name)):
            self._requested.add(name)
            # The mic prompt takes focus and the window, from a menu-bar app,
            # drops behind other apps when it closes, so bring it back. The
            # other prompt leads to System Settings, which should keep focus.
            request(name, self.show if name == "microphone" else self.refresh)
        else:
            open_settings(name)

    def relaunch(self) -> None:
        bundle_path = AppKit.NSBundle.mainBundle().bundlePath()
        log.info("relaunching %s", bundle_path)
        subprocess.Popen(["/bin/sh", "-c", 'sleep 1; open -n "$0"', bundle_path], start_new_session=True)
        rumps.quit_application()
