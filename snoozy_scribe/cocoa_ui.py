"""Small AppKit helpers shared by the app's windows.

Imports are limited to the stdlib and PyObjC (AppKit, Foundation, objc,
PyObjCTools): no rumps, yaml or `snoozy_scribe` modules. The packaged app's
setup window (#60) loads this before the full dependency set is installed.

AppKit holds targets, delegates and notification observers weakly. Every
owner keeps the objects created here in a Python list (`self._targets`), or
clicks and delegate calls silently do nothing.
"""

from __future__ import annotations

from collections.abc import Callable

import AppKit
import Foundation
import objc


class ActionTarget(AppKit.NSObject):
    """Wraps one Python callable; its action `perform:` calls `fn(sender)`."""

    def __new__(cls, fn: Callable):
        target = cls.alloc().init()
        target._fn = fn
        return target

    def perform_(self, sender):
        self._fn(sender)


class Delegate(AppKit.NSObject):
    """Window and text-field delegate forwarding to optional Python callables.

    `should_close() -> bool` (default True), `will_close()` and
    `text_changed(notification)`.
    """

    def __new__(cls, should_close=None, will_close=None, text_changed=None):
        delegate = cls.alloc().init()
        delegate._should_close = should_close
        delegate._will_close = will_close
        delegate._text_changed = text_changed
        return delegate

    def windowShouldClose_(self, _sender):
        return True if self._should_close is None else bool(self._should_close())

    def windowWillClose_(self, _notification):
        if self._will_close is not None:
            self._will_close()

    def controlTextDidChange_(self, notification):
        if self._text_changed is not None:
            self._text_changed(notification)


def _target(fn: Callable, owner_targets: list) -> ActionTarget:
    target = ActionTarget(fn)
    owner_targets.append(target)
    return target


# --- control factories ---------------------------------------------------------


def label(text: str, size: float = 13, bold: bool = False, secondary: bool = False):
    field = AppKit.NSTextField.labelWithString_(text)
    field.setFont_(AppKit.NSFont.boldSystemFontOfSize_(size) if bold else AppKit.NSFont.systemFontOfSize_(size))
    if secondary:
        field.setTextColor_(AppKit.NSColor.secondaryLabelColor())
    return field


def button(title: str, fn: Callable, owner_targets: list):
    return AppKit.NSButton.buttonWithTitle_target_action_(title, _target(fn, owner_targets), "perform:")


def text_field(placeholder: str = "", delegate: Delegate | None = None):
    field = AppKit.NSTextField.textFieldWithString_("")
    if placeholder:
        field.setPlaceholderString_(placeholder)
    if delegate is not None:
        field.setDelegate_(delegate)
    return field


def checkbox(title: str, fn: Callable, owner_targets: list):
    return AppKit.NSButton.checkboxWithTitle_target_action_(title, _target(fn, owner_targets), "perform:")


def radio(title: str, fn: Callable, owner_targets: list):
    # Radios sharing a superview and action are grouped by AppKit.
    return AppKit.NSButton.radioButtonWithTitle_target_action_(title, _target(fn, owner_targets), "perform:")


# --- modal panels ----------------------------------------------------------------


def ask_yes_no(title: str, message: str) -> bool:
    alert = AppKit.NSAlert.alloc().init()
    alert.setMessageText_(title)
    alert.setInformativeText_(message)
    alert.addButtonWithTitle_("Yes")
    alert.addButtonWithTitle_("No")
    return alert.runModal() == AppKit.NSAlertFirstButtonReturn


def choose_path(directories: bool, initial_dir: str | None = None, extensions: list[str] | None = None) -> str | None:
    panel = AppKit.NSOpenPanel.openPanel()
    panel.setCanChooseFiles_(not directories)
    panel.setCanChooseDirectories_(directories)
    panel.setAllowsMultipleSelection_(False)
    if directories:
        panel.setCanCreateDirectories_(True)
    if initial_dir:
        panel.setDirectoryURL_(Foundation.NSURL.fileURLWithPath_(initial_dir))
    if extensions and not directories:
        # Deprecated but working; allowedContentTypes would need the
        # UniformTypeIdentifiers wrapper.
        panel.setAllowedFileTypes_(extensions)
    if panel.runModal() != AppKit.NSModalResponseOK:
        return None
    url = panel.URL()
    return url.path() if url is not None else None


# --- timers and app state -----------------------------------------------------------


class Repeater:
    """Calls `fn()` every `interval` seconds on the main run loop.

    Registered in the common modes, so it keeps firing while a menu or modal
    alert is up.
    """

    def __init__(self, interval: float, fn: Callable[[], None]) -> None:
        self._interval = interval
        self._target = ActionTarget(lambda _timer: fn())
        self._timer = None

    def start(self) -> None:
        if self.is_running():
            return
        self._timer = Foundation.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            self._interval, self._target, "perform:", None, True
        )
        Foundation.NSRunLoop.currentRunLoop().addTimer_forMode_(self._timer, Foundation.NSRunLoopCommonModes)

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None

    def is_running(self) -> bool:
        return self._timer is not None and self._timer.isValid()


def _menu_item(title: str, action: str | None, key: str, modifiers: int | None = None):
    item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
    if modifiers is not None:
        item.setKeyEquivalentModifierMask_(modifiers)
    return item


def ensure_main_menu(app_name: str) -> None:
    """Install an app menu (Quit) and an Edit menu if there's no Edit menu.

    An accessory menu-bar app has no main menu, so text fields get no
    ⌘C/⌘V/⌘A/⌘Z without one. Idempotent: call it on every window show.
    """
    app = AppKit.NSApplication.sharedApplication()
    current = app.mainMenu()
    if current is not None and current.itemWithTitle_("Edit") is not None:
        return
    main = AppKit.NSMenu.alloc().initWithTitle_("")

    app_menu = AppKit.NSMenu.alloc().initWithTitle_(app_name)
    app_menu.addItem_(_menu_item(f"Quit {app_name}", "terminate:", "q"))
    app_item = AppKit.NSMenuItem.alloc().init()
    app_item.setSubmenu_(app_menu)
    main.addItem_(app_item)

    edit_menu = AppKit.NSMenu.alloc().initWithTitle_("Edit")
    shift_cmd = AppKit.NSEventModifierFlagShift | AppKit.NSEventModifierFlagCommand
    for title, action, key, modifiers in (
        ("Undo", "undo:", "z", None),
        ("Redo", "redo:", "z", shift_cmd),
        (None, None, "", None),
        ("Cut", "cut:", "x", None),
        ("Copy", "copy:", "c", None),
        ("Paste", "paste:", "v", None),
        ("Select All", "selectAll:", "a", None),
    ):
        if title is None:
            edit_menu.addItem_(AppKit.NSMenuItem.separatorItem())
        else:
            # Target nil: the action goes to the first responder.
            edit_menu.addItem_(_menu_item(title, action, key, modifiers))
    edit_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Edit", None, "")
    edit_item.setSubmenu_(edit_menu)
    main.addItem_(edit_item)

    app.setMainMenu_(main)


def set_regular(on: bool) -> None:
    """Regular (Dock icon, Cmd-Tab, app menu) or accessory (menu bar only)."""
    policy = AppKit.NSApplicationActivationPolicyRegular if on else AppKit.NSApplicationActivationPolicyAccessory
    AppKit.NSApplication.sharedApplication().setActivationPolicy_(policy)
