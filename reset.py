#!/usr/bin/env python3
"""Send the DUST reset sequence that matched method 14."""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes


user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
ULONG_PTR = getattr(wintypes, "ULONG_PTR", ctypes.c_size_t)

SW_RESTORE = 9
SW_MINIMIZE = 6
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MAPVK_VK_TO_VSC = 0
VK_CONTROL = 0x11
VK_LCONTROL = 0xA2
VK_R = 0x52
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUTUNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", INPUTUNION),
    ]


user32.BringWindowToTop.argtypes = [wintypes.HWND]
user32.BringWindowToTop.restype = wintypes.BOOL
user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
user32.MapVirtualKeyW.restype = wintypes.UINT
user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.keybd_event.argtypes = [wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, ULONG_PTR]
user32.keybd_event.restype = None
user32.mouse_event.argtypes = [
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ULONG_PTR,
]
user32.mouse_event.restype = None
kernel32.GetCurrentThreadId.argtypes = []
kernel32.GetCurrentThreadId.restype = wintypes.DWORD
kernel32.GetConsoleWindow.argtypes = []
kernel32.GetConsoleWindow.restype = wintypes.HWND


def get_window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def find_dust_window() -> tuple[int, str] | None:
    matches: list[tuple[int, str]] = []

    @WNDENUMPROC
    def enum_proc(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        title = get_window_title(hwnd)
        if "DUST" in title.upper():
            matches.append((hwnd, title))
            return False
        return True

    if not user32.EnumWindows(enum_proc, 0):
        last_error = ctypes.get_last_error()
        if last_error and not matches:
            raise ctypes.WinError(last_error)
    return matches[0] if matches else None


def get_window_center(hwnd: int) -> tuple[int, int]:
    rect = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError(ctypes.get_last_error())
    return (rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2


def focus_basic(hwnd: int) -> None:
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    for _ in range(20):
        if user32.GetForegroundWindow() == hwnd:
            return
        time.sleep(0.05)


def force_focus_transition(target_hwnd: int) -> None:
    current_hwnd = user32.GetForegroundWindow()
    console_hwnd = kernel32.GetConsoleWindow()

    if current_hwnd == target_hwnd:
        if console_hwnd and console_hwnd != target_hwnd:
            user32.ShowWindow(console_hwnd, SW_RESTORE)
            user32.BringWindowToTop(console_hwnd)
            user32.SetForegroundWindow(console_hwnd)
            time.sleep(0.15)
        else:
            user32.ShowWindow(target_hwnd, SW_MINIMIZE)
            time.sleep(0.15)

    focus_basic(target_hwnd)
    time.sleep(0.15)


def mouse_click_center(hwnd: int) -> None:
    x, y = get_window_center(hwnd)
    if not user32.SetCursorPos(x, y):
        raise ctypes.WinError(ctypes.get_last_error())
    time.sleep(0.05)
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.03)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def keybd_event_vk_combo() -> None:
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    time.sleep(0.03)
    user32.keybd_event(VK_R, 0, 0, 0)
    time.sleep(0.03)
    user32.keybd_event(VK_R, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


def make_input_sc(scan: int, flags: int = 0) -> INPUT:
    return INPUT(
        type=INPUT_KEYBOARD,
        union=INPUTUNION(
            ki=KEYBDINPUT(
                wVk=0,
                wScan=scan,
                dwFlags=KEYEVENTF_SCANCODE | flags,
                time=0,
                dwExtraInfo=0,
            )
        ),
    )


def sendinput_scancode_extended_combo() -> None:
    ctrl_scan = user32.MapVirtualKeyW(VK_LCONTROL, MAPVK_VK_TO_VSC)
    r_scan = user32.MapVirtualKeyW(VK_R, MAPVK_VK_TO_VSC)
    if not ctrl_scan or not r_scan:
        raise RuntimeError("MapVirtualKeyW failed for Ctrl or R.")
    array = (INPUT * 4)(
        make_input_sc(ctrl_scan, KEYEVENTF_EXTENDEDKEY),
        make_input_sc(r_scan),
        make_input_sc(r_scan, KEYEVENTF_KEYUP),
        make_input_sc(ctrl_scan, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP),
    )
    sent = user32.SendInput(len(array), array, ctypes.sizeof(INPUT))
    if sent != len(array):
        raise ctypes.WinError(ctypes.get_last_error())


def method_4(hwnd: int) -> None:
    force_focus_transition(hwnd)
    sendinput_scancode_extended_combo()


def method_5(hwnd: int) -> None:
    force_focus_transition(hwnd)
    mouse_click_center(hwnd)
    time.sleep(0.1)
    keybd_event_vk_combo()


def reset_sequence(hwnd: int) -> None:
    method_4(hwnd)
    time.sleep(0.15)
    method_5(hwnd)
    time.sleep(0.25)
    method_4(hwnd)
    time.sleep(0.15)
    method_5(hwnd)
    time.sleep(0.15)
    method_5(hwnd)
    time.sleep(0.35)
    method_4(hwnd)
    time.sleep(0.15)
    method_5(hwnd)
    time.sleep(0.15)
    method_5(hwnd)


def main() -> int:
    match = find_dust_window()
    if match is None:
        print('No visible window with "DUST" in its title was found.', file=sys.stderr)
        return 1

    hwnd, title = match
    reset_sequence(hwnd)
    print(f'Sent reset sequence to "{title}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
