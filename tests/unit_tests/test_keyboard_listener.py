# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import threading
from collections import deque

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


def test_dedicated_a4tech_keyboard_is_preferred_by_default():
    assert KeyboardListener.PREFERRED_DEVICE_PATH.endswith(
        "usb-HS6209_A4tech_2.4G_Wireless_Device-event-kbd"
    )


def _listener_without_device() -> KeyboardListener:
    listener = object.__new__(KeyboardListener)
    listener.state_lock = threading.Lock()
    listener.latest_data = {"key": None}
    listener._press_events = deque()
    listener._held_ctrl_keys = set()
    return listener


def test_ctrl_1_is_emitted_as_one_shortcut():
    listener = _listener_without_device()
    listener._handle_key_event("Key.leftctrl", 1)
    listener._handle_key_event("1", 1)
    listener._handle_key_event("1", 0)
    listener._handle_key_event("Key.leftctrl", 0)

    assert listener.pop_pressed_keys() == ["ctrl+1"]


def test_plain_1_remains_plain_key():
    listener = _listener_without_device()
    listener._handle_key_event("1", 1)
    listener._handle_key_event("1", 0)

    assert listener.pop_pressed_keys() == ["1"]
