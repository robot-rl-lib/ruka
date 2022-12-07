import copy
import gym
import evdev
import time

from dataclasses import dataclass
from evdev import ecodes
from typing import Dict, List


# --------------------------------------------------------------------- Axes --


class Input:
    @property
    def action_space(self) -> gym.spaces.Space:
        raise NotImplementedError()


@dataclass
class DiscreteAxis(Input):
    neg_key: int
    pos_key: int

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(3)


@dataclass
class DiscreteTrigger(Input):
    key: int

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(2)


# ------------------------------------------------------------------ Keymaps --


KEYMAP_KEYBOARD_4DOF = {
    'x': DiscreteAxis(ecodes.KEY_A, ecodes.KEY_D),
    'y': DiscreteAxis(ecodes.KEY_E, ecodes.KEY_Q),
    'z': DiscreteAxis(ecodes.KEY_S, ecodes.KEY_W),
    'roll': DiscreteAxis(ecodes.KEY_U, ecodes.KEY_O),
    'gripper': DiscreteTrigger(ecodes.KEY_SPACE),
    'speed': DiscreteTrigger(ecodes.KEY_LEFTSHIFT),
    'home': DiscreteTrigger(ecodes.KEY_HOME),
    'exit': DiscreteTrigger(ecodes.KEY_ESC),
}

KEYMAP_KEYBOARD_6DOF = {
    **KEYMAP_KEYBOARD_4DOF,
    'pitch': DiscreteAxis(ecodes.KEY_K, ecodes.KEY_I),
    'yaw': DiscreteAxis(ecodes.KEY_J, ecodes.KEY_L),
}

# ---------------------------------------------------------------------- HID --

@dataclass
class HIDConfig:
    device: str


class HID:
    def __init__(self, device: evdev.InputDevice, keymap: Dict[str, Input]):
        self._device = device
        self._keymap = copy.deepcopy(keymap)
        self._action_space = gym.spaces.Dict({
            name: input.action_space
            for name, input in self._keymap.items()
        })

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    def get_action(self):
        action = {}
        active_keys = self._device.active_keys()

        for name, input in self._keymap.items():
            if isinstance(input, DiscreteAxis):
                neg = input.neg_key in active_keys
                pos = input.pos_key in active_keys
                action[name] = 1 + int(pos) - int(neg)
            elif isinstance(input, DiscreteTrigger):
                action[name] = int(input.key in active_keys)
            else:
                assert 0

        return action

    def visualize_in_console(self):
        print()
        while True:
            action = self.get_action()
            print('\33[2K\r', end='')  # erase line
            print(action, end='', flush=True)
            time.sleep(0.01)


# ------------------------------------------------------------------ Devices --


def list_device_names() -> List[str]:
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    return [d.name for d in devices]


def find_device_by_name(name: str) -> evdev.InputDevice:
    """
    Find a device with given name and return it.

    - If no device with such name is found, raise RuntimeError.
    - If multiple devices with such name are found, raise RuntimeError.
    """
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    devices = [d for d in devices if d.name == name]
    if not devices:
        raise RuntimeError(f'No device with name={name!r} found')
    if len(devices) > 1:
        raise RuntimeError(
            f'More than 1 device with name={name!r} found: '
            f'{[(d.path, d.phys) for d in devices]}'
        )

    return devices[0]
