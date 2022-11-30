from functools import lru_cache
import evdev
from sshkeyboard import listen_keyboard, stop_listening

from getpass import getuser
import logging
import multiprocessing as mp
import os
import sys

logging.basicConfig()


@lru_cache
def get_ssh_keyboard_device():
    """
    Get ruka.robot.hid compatible sshkeyboard wrapper
    """

    return _SshKeyboardDevice()


class _SshKeyboardDevice:
    """
    ruka.robot.hid compatible sshkeyboard wrapper

    Supprots only active_keys evdev device method
    """

    def __init__(self):
        self.user = getuser()
        self._stdin_fd = sys.stdin.fileno()
        self._manager = mp.Manager()
        self._active_keys = self._manager.dict()
        self._is_listener_running = False
        self._listener_process = mp.Process(name=f'{self.user}-sshkeyboard-listener', target=self._listener_loop, daemon=True)

    def start(self):
        if not self._is_listener_running:
            self._is_listener_running = True
            self._listener_process.start()
        else:
            logging.warning(f'Listener process {self._listener_process.name} is already running!')

    def active_keys(self):
        return self._active_keys.values()

    def stop(self):
        if self._is_listener_running:
            stop_listening()
            self._listener_process.join()
            self._is_listener_running = False
        else:
            logging.warning('Nothing to stop! Listener process has not been started yet')

    def __del__(self):
        self.stop()

    def _listener_loop(self):
        sys.stdin = os.fdopen(self._stdin_fd)  # Catch parent stdin
        listen_keyboard(
            on_press=self._on_press,
            on_release=self._on_release
        )

    def _on_press(self, key):
        key_code = evdev.ecodes.ecodes[f'KEY_{key.upper()}']
        self._active_keys[key] = key_code

    def _on_release(self, key):
        del self._active_keys[key]
