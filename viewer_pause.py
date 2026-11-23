"""Pause Python-driven simulation while keeping the viewer responsive."""
from threading import Event
from time import sleep

from glfw import KEY_SPACE


class ViewerPause:
    def __init__(self):
        self.paused = Event()

    def on_key(self, key):
        if key == KEY_SPACE:
            if self.paused.is_set():
                self.paused.clear()
            else:
                self.paused.set()

    def ready(self, viewer, pump=None):
        while viewer.is_running():
            if not self.paused.is_set():
                return True
            viewer.sync()
            if pump is not None:
                pump()
            sleep(.02)
        return False
