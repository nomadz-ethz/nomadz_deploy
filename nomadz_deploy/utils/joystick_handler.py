"""Joystick input handling for teleoperation."""

import warnings
import os

# Suppress pygame warnings before importing
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Import pygame with warnings suppressed
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pygame

import threading
import time
from collections import deque
from typing import Optional, Tuple


# Standard Xbox-style pad mapping under Linux/SDL. These are pygame button
# indices (passed to joystick.get_button(i)). They are empirically validated
# in step 1.5 of the recovery integration validation plan — if the operator's
# pad reports different indices, override these via the JoystickHandler ctor.
BUTTON_A = 0
BUTTON_B = 1
BUTTON_X = 2
BUTTON_Y = 3


class JoystickHandler:
    """Handles Xbox controller input for teleoperation.

    In addition to the original axis (left stick / right stick X) reads, this
    handler now polls discrete buttons and exposes a rising-edge consumer
    (``consume_press``) so callers can implement single-shot reactions to
    button presses without debouncing the polling rate manually.
    """

    def __init__(self, deadzone: float = 0.1, num_buttons: int = 10):
        self.deadzone = deadzone
        self.joystick: Optional[pygame.joystick.Joystick] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Current joystick values
        self.left_stick_x = 0.0  # lateral velocity (left/right)
        self.left_stick_y = 0.0  # forward velocity (up/down)
        self.right_stick_x = 0.0  # yaw velocity (left/right)

        # Calibration values
        self.calibrated = False
        self.center_left_x = 0.0
        self.center_left_y = 0.0
        self.center_right_x = 0.0

        # Button state. ``_button_prev`` holds the last polled level for each
        # button; ``_button_pressed_edges`` is a queue of (button_id) entries
        # for every rising edge that hasn't been consumed yet. consume_press()
        # drains a specific id from the queue. The lock guards both fields
        # because update_values() runs on the polling thread and consume_press
        # is called from the state-machine thread.
        self._num_buttons = num_buttons
        self._button_prev = [False] * num_buttons
        self._button_pressed_edges: deque = deque()
        self._button_lock = threading.Lock()

        self._init_pygame()

    def _init_pygame(self):
        """Initialize pygame and joystick."""
        # Suppress pygame output completely
        import os
        import sys
        from io import StringIO
        
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
        
        # Temporarily redirect stdout to suppress pygame messages
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            pygame.init()
            pygame.joystick.init()
        finally:
            sys.stdout = old_stdout

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected. Please connect an Xbox controller.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Initialized joystick: {self.joystick.get_name()}")

    def calibrate(self, duration: float = 2.0):
        """Calibrate joystick centers by averaging readings over time."""
        print("Calibrating joystick... Please keep sticks centered.")

        readings_left_x = []
        readings_left_y = []
        readings_right_x = []

        start_time = time.time()
        while time.time() - start_time < duration:
            pygame.event.pump()
            readings_left_x.append(self.joystick.get_axis(0))  # Left stick X
            readings_left_y.append(self.joystick.get_axis(1))  # Left stick Y
            readings_right_x.append(self.joystick.get_axis(3))  # Right stick X
            time.sleep(0.01)

        self.center_left_x = sum(readings_left_x) / len(readings_left_x)
        self.center_left_y = sum(readings_left_y) / len(readings_left_y)
        self.center_right_x = sum(readings_right_x) / len(readings_right_x)

        self.calibrated = True
        print("Calibration complete.")

    def _apply_deadzone(self, value: float, center: float) -> float:
        """Apply deadzone to axis value."""
        adjusted = value - center
        if abs(adjusted) < self.deadzone:
            return 0.0
        return max(-1.0, min(1.0, adjusted))

    def update_values(self):
        """Update joystick values from current readings."""
        if not self.joystick or not self.calibrated:
            return

        pygame.event.pump()

        # Left stick: forward/backward (Y) and left/right (X)
        raw_left_x = self.joystick.get_axis(0)
        raw_left_y = self.joystick.get_axis(1)

        self.left_stick_x = self._apply_deadzone(raw_left_x, self.center_left_x)
        self.left_stick_y = self._apply_deadzone(raw_left_y, self.center_left_y)

        # Right stick: yaw (X only)
        raw_right_x = self.joystick.get_axis(3)
        self.right_stick_x = self._apply_deadzone(raw_right_x, self.center_right_x)

        # Buttons: rising-edge detection. We poll because pygame's event API
        # would require draining JOYBUTTONDOWN/UP events from the same queue
        # we already pump for axis updates, and event.pump() doesn't return
        # them. get_button() is synchronous and gives us the current level.
        with self._button_lock:
            for i in range(self._num_buttons):
                cur = bool(self.joystick.get_button(i))
                if cur and not self._button_prev[i]:
                    self._button_pressed_edges.append(i)
                self._button_prev[i] = cur

    def get_velocities(self, vx_max: float, vy_max: float, vyaw_max: float) -> Tuple[float, float, float]:
        """Get velocity commands from joystick.

        Returns:
            Tuple of (forward_vel, lateral_vel, yaw_vel)
        """
        if not self.calibrated:
            return 0.0, 0.0, 0.0

        # Left stick Y: forward/backward (negative Y is forward in pygame)
        forward_vel = -self.left_stick_y * vx_max

        # Left stick X: left/right, inverted to match desired robot motion
        lateral_vel = -self.left_stick_x * vy_max

        # Right stick X: yaw, inverted to match desired robot motion
        yaw_vel = -self.right_stick_x * vyaw_max

        return forward_vel, lateral_vel, yaw_vel

    def axes(self) -> Tuple[float, float, float]:
        """Return the latest stick values as ``(vx, vy, vyaw)`` in [-1, 1].

        Same sign convention as ``get_velocities`` (forward is +x, left is +y,
        CCW yaw is +z) but unscaled — the caller multiplies by per-task limits.
        """
        if not self.calibrated:
            return 0.0, 0.0, 0.0
        return -self.left_stick_y, -self.left_stick_x, -self.right_stick_x

    def consume_press(self, button_id: int) -> bool:
        """Return True exactly once per rising edge of ``button_id``.

        Drains the next pending press of ``button_id`` from the edge queue.
        Other buttons' edges remain queued. Used by the recovery state machine
        to react to Y/B/X presses without firing repeatedly while held.
        """
        with self._button_lock:
            try:
                self._button_pressed_edges.remove(button_id)
                return True
            except ValueError:
                return False

    def drain_button_events(self) -> list:
        """Drain and return all pending press edges as a list of button ids.

        Useful for diagnostic publishers that want to emit a ROS message per
        button press. Returns an empty list if there are no pending edges.
        """
        with self._button_lock:
            events = list(self._button_pressed_edges)
            self._button_pressed_edges.clear()
            return events

    def start(self):
        """Start joystick reading thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop joystick reading."""
        self.running = False
        if self.thread:
            self.thread.join()

    def _read_loop(self):
        """Main reading loop."""
        while self.running:
            self.update_values()
            time.sleep(0.01)  # 100Hz update rate

    def __del__(self):
        """Cleanup pygame on destruction."""
        self.stop()
        pygame.quit()
