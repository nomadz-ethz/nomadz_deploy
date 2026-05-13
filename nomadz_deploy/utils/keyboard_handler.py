"""Keyboard input handler — drop-in alternative to ``JoystickHandler``.

Lets the operator drive the recovery state machine with the keyboard before
they have a physical pygame controller plugged in (e.g. on the workstation
during sim2sim, or on the robot during early bring-up). Exposes the same
public interface the portal and FSM consume:

    .calibrate(), .start(), .stop()
    .axes() -> (vx, vy, vyaw) in [-1, 1]
    .get_velocities(vx_max, vy_max, vyaw_max) -> scaled tuple
    .consume_press(button_id) -> bool        # rising-edge consumer
    .drain_button_events() -> list[int]
    .calibrated: bool
    BUTTON_A / BUTTON_B / BUTTON_X / BUTTON_Y constants on the module

Key bindings (single-char, cbreak stdin, no Enter required):

* ``w`` / ``s`` — increment / decrement ``vx`` by ``axis_step`` (clamped to ±1)
* ``a`` / ``d`` — increment / decrement ``vy``
* ``q`` / ``e`` — increment / decrement ``vyaw``
* `` `` (space) — zero all three axes
* ``y`` / ``b`` / ``x`` — emit a single press of BUTTON_Y / BUTTON_B / BUTTON_X
* ``Ctrl-C`` — propagated to the parent (the portal's signal handler exits)

The handler runs a daemon thread that reads stdin in cbreak mode (no echo,
no line buffering). On stop, terminal settings are restored.

See ``docs/RECOVERY_INTEGRATION_PLAN.md`` and
``docs/2026-05-08-recoveryprototype.md`` for how this fits into the FSM.
"""
from __future__ import annotations

import os
import select
import sys
import termios
import threading
import tty
from collections import deque
from typing import Tuple

# Re-export the same button constants JoystickHandler defines, so callers
# can import either module and still address `BUTTON_Y` etc. by the same
# integer value.
from .joystick_handler import (
    BUTTON_A,
    BUTTON_B,
    BUTTON_X,
    BUTTON_Y,
)

__all__ = [
    "KeyboardHandler",
    "BUTTON_A",
    "BUTTON_B",
    "BUTTON_X",
    "BUTTON_Y",
]


class KeyboardHandler:
    """Stdin-driven stand-in for :class:`JoystickHandler`.

    Parameters
    ----------
    axis_step:
        How much each w/s/a/d/q/e press changes the axis target. Default
        ``0.25`` so four presses saturate the axis. Press ``space`` to
        zero everything.
    """

    def __init__(self, axis_step: float = 0.25) -> None:
        self.axis_step = float(axis_step)

        # Public-shape attributes mirrored from JoystickHandler so the FSM
        # and the portal don't need a different code path.
        self.calibrated: bool = False
        self.running: bool = False
        self.thread: threading.Thread | None = None

        # Axis state, signed, in [-1, 1]. Matches the sign convention the
        # portal expects: vx forward, vy left, vyaw counter-clockwise.
        self._vx = 0.0
        self._vy = 0.0
        self._vyaw = 0.0

        # Edge-triggered button queue — same semantics as
        # JoystickHandler._button_pressed_edges so consume_press() is a
        # drop-in.
        self._button_pressed_edges: deque[int] = deque()
        self._button_lock = threading.Lock()

        # Saved tty attrs for stop().
        self._stdin_fd = sys.stdin.fileno()
        self._old_term_attrs = None
        self._tty_is_active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def calibrate(self, duration: float = 0.0) -> None:
        """No-op. Present so the portal's calibrate-then-start chain just
        works regardless of which handler is in use.
        """
        self.calibrated = True
        print("Keyboard handler ready. "
              "Use w/s/a/d/q/e for axes (Space = zero); y/b/x for buttons.")

    def start(self) -> None:
        if self.running:
            return
        if not sys.stdin.isatty():
            raise RuntimeError(
                "KeyboardHandler needs a real TTY on stdin — got a pipe or "
                "redirect. Run the deploy in an interactive terminal.")
        # Switch stdin to cbreak so we get one keystroke at a time without
        # waiting for Enter, and without echoing characters back into the
        # log stream (otherwise every press shows up in the deploy log).
        self._old_term_attrs = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)
        self._tty_is_active = True
        self.running = True
        self.thread = threading.Thread(
            target=self._read_loop, name="kbd_handler", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=0.5)
        # Restore the terminal whether or not the thread joined cleanly —
        # leaving cbreak on means the operator's shell is broken after
        # this exits.
        if self._tty_is_active and self._old_term_attrs is not None:
            try:
                termios.tcsetattr(
                    self._stdin_fd, termios.TCSADRAIN, self._old_term_attrs)
            except Exception:
                pass
            self._tty_is_active = False

    # ------------------------------------------------------------------
    # Public interface (matches JoystickHandler)
    # ------------------------------------------------------------------

    def axes(self) -> Tuple[float, float, float]:
        if not self.calibrated:
            return 0.0, 0.0, 0.0
        return self._vx, self._vy, self._vyaw

    def get_velocities(
        self, vx_max: float, vy_max: float, vyaw_max: float
    ) -> Tuple[float, float, float]:
        if not self.calibrated:
            return 0.0, 0.0, 0.0
        return (
            self._vx * vx_max,
            self._vy * vy_max,
            self._vyaw * vyaw_max,
        )

    def consume_press(self, button_id: int) -> bool:
        """Return True exactly once per rising edge of ``button_id``."""
        with self._button_lock:
            try:
                self._button_pressed_edges.remove(button_id)
                return True
            except ValueError:
                return False

    def drain_button_events(self) -> list[int]:
        with self._button_lock:
            events = list(self._button_pressed_edges)
            self._button_pressed_edges.clear()
            return events

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        """Block on stdin in cbreak mode, dispatch each keystroke."""
        while self.running:
            # 100 ms timeout so we wake up to check `self.running` even
            # when the operator isn't typing.
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            try:
                ch = os.read(self._stdin_fd, 1).decode(errors="ignore")
            except OSError:
                break
            if not ch:
                continue
            self._handle_char(ch.lower())

    def _handle_char(self, ch: str) -> None:
        # Ctrl-C arrives as the literal byte 0x03 in cbreak; raise SIGINT
        # so the portal's signal handler runs cleanup.
        if ch == "\x03":
            os.kill(os.getpid(), 2)  # SIGINT
            return
        if ch == "w":
            self._vx = _clamp(self._vx + self.axis_step)
        elif ch == "s":
            self._vx = _clamp(self._vx - self.axis_step)
        elif ch == "a":
            self._vy = _clamp(self._vy + self.axis_step)
        elif ch == "d":
            self._vy = _clamp(self._vy - self.axis_step)
        elif ch == "q":
            self._vyaw = _clamp(self._vyaw + self.axis_step)
        elif ch == "e":
            self._vyaw = _clamp(self._vyaw - self.axis_step)
        elif ch == " ":
            self._vx = 0.0
            self._vy = 0.0
            self._vyaw = 0.0
        elif ch == "y":
            self._enqueue_press(BUTTON_Y)
        elif ch == "b":
            self._enqueue_press(BUTTON_B)
        elif ch == "x":
            self._enqueue_press(BUTTON_X)
        # Unknown chars are silently ignored — the operator's likely just
        # typing into the wrong terminal.

    def _enqueue_press(self, button_id: int) -> None:
        with self._button_lock:
            self._button_pressed_edges.append(button_id)


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))
