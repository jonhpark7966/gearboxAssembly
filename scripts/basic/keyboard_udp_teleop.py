#!/usr/bin/env python3
"""
Keyboard-only UDP teleop sender for the gearboxAssembly demo.

This feeds the UDP receiver in `assembly_scene_teleop_demo.py`:
  - Run the Isaac scene (listens on port 5005 by default)
  - Run this script in a terminal to stream commands

Default key mapping (lowercase letters):
  Left arm joints : q/a, w/s, e/d, r/f, t/g, y/h  (+ / -)
  Right arm joints: u/j, i/k, o/l, p/; , [/' , ]/\
  Grippers        : z/x (left open/close), n/m (right open/close)
  Other           : space (zero arms), r (reset grippers open),
                    c (center pose), q (quit)
"""

import argparse
import json
import select
import socket
import sys
import termios
import time
import tty


def _read_key(timeout: float) -> str | None:
    """Non-blocking single-char read from stdin."""
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.read(1)
    return None


class _RawStdin:
    """Context manager to switch stdin to raw mode and restore on exit."""

    def __enter__(self):
        self._old = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._old:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def main():
    parser = argparse.ArgumentParser(description="Keyboard UDP teleop sender")
    parser.add_argument("--ip", default="127.0.0.1", help="Receiver IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5005, help="Receiver UDP port (default: 5005)")
    parser.add_argument("--step", type=float, default=2.0, help="Joint increment in degrees per keypress")
    parser.add_argument("--grip-step", type=float, default=0.05, help="Gripper increment per keypress (0-1)")
    parser.add_argument("--rate", type=float, default=30.0, help="Send rate in Hz")
    args = parser.parse_args()

    left_deg = [0.0] * 6
    right_deg = [0.0] * 6
    left_grip = 1.0   # 1.0 ≈ open (scaled to 0.04 in the sim script)
    right_grip = 1.0

    # Key mappings for joint increments
    keymap = {
        "q": ("l", 0, +1), "a": ("l", 0, -1),
        "w": ("l", 1, +1), "s": ("l", 1, -1),
        "e": ("l", 2, +1), "d": ("l", 2, -1),
        "r": ("l", 3, +1), "f": ("l", 3, -1),
        "t": ("l", 4, +1), "g": ("l", 4, -1),
        "y": ("l", 5, +1), "h": ("l", 5, -1),

        "u": ("r", 0, +1), "j": ("r", 0, -1),
        "i": ("r", 1, +1), "k": ("r", 1, -1),
        "o": ("r", 2, +1), "l": ("r", 2, -1),
        "p": ("r", 3, +1), ";": ("r", 3, -1),
        "[": ("r", 4, +1), "'": ("r", 4, -1),
        "]": ("r", 5, +1), "\\": ("r", 5, -1),
    }

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.ip, args.port)
    send_period = 1.0 / args.rate
    last_print = 0.0

    def print_state(force=False):
        nonlocal last_print
        now = time.time()
        if not force and now - last_print < 0.5:
            return
        last_print = now
        msg = (
            f"L(deg): {['%5.1f' % v for v in left_deg]}  "
            f"R(deg): {['%5.1f' % v for v in right_deg]}  "
            f"Lgrip: {left_grip:0.2f}  Rgrip: {right_grip:0.2f}"
        )
        print("\r" + msg + " " * 4, end="", flush=True)

    print("Keyboard UDP teleop -> %s:%d" % dest)
    print("Press mapped keys to move, space to zero arms, c to center, r to open grippers, q to quit.")

    try:
        with _RawStdin():
            while True:
                start = time.time()
                key = _read_key(timeout=0.0)

                if key:
                    if key == "q":
                        break
                    elif key == " ":
                        left_deg = [0.0] * 6
                        right_deg = [0.0] * 6
                    elif key == "c":
                        # A mild neutral pose
                        left_deg = [0.0, -20.0, 40.0, 0.0, 40.0, 0.0]
                        right_deg = [0.0, -20.0, 40.0, 0.0, 40.0, 0.0]
                    elif key == "r":
                        left_grip = 1.0
                        right_grip = 1.0
                    elif key == "z":
                        left_grip = clamp(left_grip + args.grip_step, 0.0, 1.0)
                    elif key == "x":
                        left_grip = clamp(left_grip - args.grip_step, 0.0, 1.0)
                    elif key == "n":
                        right_grip = clamp(right_grip + args.grip_step, 0.0, 1.0)
                    elif key == "m":
                        right_grip = clamp(right_grip - args.grip_step, 0.0, 1.0)
                    elif key in keymap:
                        arm, idx, sign = keymap[key]
                        if arm == "l":
                            left_deg[idx] += sign * args.step
                        else:
                            right_deg[idx] += sign * args.step

                payload = {
                    "ts": time.time(),
                    "left_arm_deg": left_deg,
                    "right_arm_deg": right_deg,
                    "left_gripper": left_grip,
                    "right_gripper": right_grip,
                }
                sock.sendto((json.dumps(payload) + "\n").encode("utf-8"), dest)
                print_state()

                elapsed = time.time() - start
                to_sleep = send_period - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
    finally:
        print("\nExiting...")
        sock.close()


if __name__ == "__main__":
    main()
