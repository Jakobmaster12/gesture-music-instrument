"""Clean live-performance UI for Loop Forge.

Visual direction:
- circles/orbs instead of dashboard cards
- no rack/shelf metaphor on screen
- held instruments never cover the hand
- parked loops remain at their actual screen position
- restrained typography and line work

The engine state and gesture logic stay in the existing modules.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import Config
from ..mapping import tokens as tk
from ..mapping.presets import colour_of, label_of, preset, short_of
from ..vision.types import HAND_CONNECTIONS, EngineSnapshot, HandView
from .theme import (
    ACCENT, DANGER, INK, INK_FAINT, INK_SOFT, LINE, LINE_SOFT, PANEL,
    TRACK_TITLE, TYPE_HEAD, TYPE_LABEL, TYPE_MICRO, TYPE_TITLE,
    Painter, arc, clamp01, label, label_width, scale, text, text_width, unit,
)

STEPS = 16
LEGEND = (
    ("PINCH", "take / place"),
    ("FIST", "mute"),
    ("OPEN", "play"),
    ("SPREAD", "expression"),
    ("TILT", "volume"),
    ("THUMB", "tone"),
)


class Display:
    def __init__(self, config: Config):
        self.config = config
        self.window = config.ui.window_name
        self.fullscreen = config.ui.fullscreen
        self.paint = Painter()
        self._created = False
        self._vignette: Optional[np.ndarray] = None
        self.u = 1.0

    def px(self, value: float) -> int:
        return int(round(value * self.u))

    def _ensure_window(self) -> None:
        if self._created:
            return
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._created = True

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self._ensure_window()
        cv2.setWindowProperty(
            self.window, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
        )

    def render(self, image: np.ndarray, snapshot: EngineSnapshot,
               status: Optional[str] = None) -> np.ndarray:
        canvas = image.copy()
        h, w = canvas.shape[:2]
        self.u = unit(h)
        pulse = self._beat_pulse()
        step = self._current_step()
        self._damp(canvas)

        if self.config.ui.draw_landmarks:
            for hand in snapshot.hands:
                self._draw_skeleton(canvas, hand)

        self._draw_palette(canvas, snapshot)
        by_token = {t.id: t for t in snapshot.tokens}

        for token in snapshot.tokens:
            if token.state == tk.PLACED:
                self._draw_parked_orb(canvas, token, pulse)
            elif token.state == tk.FUSING:
                self._draw_fusion_orb(canvas, token)

        for hand in snapshot.hands:
            if not hand.holding:
                continue
            token = by_token.get(hand.token_id)
            if token is not None:
                self._draw_held_orb(canvas, hand, token, step)

        self._draw_header(canvas, snapshot, status)
        self._draw_footer(canvas, snapshot)
        return canvas

    def _current_step(self) -> int:
        return int(time.time() * self.config.audio.bpm / 60.0 * 4.0) % STEPS

    def _beat_pulse(self) -> float:
        beats = time.time() * self.config.audio.bpm / 60.0
        return math.exp(-5.0 * (beats % 1.0))

    def _damp(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        if self._vignette is None or self._vignette.shape[:2] != (h, w):
            ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
            xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
            radial = np.sqrt(xs * xs + ys * ys) / 1.414
            fade = (0.43 - 0.20 * radial).astype(np.float32)
            self._vignette = cv2.merge([fade, fade, fade])
        cv2.multiply(canvas, self._vignette, dst=canvas, dtype=cv2.CV_8U)

    def _draw_skeleton(self, canvas: np.ndarray, hand: HandView) -> None:
        if hand.landmarks is None:
            return
        h, w = canvas.shape[:2]
        points = [(int(p[0] * w), int(p[1] * h)) for p in hand.landmarks]
        shade = scale(INK, 0.22 if hand.holding else 0.12)
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, points[a], points[b], shade, 1, cv2.LINE_AA)
        for idx in (4, 8, 12, 16, 20):
            cv2.circle(canvas, points[idx], max(self.px(1.5), 1), shade, -1, cv2.LINE_AA)

    # -------------------------- palette --------------------------------
    def _draw_palette(self, canvas: np.ndarray, snapshot: EngineSnapshot) -> None:
        if not snapshot.library:
            return
        x = self.px(52)
        top = self.px(120)
        gap = self.px(64)
        label(canvas, "SOUNDS", x, self.px(72), TYPE_LABEL * self.u, INK_FAINT, tracking=self.px(2.8))
        label(canvas, "PINCH TO TAKE", x, self.px(89), TYPE_MICRO * self.u, INK_FAINT, tracking=self.px(1.0))
        hovered = {h.selection for h in snapshot.hands if not h.holding and h.zone == tk.LIBRARY}
        radius = self.px(23)
        for i, entry in enumerate(snapshot.library[:8]):
            cy = top + i * gap
            colour = colour_of(entry.preset_ids[0])
            active = i in hovered
            if active:
                self.paint.glow(canvas, x, cy, radius * 3, colour, 0.24)
            cv2.circle(canvas, (x, cy), radius, scale(PANEL, 0.96), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, cy), radius, scale(colour, 0.92), 2 if active else 1, cv2.LINE_AA)
            text(canvas, short_of(entry.preset_ids[0]), x, cy + self.px(6), TYPE_HEAD * self.u,
                 INK if active else INK_SOFT, "bold", TRACK_TITLE, align="center")
            label(canvas, label_of(entry.preset_ids[0]), x + radius + self.px(12), cy + self.px(4),
                  TYPE_MICRO * self.u, INK_SOFT if active else INK_FAINT)

    # ----------------------- parked loops -------------------------------
    def _draw_parked_orb(self, canvas: np.ndarray, token, pulse: float) -> None:
        h, w = canvas.shape[:2]
        cx, cy = int(token.x * w), int(token.y * h)
        colour = colour_of(token.preset_ids[0])
        volume = 0.0 if token.muted else max((r.volume for r in token.recipes), default=0.0)
        radius = self.px(30 + 4 * pulse * volume)
        self.paint.glow(canvas, cx, cy, int(radius * 2.2), colour, 0.05 + 0.10 * pulse * volume)
        cv2.circle(canvas, (cx, cy), radius, scale(PANEL, 0.96), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), radius, scale(colour, 0.88), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), self.px(4), colour, -1, cv2.LINE_AA)
        text(canvas, "+".join(short_of(pid) for pid in token.preset_ids), cx, cy + self.px(5),
             TYPE_HEAD * self.u, INK, "bold", TRACK_TITLE, align="center")
        label(canvas, label_of(token.preset_ids[0]), cx, cy + radius + self.px(16),
              TYPE_MICRO * self.u, INK_FAINT, align="center")
        arc(canvas, cx, cy, radius + self.px(6), self._loop_phase(max(preset(token.preset_ids[0]).bars, 1)), colour, self.px(2))

    # ------------------------ held instrument ---------------------------
    def _draw_held_orb(self, canvas: np.ndarray, hand: HandView, token, step: int) -> None:
        h, w = canvas.shape[:2]
        hx, hy = int(hand.x * w), int(hand.y * h)
        colour = colour_of(token.preset_ids[0])
        active = not token.muted
        radius = self.px(35)

        # Only a ring around the hand. The hand itself stays completely readable.
        if active:
            self.paint.glow(canvas, hx, hy, radius * 2, colour, 0.08 + 0.10 * token.recipes[0].volume)
        cv2.circle(canvas, (hx, hy), radius, scale(colour, 0.32), 2, cv2.LINE_AA)
        cv2.circle(canvas, (hx, hy), radius + self.px(7), scale(colour, 0.14), 1, cv2.LINE_AA)

        # Label is placed away from the hand, not on top of it.
        sx = self.px(84) if hand.x < 0.5 else -self.px(84)
        sy = -self.px(42)
        lx = max(self.px(140), min(w - self.px(140), hx + sx))
        ly = max(self.px(112), min(h - self.px(90), hy + sy))
        name = "+".join(short_of(pid) for pid in token.preset_ids)
        state = "PLAY" if active else "MUTE"
        width = text_width(name, TYPE_LABEL * self.u, "bold", TRACK_TITLE) + label_width(state, TYPE_MICRO * self.u) + self.px(24)
        hh = self.px(25)
        rect = (lx - width // 2, ly - hh // 2, width, hh)
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), scale(PANEL, 0.94), -1)
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), scale(colour, 0.65), 1, cv2.LINE_AA)
        text(canvas, name, rect[0] + self.px(9), ly + self.px(5), TYPE_LABEL * self.u, INK, "bold", TRACK_TITLE)
        label(canvas, state, rect[0] + rect[2] - self.px(9), ly + self.px(4), TYPE_MICRO * self.u,
              colour if active else DANGER, align="right")

        # Small beat ticks orbit the ring.
        angle = step / STEPS * math.tau
        tx = int(hx + math.cos(angle) * (radius + self.px(7)))
        ty = int(hy + math.sin(angle) * (radius + self.px(7)))
        cv2.circle(canvas, (tx, ty), self.px(3), INK, -1, cv2.LINE_AA)

    def _draw_fusion_orb(self, canvas: np.ndarray, token) -> None:
        h, w = canvas.shape[:2]
        cx, cy = int(token.x * w), int(token.y * h)
        p = clamp01(token.fuse_progress)
        radius = self.px(38 + 12 * p)
        self.paint.glow(canvas, cx, cy, int(radius * 2.4), ACCENT, 0.20 * (1.0 - p))
        cv2.circle(canvas, (cx, cy), radius, scale(PANEL, 0.94), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), radius, ACCENT, 2, cv2.LINE_AA)
        arc(canvas, cx, cy, radius + self.px(6), p, INK, self.px(2))

    # -------------------------- header ----------------------------------
    def _draw_header(self, canvas: np.ndarray, snapshot: EngineSnapshot, status: Optional[str]) -> None:
        h, w = canvas.shape[:2]
        held = sum(1 for t in snapshot.tokens if t.held)
        placed = sum(1 for t in snapshot.tokens if t.state == tk.PLACED)
        y = self.px(42)
        text(canvas, "LOOP FORGE", self.px(42), y, TYPE_TITLE * self.u, INK, "bold", self.px(2.2))
        label(canvas, "LIVE / MULTI-USER", self.px(44), y + self.px(22), TYPE_MICRO * self.u, INK_FAINT, tracking=self.px(1.0))

        self._metric(canvas, w // 2 - self.px(70), y, str(held), "IN HAND")
        self._metric(canvas, w // 2 + self.px(60), y, str(placed), "PLACED")

        right = w - self.px(42)
        text(canvas, f"{self.config.audio.bpm:.0f} BPM", right, y, TYPE_HEAD * self.u, INK_SOFT, "medium", align="right")
        label(canvas, f"{snapshot.fps:.0f} FPS", right, y + self.px(21), TYPE_MICRO * self.u, INK_FAINT, align="right")
        if status:
            label(canvas, status.upper(), right, y + self.px(37), TYPE_MICRO * self.u, ACCENT, align="right", tracking=self.px(1.1))

        cv2.line(canvas, (self.px(42), self.px(68)), (w - self.px(42), self.px(68)), LINE_SOFT, 1, cv2.LINE_AA)
        bar_w = self.px(180)
        x0 = (w - bar_w) // 2
        yy = self.px(78)
        cv2.rectangle(canvas, (x0, yy), (x0 + bar_w, yy + self.px(2)), LINE_SOFT, -1)
        cv2.rectangle(canvas, (x0, yy), (x0 + int(bar_w * clamp01(snapshot.macros.riser)), yy + self.px(2)), ACCENT, -1)
        label(canvas, "BUILD", x0 - self.px(10), yy + self.px(5), TYPE_MICRO * self.u, INK_FAINT, align="right")
        if snapshot.macros.drop > 0.25:
            label(canvas, "DROP", w // 2, self.px(104), TYPE_HEAD * self.u, DANGER, "bold", align="center", tracking=self.px(1.5))

    def _metric(self, canvas, x: int, y: int, value: str, name: str) -> None:
        text(canvas, value, x, y, TYPE_HEAD * self.u, INK_SOFT, "medium", align="center")
        label(canvas, name, x, y + self.px(17), TYPE_MICRO * self.u, INK_FAINT, align="center", tracking=self.px(0.7))

    def _draw_footer(self, canvas: np.ndarray, snapshot: EngineSnapshot) -> None:
        h, w = canvas.shape[:2]
        baseline = h - self.px(22)

        
        cv2.line(canvas, (self.px(42), h - self.px(48)), (w - self.px(42), h - self.px(48)), LINE_SOFT, 1, cv2.LINE_AA)
        size = TYPE_MICRO * self.u
        widths = [label_width(k, size) + self.px(7) + text_width(v, size, "regular") for k, v in LEGEND]
        total = sum(widths) + self.px(26) * (len(widths) - 1)
        x = max(self.px(42), (w - total) // 2)
        for (key, value), width in zip(LEGEND, widths):
            after = label(canvas, key, x, baseline, size, INK_SOFT, tracking=self.px(0.7))
            text(canvas, value, after + self.px(7), baseline, size, INK_FAINT, "regular")
            x += width + self.px(26)

    def _loop_phase(self, bars: int) -> float:
        beats = time.time() * self.config.audio.bpm / 60.0
        return (beats / (4.0 * max(bars, 1))) % 1.0

    def show(self, canvas: np.ndarray) -> int:
        self._ensure_window()
        cv2.imshow(self.window, canvas)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        if self._created:
            cv2.destroyAllWindows()
            self._created = False


def save_preview(canvas: np.ndarray, path: str) -> str:
    cv2.imwrite(path, canvas)
    return path
