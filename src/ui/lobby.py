"""Startbildschirm.

Ein Spieler, ein Knopf: "START THE PARTY". Ausloesen geht per Tastatur
(Enter oder Leertaste) oder ohne Tastatur, indem man die Hand ueber den
Knopf haelt, bis der Ring voll ist. Das ist auf einer Party wichtig,
weil niemand zum Laptop laufen soll.

Gezeichnet wird mit denselben Bausteinen wie die Spielanzeige - gleiche
Schrift, gleiche Farben, gleiches Glas. Der Startbildschirm ist der
erste Eindruck; sieht er aus wie ein anderes Programm, wirkt das ganze
Instrument billiger, als es ist.
"""

from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np

from ..config import Config
from ..vision.types import HandObservation
from .theme import (
    ACCENT,
    ACCENT_DIM,
    INK,
    INK_FAINT,
    INK_SOFT,
    LINE_SOFT,
    PANEL,
    TRACK_HERO,
    TYPE_HERO,
    TYPE_LABEL,
    TYPE_MICRO,
    TYPE_TITLE,
    Painter,
    arc,
    backdrop,
    clamp01,
    label,
    label_width,
    mix,
    scale,
    text,
    text_width,
    unit,
)

GESTURES = (
    ("Pinch", "Loop nehmen"),
    ("Hand neigen", "blättern und lauter"),
    ("Finger zeigen", "Beat verdichten"),
    ("Handgelenk drehen", "Loop halten"),
    ("Hände zusammen", "verschmelzen"),
)


class Lobby:
    """Ein einziger Startknopf statt Spieleranzahl-Auswahl."""

    def __init__(self, config: Config, dwell_frames: int = 34):
        self.config = config
        self.dwell_frames = dwell_frames
        self.progress = 0.0
        self._pulse = 0.0
        self.paint = Painter()
        self.u = 1.0
        # Knopf mittig, normierte Koordinaten.
        self.rect = (0.30, 0.45, 0.40, 0.145)

    # ------------------------------------------------------------------
    def px(self, value: float) -> int:
        return int(round(value * self.u))

    def _contains(self, x: float, y: float) -> bool:
        rx, ry, rw, rh = self.rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def update(self, observations: List[HandObservation]) -> Optional[int]:
        """Gibt 1 zurueck, sobald der Startknopf voll aufgeladen ist."""
        self._pulse += 0.08
        centers = [(float(o.center[0]), float(o.center[1])) for o in observations]
        hovered = any(self._contains(x, y) for x, y in centers)

        if hovered:
            self.progress += 1.0 / max(self.dwell_frames, 1)
            if self.progress >= 1.0:
                return 1
        else:
            self.progress = max(0.0, self.progress - 0.05)
        return None

    def handle_key(self, key: int) -> Optional[int]:
        if key in (13, 10, 32):        # Enter oder Leertaste
            return 1
        return None

    def reset(self) -> None:
        self.progress = 0.0

    # ------------------------------------------------------------------
    def render(self, image: np.ndarray) -> np.ndarray:
        canvas = image.copy()
        h, w = canvas.shape[:2]
        self.u = unit(h)
        backdrop(canvas, 0.86)

        # Wortmarke: gesperrte Versalien ueber einer feinen Linie. Die
        # Sperrung macht aus zwei Woertern ein Logo.
        top = int(h * 0.20)
        text(canvas, "LOOP FORGE", w // 2, top, TYPE_HERO * self.u, INK, "bold",
             self.px(TRACK_HERO), align="center")
        width = text_width("LOOP FORGE", TYPE_HERO * self.u, "bold",
                           self.px(TRACK_HERO))
        cv2.line(canvas, (w // 2 - width // 2, top + self.px(16)),
                 (w // 2 + width // 2, top + self.px(16)), LINE_SOFT, 1, cv2.LINE_AA)
        label(canvas, "gestengesteuertes Instrument", w // 2, top + self.px(38),
              TYPE_LABEL * self.u, INK_FAINT, align="center")

        self._draw_cta(canvas)

        hint_y = int((self.rect[1] + self.rect[3]) * h) + self.px(52)
        label(canvas, "Hand über den Knopf halten  ·  oder Enter", w // 2, hint_y,
              TYPE_MICRO * self.u, INK_SOFT, align="center")

        self._draw_gestures(canvas)
        return canvas

    def _draw_gestures(self, canvas: np.ndarray) -> None:
        """Die fuenf Gesten in einer Zeile. Wer das liest, kann sofort spielen."""
        h, w = canvas.shape[:2]
        size = TYPE_MICRO * self.u
        widths = [label_width(key, size) + self.px(10)
                  + text_width(value, size, "regular")
                  for key, value in GESTURES]
        gap = self.px(30)
        total = sum(widths) + gap * (len(GESTURES) - 1)
        cursor = (w - total) // 2
        baseline = int(h * 0.87)
        cv2.line(canvas, (cursor - self.px(34), baseline - self.px(28)),
                 (cursor + total + self.px(34), baseline - self.px(28)),
                 LINE_SOFT, 1, cv2.LINE_AA)
        for (key, value), width in zip(GESTURES, widths):
            after = label(canvas, key, cursor, baseline, size, INK_SOFT)
            text(canvas, value, after + self.px(10), baseline, size,
                 scale(INK_FAINT, 0.95), "regular")
            cursor += width + gap

    def _draw_cta(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        rx, ry, rw, rh = self.rect
        x, y = int(rx * w), int(ry * h)
        width, height = int(rw * w), int(rh * h)
        cx, cy = x + width // 2, y + height // 2
        radius = height // 2

        breathe = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(self._pulse))
        active = self.progress > 0.01
        tone = ACCENT if active else mix(ACCENT_DIM, ACCENT, breathe)

        # Der Knopf atmet: ein weicher Schein darunter, der mit dem Puls
        # kommt und geht. Das zieht den Blick, ohne zu blinken.
        self.paint.glow(canvas, cx, cy, int(width * 0.62), ACCENT,
                        0.10 + 0.16 * breathe + 0.25 * self.progress)
        self.paint.shadow(canvas, (x, y, width, height), radius,
                          spread=self.px(26), strength=0.6, drop=self.px(8))
        self.paint.glass(canvas, (x, y, width, height), radius,
                         tint=mix(PANEL, ACCENT_DIM, 0.25 + 0.3 * self.progress),
                         alpha=0.95)
        self.paint.outline(canvas, (x, y, width, height), radius, tone, 2)

        text(canvas, "START THE PARTY", cx, cy + self.px(9), TYPE_TITLE * self.u * 1.5,
             INK, "bold", self.px(4), align="center")

        # Aufladung als Ring an der linken Seite des Knopfs plus Linie auf
        # der Unterkante - eines zeigt die Zeit, das andere den Fortschritt.
        if active:
            span = width - 2 * radius
            fill = int(span * clamp01(self.progress))
            cv2.line(canvas, (x + radius, y + height - 1),
                     (x + radius + fill, y + height - 1), ACCENT,
                     max(2, self.px(3)), cv2.LINE_AA)
            ring = int(radius * 0.42)
            cv2.circle(canvas, (x + radius, cy), ring, scale(INK, 0.18), 1, cv2.LINE_AA)
            arc(canvas, x + radius, cy, ring, self.progress, ACCENT, max(2, self.px(3)))


def countdown_overlay(image: np.ndarray, seconds_left: float) -> np.ndarray:
    """Kurzer Countdown, damit die Leute sich vor dem Start aufstellen."""
    canvas = image.copy()
    h, w = canvas.shape[:2]
    u = unit(h)
    backdrop(canvas, 0.58)

    seconds = max(1, int(math.ceil(seconds_left)))
    # Der Ring laeuft innerhalb der laufenden Sekunde einmal herum.
    fraction = 1.0 - (seconds_left - math.floor(seconds_left))
    radius = int(min(w, h) * 0.11)
    cx, cy = w // 2, int(h * 0.47)

    paint = Painter()
    paint.glow(canvas, cx, cy, int(radius * 1.9), ACCENT, 0.22)
    cv2.circle(canvas, (cx, cy), radius, scale(INK, 0.16), 1, cv2.LINE_AA)
    arc(canvas, cx, cy, radius, fraction, ACCENT, max(2, int(3 * u)))

    text(canvas, str(seconds), cx, cy + int(radius * 0.42), TYPE_HERO * u * 1.9,
         INK, "bold", align="center")
    label(canvas, "gleich geht es los", cx, cy + radius + int(48 * u),
          TYPE_LABEL * u, INK_SOFT, align="center", tracking=int(3 * u))
    return canvas
