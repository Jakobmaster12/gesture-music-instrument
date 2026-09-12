"""Kernlogik.

Ablauf pro Frame:

    Haende erkennen
      -> Identitaeten ueber Frames halten (HandTracker)
      -> Loops greifen, formen, halten, fusionieren, ablegen (TokenForge)
      -> Snapshot an OSC, Audio und Anzeige

Es gibt keine festen Kanaele mehr. Was klingt, ergibt sich aus den
Tokens, die gerade in einer Hand liegen oder im Bild abgelegt sind.
"""

from __future__ import annotations

import time
from typing import List, Optional

from .config import Config
from .mapping.macros import MacroDetector
from .mapping.tokens import TokenForge
from .vision.hand_tracker import HandTracker
from .vision.types import EngineSnapshot, HandObservation


class InstrumentEngine:
    def __init__(self, config: Config, seed: Optional[int] = None):
        self.config = config
        self.hands = HandTracker(config.tracks)
        self.forge = TokenForge(config.forge, config.mapping, seed=seed)
        self.macro = MacroDetector(config.mapping)

        self._last_time = time.perf_counter()
        self._fps = 0.0

    # ------------------------------------------------------------------
    def set_performers(self, performers: int) -> None:
        self.config.performers = performers
        self.hands.reset()
        self.forge.clear()

    def shuffle_library(self) -> None:
        self.forge.shuffle_library()

    def place_all(self) -> None:
        self.forge.place_all()

    def clear(self) -> None:
        self.forge.clear()

    # ------------------------------------------------------------------
    def update(
        self, observations: List[HandObservation], dt: Optional[float] = None
    ) -> EngineSnapshot:
        frame_time = self._tick() if dt is None else dt
        tracks = self.hands.update(observations)
        tokens = self.forge.update(tracks, frame_time)
        macros = self.macro.update(self.forge.live_openness)
        # Der Detektor sieht nur die Haende, die gerade einen Loop tunen.
        # Angezeigt werden soll aber, wie viele Haende ueberhaupt da sind.
        macros.active_hands = len(tracks)
        return EngineSnapshot(
            tokens=tokens,
            slots=self.forge.slots(),
            library=self.forge.library_entries(),
            rack=self.forge.rack_entries(),
            hands=self.forge.hands,
            macros=macros,
            fps=round(self._fps, 1),
        )

    # ------------------------------------------------------------------
    def _tick(self) -> float:
        """Frametaktung. Gibt die Bildzeit in Sekunden zurueck.

        Die Grenzen fangen den ersten Frame und kurze Haenger ab, sonst
        wuerde die Wischerkennung ueberreagieren.
        """
        now = time.perf_counter()
        delta = min(max(now - self._last_time, 1.0 / 120.0), 1.0 / 10.0)
        self._last_time = now
        instant = 1.0 / delta
        self._fps = instant if self._fps == 0 else self._fps * 0.9 + instant * 0.1
        return delta
