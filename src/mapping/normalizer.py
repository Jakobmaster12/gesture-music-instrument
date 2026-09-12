"""Glaettung und Wertaufbereitung.

Rohe Landmarks zittern von Frame zu Frame. Ohne Filter klingt das
Ergebnis nervoes. Ein einfacher exponentieller Filter reicht hier aus
und kostet praktisch keine Rechenzeit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Ema:
    """Exponentieller Glaettungsfilter. alpha 0 = roh, 1 = eingefroren."""

    alpha: float = 0.35
    value: Optional[float] = None

    def update(self, target: float) -> float:
        if self.value is None:
            self.value = float(target)
        else:
            a = min(max(self.alpha, 0.0), 0.999)
            self.value = self.value * a + float(target) * (1.0 - a)
        return self.value

    def reset(self, value: Optional[float] = None) -> None:
        self.value = value


@dataclass
class ChannelSmoother:
    """Haelt einen Filter pro Parameter eines Kanals."""

    alpha: float = 0.35
    filters: Dict[str, Ema] = field(default_factory=dict)
    missing_frames: int = 0

    def smooth(self, name: str, value: float) -> float:
        flt = self.filters.get(name)
        if flt is None:
            flt = Ema(alpha=self.alpha)
            self.filters[name] = flt
        return flt.update(value)

    def decay(self, name: str, target: float = 0.0, speed: float = 0.25) -> float:
        """Faehrt einen Parameter sanft gegen einen Zielwert."""
        flt = self.filters.get(name)
        if flt is None or flt.value is None:
            return target
        flt.value = flt.value + (target - flt.value) * speed
        return flt.value

    def reset(self) -> None:
        for flt in self.filters.values():
            flt.reset(0.0)

    def reset_to(self, volume: float, cutoff: float) -> None:
        """Filter auf feste Werte setzen.

        Wird beim Auftauen gebraucht: sonst wuerde der Loop kurz auf den
        alten Glaettungsstand zurueckspringen, bevor die Hand wieder greift.
        """
        for name, value in (("volume", volume), ("cutoff", cutoff)):
            flt = self.filters.get(name)
            if flt is None:
                flt = Ema(alpha=self.alpha)
                self.filters[name] = flt
            flt.reset(float(value))
