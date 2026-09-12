"""Geometrie der beiden Seitenspalten: eine Quelle fuer Schmiede und Anzeige.

Links steht die Bibliothek, rechts das Regal. Beide sind Spalten aus
gleich hohen Faechern an festen Plaetzen - kein Scrollmenu, keine Liste,
die sich verschiebt, sobald etwas dazukommt. Genau das war vorher das
Problem: die abgelegten Loops haben sich in dieselbe Spalte gedraengt,
alles wurde kleiner und ist bei jeder Ablage verrutscht.

Damit eine Hand exakt das Fach trifft, das die Anzeige zeichnet, teilen
sich Schmiede (Auswahl) und Anzeige (Zeichnung) dieselbe Formel - sonst
laufen Auswahl und Bild auseinander.

`Column.pick` ist der zweite Teil der Antwort auf "Auswahl ist muehsam":
die Hand muss ueber die Mitte des Nachbarfachs hinaus, bevor die Auswahl
umspringt. Ohne diese Sperre flackert die Auswahl an jeder Fachgrenze,
und genau dort steht die Hand am haeufigsten.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    """Eine Spalte aus `slots` gleich hohen Faechern, normalisiert 0..1."""

    x: float
    top: float
    bottom: float
    slots: int

    @classmethod
    def packed(cls, x: float, top: float, bottom: float, slots: int,
               max_pitch: float = 0.16) -> "Column":
        """Spalte, die bei wenigen Faechern nicht auseinanderfaellt.

        Acht Beats fuellen die Spalte von selbst. Bei zwei eigenen Loops
        waere ein Fach sonst ein Drittel bildschirmhoch - gross genug
        zum Treffen, aber es sieht aus, als fehle etwas. Deshalb wird
        die Spalte in dem Fall kuerzer und rueckt in die Mitte.
        """
        slots = max(slots, 1)
        span = min(bottom - top, max_pitch * slots)
        centre = (top + bottom) * 0.5
        return cls(x, centre - span * 0.5, centre + span * 0.5, slots)

    @property
    def span(self) -> float:
        return max(self.bottom - self.top, 1e-6)

    @property
    def pitch(self) -> float:
        """Abstand zweier Fachmitten."""
        return self.span / max(self.slots, 1)

    def slot_y(self, index: int) -> float:
        """Mitte des `index`-ten Fachs."""
        return self.top + self.pitch * (index + 0.5)

    def position(self, y: float) -> float:
        """Stufenlose Fachposition einer Hand auf Hoehe `y`."""
        return (y - self.top) / self.pitch - 0.5

    def nearest(self, y: float) -> int:
        if self.slots <= 1:
            return 0
        return max(0, min(self.slots - 1, int(round(self.position(y)))))

    def pick(self, y: float, current: int, hysteresis: float = 0.35) -> int:
        """Auswahl mit Sperre: erst deutlich im Nachbarfach springt sie um.

        `current` ist die bisherige Auswahl, `hysteresis` der zusaetzliche
        Weg in Fachhoehen, den die Hand darueber hinaus zuruecklegen muss.
        """
        if self.slots <= 1:
            return 0
        current = max(0, min(self.slots - 1, current))
        position = self.position(y)
        if position > current + 0.5 + hysteresis:
            return min(self.slots - 1, int(round(position)))
        if position < current - 0.5 - hysteresis:
            return max(0, int(round(position)))
        return current


def library_slot_y(index: int, count: int, top: float, bottom: float) -> float:
    """Normalisierte Y Position des `index`-ten von `count` Faechern."""
    return Column(0.0, top, bottom, count).slot_y(index)


def nearest_library_index(y: float, count: int, top: float, bottom: float) -> int:
    """Welches Fach liegt einer Hand auf Hoehe `y` am naechsten."""
    if count <= 0:
        return 0
    return Column(0.0, top, bottom, count).nearest(y)
