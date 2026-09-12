"""Der Inhalt der Kiste.

Jedes Preset ist ein spielbarer Loop: ein Klang plus vier Dichtestufen
mit je 16 Steps. Aus diesen Bausteinen werden die Tokens gebaut, die die
Spieler greifen, einfrieren, fusionieren und ablegen.

Farben sind BGR, weil die Anzeige mit OpenCV zeichnet. Die acht Toene
sind als Reihe gedacht: gleiche Helligkeit, gedaempfte Saettigung, gut
unterscheidbar. Volle Primaerfarben waeren auf dunklem Grund lauter als
die Musik und wirken auf einem Beamer billig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

Pattern = Tuple[int, ...]

ALL16 = tuple(range(16))


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    short: str
    family: str
    bank: str                       # Schluessel in die Sample Bank
    note_mode: str                  # wie der Sample Index pro Step gewaehlt wird
    patterns: Tuple[Pattern, ...]   # vier Dichtestufen
    colour: Tuple[int, int, int]
    gain: float = 1.0
    # "steps" baut den Beat aus einzelnen Schlaegen auf einem 16tel Raster,
    # "loop" spielt eine fertige Audiodatei ueber `bars` Takte. Beides
    # laeuft sonst durch dieselbe Kette - Lautstaerke, Klangfarbe und
    # Dichte bleiben dieselben Regler.
    kind: str = "steps"
    bars: int = 1                   # Laenge eines Loops in Takten
    variants: int = 1               # so viele Dateien liegen fuer diesen Loop bereit

    def pattern_for(self, density: float) -> Pattern:
        level = min(3, max(0, int(density * 3.999)))
        return self.patterns[level]

    def level_for(self, density: float) -> int:
        """Welche der vier Dichtestufen die Finger gerade meinen."""
        return min(3, max(0, int(density * 3.999)))

    def variant_for(self, density: float) -> int:
        """Welche Datei eines Loops zu dieser Dichte gehoert.

        Liegen vier Varianten bereit, bekommt jede Fingerzahl ihre eigene.
        Bei weniger Dateien werden die Stufen gleichmaessig verteilt, bei
        nur einer Datei bleibt es immer dieselbe.
        """
        count = max(self.variants, 1)
        return min(count - 1, self.level_for(density) * count // 4)

    @property
    def steps_per_loop(self) -> int:
        return 16 * max(self.bars, 1)


BUILTIN_PRESETS: List[Preset] = [
    # ---------------- Drums ----------------
    Preset(
        id="kick_four", label="KICK 4x4", short="KCK", family="drums",
        bank="kick", note_mode="accent",
        patterns=((0, 8), (0, 4, 8, 12), (0, 4, 8, 12, 14), (0, 2, 4, 6, 8, 10, 12, 14)),
        colour=(108, 122, 255),
    ),
    Preset(
        id="clap", label="CLAP", short="CLP", family="drums",
        bank="clap", note_mode="single",
        patterns=((4, 12), (4, 12), (4, 12, 14), (4, 7, 12, 14, 15)),
        colour=(118, 190, 255),
    ),
    Preset(
        id="hats_closed", label="HI-HATS", short="HAT", family="drums",
        bank="hats", note_mode="hat",
        patterns=((0, 4, 8, 12), (0, 2, 4, 6, 8, 10, 12, 14), ALL16, ALL16),
        colour=(196, 224, 123),
    ),
    # ---------------- Bass ----------------
    Preset(
        id="bass_sub", label="SUB BASS", short="SUB", family="bass",
        bank="sub", note_mode="bass",
        patterns=((0, 8), (0, 6, 8, 14), (0, 3, 6, 8, 11, 14),
                  (0, 2, 3, 6, 8, 10, 11, 14)),
        colour=(255, 166, 111),
    ),
    Preset(
        id="bass_acid", label="ACID BASS", short="ACD", family="bass",
        bank="acid", note_mode="lead",
        patterns=((0, 3, 6, 10), (0, 2, 3, 6, 8, 10, 13),
                  (0, 1, 3, 4, 6, 8, 10, 11, 13, 14), ALL16),
        colour=(255, 139, 180),
    ),
    # ---------------- Melodie ----------------
    Preset(
        id="lead_pluck", label="PLUCK LEAD", short="LED", family="lead",
        bank="lead", note_mode="lead",
        patterns=((0, 6, 12), (0, 3, 6, 9, 12), (0, 2, 4, 6, 8, 10, 12, 14), ALL16),
        colour=(122, 224, 140),
    ),
    Preset(
        id="chord_stab", label="CHORD STAB", short="STB", family="chord",
        bank="stab", note_mode="bar",
        patterns=((0,), (0, 8), (0, 6, 8), (0, 3, 6, 8, 11, 14)),
        colour=(208, 143, 255),
    ),
    # ---------------- FX ----------------
    Preset(
        id="sweep", label="NOISE SWEEP", short="SWP", family="fx",
        bank="sweep", note_mode="single",
        patterns=((0,), (0,), (0, 8), (0, 8)),
        colour=(191, 166, 154), gain=0.8,
    ),
]

# Die aktive Bibliothek. Sie wird beim Start ersetzt, sobald im Ordner
# `loops/` eigene Loops liegen - siehe `src/audio/loops.py`. Alle drei
# Behaelter werden dabei an Ort und Stelle gefuellt, damit Module, die
# sie einmal importiert haben (OSC Bruecke, Anzeige), weiter dieselbe
# Liste sehen.
PRESETS: List[Preset] = list(BUILTIN_PRESETS)
PRESET_BY_ID: Dict[str, Preset] = {item.id: item for item in PRESETS}
PRESET_IDS: List[str] = [item.id for item in PRESETS]

WHITE = (245, 245, 245)

# Eine kuratierte Reihe fuer eigene Loops ohne eigene Farbangabe:
# gleiche Helligkeit, gedaempfte Saettigung, gut unterscheidbar (BGR).
LOOP_COLOURS: Tuple[Tuple[int, int, int], ...] = (
    (108, 122, 255), (118, 190, 255), (196, 224, 123), (255, 166, 111),
    (255, 139, 180), (122, 224, 140), (208, 143, 255), (191, 166, 154),
    (150, 205, 255), (205, 190, 130),
)


def use_presets(items: List[Preset]) -> List[Preset]:
    """Setzt die aktive Bibliothek. Muss vor dem Bau der Engine laufen."""
    if not items:
        return PRESETS
    PRESETS[:] = list(items)
    PRESET_BY_ID.clear()
    PRESET_BY_ID.update({item.id: item for item in PRESETS})
    PRESET_IDS[:] = [item.id for item in PRESETS]
    return PRESETS


def reset_presets() -> List[Preset]:
    """Zurueck zu den eingebauten Beats - vor allem fuer Tests."""
    return use_presets(list(BUILTIN_PRESETS))


def preset(preset_id: str) -> Preset:
    return PRESET_BY_ID[preset_id]


def colour_of(preset_id: str) -> Tuple[int, int, int]:
    found = PRESET_BY_ID.get(preset_id)
    return found.colour if found else WHITE


def label_of(preset_id: str) -> str:
    found = PRESET_BY_ID.get(preset_id)
    return found.label if found else preset_id.upper()


def short_of(preset_id: str) -> str:
    found = PRESET_BY_ID.get(preset_id)
    return found.short if found else preset_id[:3].upper()
