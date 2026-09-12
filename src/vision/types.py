"""Datentypen, die zwischen Vision, Mapping und UI geteilt werden."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # nur fuer die Typannotation, sonst gaebe es einen Importzyklus
    from ..mapping.tokens import Token

import numpy as np

# Landmark Indizes der MediaPipe Hand (21 Punkte)
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

FINGER_CHAINS = (
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class HandObservation:
    """Eine erkannte Hand in normalisierten Bildkoordinaten (0..1)."""

    landmarks: np.ndarray          # shape (21, 3)
    handedness: str                # "left" oder "right"
    score: float = 1.0

    @property
    def wrist(self) -> np.ndarray:
        return self.landmarks[WRIST]

    @property
    def center(self) -> np.ndarray:
        """Mittelpunkt aus Handwurzel und Mittelfinger Grundgelenk."""
        return (self.landmarks[WRIST] + self.landmarks[MIDDLE_MCP]) * 0.5

    @property
    def scale(self) -> float:
        """Handgroesse als Referenz fuer distanzbasierte Metriken."""
        diff = self.landmarks[MIDDLE_MCP][:2] - self.landmarks[WRIST][:2]
        return float(max(np.linalg.norm(diff), 1e-6))


@dataclass
class Frame:
    """Ein Kamerabild plus Metadaten."""

    image: np.ndarray
    index: int = 0
    timestamp_ms: int = 0

    @property
    def size(self):
        h, w = self.image.shape[:2]
        return w, h


@dataclass
class SoundSlot:
    """Eine klingende Schicht: ein Preset mit seinen Werten.

    Ein Token mit einem Beat liefert einen Slot, ein fusioniertes Token
    liefert einen Slot pro enthaltenem Beat. Die Sound Engine haelt fuer
    jeden Slot eine eigene Stimme.
    """

    key: str                       # stabil ueber Frames: "<token>:<index>"
    preset_id: str
    gain: float = 0.0
    cutoff: float = 0.5
    density: float = 0.0


@dataclass
class BrowserEntry:
    """Ein Fach in einer der beiden Spalten.

    "preset" ist ein frischer Beat aus der Bibliothek links, "loop" ein
    bereits gebauter Loop, der im Regal rechts liegt und zurueck in die
    Hand kann.
    """

    kind: str
    preset_ids: List[str]
    token_id: Optional[int] = None


@dataclass
class HandView:
    """Was die Anzeige ueber eine Hand wissen muss.

    Die Werte kommen fertig aufbereitet aus der Schmiede, damit die
    Anzeige die Gestenerkennung nicht ein zweites Mal rechnen muss.
    """

    hand_id: int
    x: float
    y: float
    handedness: str = "right"
    holding: bool = False
    token_id: Optional[int] = None
    zone: str = ""                 # "library", "rack" oder leer
    selection: int = 0             # Fach, auf das die Hand gerade zeigt
    take_progress: float = 0.0
    delete_progress: float = 0.0   # nur im Regal: Weg bis zum Loeschen
    fingers: int = 0
    facing: float = 1.0
    lean: float = 0.5
    thumb: float = 0.0
    landmarks: Optional[np.ndarray] = None


@dataclass
class MacroState:
    riser: float = 0.0
    drop: float = 0.0
    group_open: float = 0.0   # mediane Handoeffnung, 0 = Faeuste, 1 = offen
    active_hands: int = 0


@dataclass
class EngineSnapshot:
    """Alles, was OSC, Audio und Anzeige pro Frame brauchen."""

    tokens: List["Token"]
    slots: List[SoundSlot]
    library: List[BrowserEntry]              # linke Spalte, ein Fach pro Beat
    rack: List[Optional[BrowserEntry]]       # rechte Spalte, leere Faecher als None
    hands: List[HandView]
    macros: MacroState
    fps: float = 0.0
