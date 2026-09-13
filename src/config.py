"""Zentrale Konfiguration.

Alle Werte koennen ueber eine JSON Datei (config.json) ueberschrieben werden.
Aufruf: Config.load("config.json")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

MAX_PERFORMERS = 3


@dataclass
class CameraConfig:
    device: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 60
    mirror: bool = True  # Selfie Ansicht, Bild wird horizontal gespiegelt


@dataclass
class TrackingConfig:
    max_hands: int = 6
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # Wieviele Frames eine Hand fehlen darf, bevor der Kanal stumm geht.
    hold_frames: int = 8


@dataclass
class MappingConfig:
    """Wie eine Handhaltung in Zahlen uebersetzt wird.

    Absichtlich ohne jede Bildposition: gespielt wird mit der Haltung der
    Hand, nicht mit ihrem Standort. Die Hand darf dabei ruhig vor dem
    Koerper bleiben.
    """

    smoothing: float = 0.35        # 0 = kein Smoothing, 1 = eingefroren

    # Finger zaehlen -> Dichte des Beats
    finger_extend_max: float = 0.45   # bis hierher gilt ein Finger als gestreckt
    finger_hold_frames: int = 3       # so lange muss eine Zahl stabil sein

    # Hand neigen -> Lautstaerke
    lean_degrees: float = 55.0     # so weit kippen fuer den vollen Weg

    # Daumen abspreizen -> Klangfarbe
    thumb_min: float = 0.33        # angelegter Daumen, in Handgroessen
    thumb_max: float = 0.88        # weit abgespreizter Daumen

    # Handgelenk drehen -> Loop halten
    flip_threshold: float = 0.25   # Totzone um die Handkante herum
    flip_frames: int = 3           # so viele Frames muss die Seite stimmen
    flip_arm: float = 0.55         # darueber gilt die Hand als flach zur Kamera

    # Wo die Hand hinzeigt: der Punkt, an dem Daumen und Zeigefinger sich
    # treffen, gemessen in Handgroessen ueber der Handwurzel entlang der
    # Handachse. Damit wird in den Spalten gezielt.
    grip_reach: float = 1.30

    # Daumen und Zeigefinger -> greifen und ablegen
    pinch_max: float = 0.30        # geschlossen, in Handgroessen
    pinch_clearance: float = 1.4   # Mittelfinger muss deutlich weiter weg sein
    # Die enge Schwelle trennt den Pinch von der Faust: dort liegt der
    # Daumen je nach Haltung zufaellig nah am Zeigefinger. Seit der Pinch
    # auch ablegt, waere so eine Verwechslung teuer.

    # Macro Trigger, gemessen an der medianen Handoeffnung (0 = Faust)
    riser_open: float = 0.85       # so weit muessen alle Haende offen sein
    riser_frames: int = 12
    drop_closed: float = 0.15      # so weit muessen alle Faeuste geschlossen sein
    drop_frames: int = 6


@dataclass
class ForgeConfig:
    """Die beiden Seitenspalten, Loops in der Hand, Fusion und Ablage.

    Links die Bibliothek mit den fertigen Beats, rechts das Regal mit den
    selbst gebauten Loops. Beide Spalten haben feste Faecher: ein Fach
    bleibt an seinem Platz, egal wie viel gerade drin liegt. Wie beim
    Papierkorb entscheidet in den Spalten bewusst die Position der Hand,
    nicht ihre Haltung - man sieht das Fach und greift danach.
    """

    # Bibliothek links
    library_x: float = 0.115        # Mitte der Spalte, normalisiert
    library_width: float = 0.185    # Breite eines Fachs
    library_top: float = 0.205
    library_bottom: float = 0.925
    library_capture_x: float = 0.32 # bis hierhin zaehlt eine leere Hand als "an der Spalte"

    # Regal rechts
    rack_x: float = 0.885
    rack_width: float = 0.185
    rack_top: float = 0.205
    rack_bottom: float = 0.775
    rack_capture_x: float = 0.68    # ab hier zaehlt eine leere Hand als "am Regal"
    max_parked: int = 5             # so viele Faecher hat das Regal

    # Auswahl in einer Spalte. Die Glaettung haelt eine ruhende Hand
    # still, darf aber eine bewegte nicht bremsen - sonst zeigt die
    # Spalte noch auf ein Fach, an dem die Finger laengst vorbei sind.
    # Deshalb loest sich beides, sobald die Hand sichtbar wandert.
    select_hysteresis: float = 0.22 # zusaetzlicher Weg in Fachhoehen bis zum Umspringen
    select_smoothing: float = 0.35  # Glaettung der ruhenden Handhoehe, 0 = roh
    select_settle: float = 0.012    # Weg pro Frame, ab dem Glaettung und Sperre fallen

    take_frames: int = 3           # so lange den Pinch halten, dann greift die Hand
    take_cooldown: int = 16        # danach kurz gesperrt

    # Halten
    release_frames: int = 10       # Hand weg: Loop wandert ins Regal

    # Einfrieren durch Drehen des Handgelenks
    freeze_frames: int = 3         # so lange muss der Handruecken zu sehen sein

    # Fusion: Haende zusammenfuehren
    fuse_distance: float = 0.22    # so nah muessen die Haende sich kommen
    fuse_frames: int = 3           # so lange zusammenhalten
    fuse_duration: int = 14        # Laenge der Spiralanimation in Frames
    max_recipes: int = 4           # so viele Beats passen in einen Loop

    # Ablegen durch einen Pinch mit dem Loop in der Hand
    min_hold_frames: int = 10      # vorher laesst sich nichts ablegen
    place_frames: int = 4          # so lange den Pinch halten, dann loslassen
    discard_frames: int = 22       # Pinch so lange halten wirft den Loop weg
    fly_frames: int = 12           # Flug ins Regal
    max_tokens: int = 8

    # Papierkorb unter dem Regal: einen gehaltenen Loop dorthin tragen und
    # kurz halten loescht ihn. Zusammen mit den beiden Spalten die einzige
    # Stelle, an der die Position der Hand zaehlt - das Spielen selbst
    # bleibt komplett positionsfrei.
    trash_x: float = 0.885
    trash_y: float = 0.855
    trash_radius: float = 0.075
    trash_frames: int = 24


@dataclass
class TrackConfig:
    """Wiedererkennung einzelner Haende ueber mehrere Frames."""

    max_match_distance: float = 0.16
    max_missing_frames: int = 12
    velocity_damping: float = 0.55


@dataclass
class OscConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 9000
    send_rate: int = 60            # maximale Sendungen pro Sekunde
    # Nur senden, wenn sich ein Wert um mehr als delta geaendert hat.
    min_delta: float = 0.004


@dataclass
class AudioConfig:
    enabled: bool = True           # interne Sound Engine (kein DAW noetig)
    # Woher die Beats kommen:
    #   "auto"    eigene Loops aus dem Ordner loops/, sonst die eingebauten
    #   "loops"   nur eigene Loops
    #   "builtin" nur die eingebauten Beats
    library: str = "auto"
    samplerate: int = 44100
    blocksize: int = 256
    bpm: float = 128.0
    master_gain: float = 0.80
    device: Any = None             # None = Systemstandard


@dataclass
class UiConfig:
    show_window: bool = True
    fullscreen: bool = False
    window_name: str = "Gesture Music Instrument"
    draw_landmarks: bool = True
    countdown_seconds: float = 3.0


@dataclass
class Config:
    performers: int = 1
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    forge: ForgeConfig = field(default_factory=ForgeConfig)
    tracks: TrackConfig = field(default_factory=TrackConfig)
    osc: OscConfig = field(default_factory=OscConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ui: UiConfig = field(default_factory=UiConfig)

    @property
    def hand_budget(self) -> int:
        """Zwei Haende pro Person."""
        return min(max(self.performers, 1), MAX_PERFORMERS) * 2

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = cls()
        if not path:
            return cfg
        if not os.path.exists(path):
            return cfg
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cfg.merge(data)

    def merge(self, data: Dict[str, Any]) -> "Config":
        sub = {f.name: f.type for f in fields(self)}
        for key, value in data.items():
            if key not in sub:
                continue
            current = getattr(self, key)
            if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if hasattr(current, sub_key):
                        setattr(current, sub_key, sub_value)
            else:
                setattr(self, key, value)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
