"""Gestenmetriken: alles, was die Haltung einer Hand verraet.

Das Instrument soll spielbar sein, waehrend die Hand entspannt vor dem
Koerper bleibt. Niemand soll quer durch das Bild wandern, um lauter zu
werden. Deshalb liest dieses Modul ausschliesslich die *Haltung* der
Hand aus und nie ihre Position:

    finger_count   wie viele Finger stehen (0..4)    -> Dichte des Beats
    lean           Neigung der Hand nach aussen      -> Lautstaerke
    thumb_open     wie weit der Daumen absteht       -> Klangfarbe
    palm_facing    Handflaeche oder Handruecken      -> Halten (Freeze)
    pinch          Daumen und Zeigefinger beruehren  -> Greifen und Ablegen
    openness       Faust bis weit offene Hand        -> Gruppen Macros
    grip_point     wo die Finger zugreifen           -> Zielen in den Spalten

Alle Werte sind auf die Handgroesse normiert. Dadurch ist es egal, wie
weit jemand von der Kamera weg steht.

Zur Spiegelung: `palm_facing` und `lean` rechnen die Haendigkeit heraus.
MediaPipe vergibt das Label so, als waere das Bild *nicht* gespiegelt.
Spiegelt man das Bild, kippen Label und Geometrie gemeinsam - das
Produkt aus beiden bleibt also gleich. Beide Werte stimmen deshalb mit
und ohne Selfie Ansicht.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from ..config import MappingConfig
from ..vision.types import (
    FINGER_CHAINS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    THUMB_TIP,
    WRIST,
    HandObservation,
)

# Flaeche des Handtellerdreiecks bei voll zur Kamera gedrehter Hand,
# gemessen in Handgroessen zum Quadrat. Dient als Bezug fuer palm_facing.
PALM_REFERENCE = 0.62


def clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def clamp_signed(value: float) -> float:
    return float(min(1.0, max(-1.0, value)))


def _distance(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a][:2] - points[b][:2]))


def _hand_sign(obs: HandObservation) -> float:
    """+1 fuer eine rechte Hand im ungespiegelten Bild, sonst -1."""
    return 1.0 if obs.handedness.startswith("r") else -1.0


# ----------------------------------------------------------------------
# Finger
# ----------------------------------------------------------------------
def finger_curl(obs: HandObservation, chain) -> float:
    """0 = Finger gestreckt, 1 = Finger komplett eingeklappt."""
    mcp, _pip, tip = chain
    base = _distance(obs.landmarks, WRIST, mcp)
    if base < 1e-6:
        return 0.0
    ratio = _distance(obs.landmarks, WRIST, tip) / base
    # Gestreckt liegt bei ca. 2.4, geschlossene Faust bei ca. 1.1
    return clamp01((2.35 - ratio) / 1.15)


def curl_values(obs: HandObservation) -> List[float]:
    return [finger_curl(obs, chain) for chain in FINGER_CHAINS]


def extended_fingers(obs: HandObservation, config: MappingConfig) -> List[bool]:
    """Zeigefinger, Mittelfinger, Ringfinger, kleiner Finger - je gestreckt?"""
    return [curl <= config.finger_extend_max for curl in curl_values(obs)]


def finger_count(obs: HandObservation, config: MappingConfig) -> int:
    """Anzahl gestreckter Finger ohne Daumen, 0..4.

    Der Daumen zaehlt bewusst nicht mit: er steuert eigenstaendig die
    Klangfarbe und soll die Zaehlgeste nicht verfaelschen.
    """
    return sum(1 for flag in extended_fingers(obs, config) if flag)


def is_fist(obs: HandObservation, config: MappingConfig) -> bool:
    return finger_count(obs, config) == 0


def density_from_count(count: int) -> float:
    """Fingerzahl auf die vier Dichtestufen eines Presets abbilden."""
    level = min(3, max(0, count - 1))
    return (level + 0.5) / 4.0


# ----------------------------------------------------------------------
# Haltung
# ----------------------------------------------------------------------
def tilt(obs: HandObservation) -> float:
    """Neigung in der Bildebene, -1 = ganz nach links, +1 = ganz nach rechts.

    0 bedeutet: die Finger zeigen nach oben. Gemessen wird die Achse von
    der Handwurzel zum Grundgelenk des Mittelfingers, also der Handruecken
    selbst - das bleibt auch dann stabil, wenn die Finger eingeklappt sind.
    """
    axis = obs.landmarks[MIDDLE_MCP][:2] - obs.landmarks[WRIST][:2]
    if float(np.linalg.norm(axis)) < 1e-6:
        return 0.0
    angle = math.atan2(float(axis[0]), -float(axis[1]))
    return clamp_signed(angle / (math.pi * 0.5))


def lean(obs: HandObservation, config: MappingConfig) -> float:
    """Neigung nach aussen als 0..1. Aufrechte Hand liegt in der Mitte.

    Nach aussen kippen (vom Koerper weg) macht lauter, nach innen leiser.
    Beide Haende fuehlen sich dadurch gleich an.
    """
    span = max(config.lean_degrees, 1.0) / 90.0
    outward = -tilt(obs) * _hand_sign(obs)
    return clamp01(0.5 + 0.5 * (outward / span))


def palm_facing(obs: HandObservation) -> float:
    """+1 = Handflaeche zur Kamera, 0 = Hand steht auf der Kante, -1 = Ruecken.

    Berechnet aus der vorzeichenbehafteten Flaeche des Dreiecks
    Handwurzel / Zeigefingergrundgelenk / Kleinfingergrundgelenk. Dreht
    sich das Handgelenk, klappt das Dreieck um - genau das ist die Geste
    zum Einfrieren.
    """
    points = obs.landmarks
    a = points[INDEX_MCP][:2] - points[WRIST][:2]
    b = points[PINKY_MCP][:2] - points[WRIST][:2]
    cross = float(a[0] * b[1] - a[1] * b[0])
    area = -cross / (obs.scale * obs.scale * PALM_REFERENCE)
    return clamp_signed(area * _hand_sign(obs))


def thumb_open(obs: HandObservation, config: MappingConfig) -> float:
    """Wie weit der Daumen vom Zeigefingergrundgelenk absteht, 0..1."""
    raw = _distance(obs.landmarks, THUMB_TIP, INDEX_MCP) / obs.scale
    span = max(config.thumb_max - config.thumb_min, 1e-6)
    return clamp01((raw - config.thumb_min) / span)


def pinch_value(obs: HandObservation) -> float:
    """Abstand Daumenspitze zu Zeigefingerspitze in Handgroessen."""
    return _distance(obs.landmarks, THUMB_TIP, INDEX_TIP) / obs.scale


def grip_point(obs: HandObservation, config: MappingConfig) -> np.ndarray:
    """Der Punkt, an dem Daumen und Zeigefinger sich treffen, shape (2,).

    Damit zielt man in den beiden Spalten. Getrackt wird sonst die
    Handmitte zwischen Handwurzel und Fingergrundgelenken - die liegt
    rund eine Fachhoehe unter den Fingern, mit denen man sichtbar
    greift. Genau dieser Versatz hat die Auswahl daneben treffen lassen:
    die Hand zeigte auf ein Fach, ausgewaehlt war das darunter.

    Gerechnet wird der Punkt aus der Handachse und nicht aus den
    Fingerspitzen. Beim Zugreifen klappt der Zeigefinger ein; ein
    Zielpunkt an den Spitzen wuerde in genau dem Moment wandern, in dem
    die Auswahl stillstehen muss. Die Achse Handwurzel -> Grundgelenk
    des Mittelfingers bleibt dagegen unberuehrt davon, wie die Finger
    gerade stehen, und dreht sich mit der Hand.
    """
    points = obs.landmarks
    wrist = points[WRIST][:2]
    axis = points[MIDDLE_MCP][:2] - wrist
    return np.asarray(wrist + axis * config.grip_reach, dtype=np.float32)


def is_pinch(obs: HandObservation, config: MappingConfig) -> bool:
    """Daumen und Zeigefinger beruehren sich - und sonst nichts.

    Die zweite Bedingung trennt den Pinch von der Faust: dort liegt der
    Daumen ueber allen Fingern und ist vom Mittelfinger genauso weit weg
    wie vom Zeigefinger. Beim echten Pinch ist der Mittelfinger deutlich
    weiter entfernt.
    """
    index = pinch_value(obs)
    if index > config.pinch_max:
        return False
    middle = _distance(obs.landmarks, THUMB_TIP, MIDDLE_TIP) / obs.scale
    return middle > index * config.pinch_clearance


# ----------------------------------------------------------------------
# Gesamthaltung - Grundlage der Gruppenmacros
# ----------------------------------------------------------------------
def openness(obs: HandObservation, config: MappingConfig) -> float:
    """0 = geschlossene Faust, 1 = alle Finger gestreckt, Daumen abgespreizt.

    Die Gruppenmacros sind damit reine Haltungsgesten: gemeinsam die
    Haende oeffnen zieht den Riser hoch, gemeinsam die Faust ballen
    loest den Drop aus. Niemand muss dafuer die Haende ueber den Kopf
    nehmen - der Arm kann entspannt vor dem Koerper bleiben.

    Die Fingerzahl traegt drei Viertel, der Daumen ein Viertel. So
    reicht eine Hand mit vier gestreckten Fingern allein noch nicht fuer
    den Riser: der Daumen muss bewusst mit abstehen.
    """
    fingers = finger_count(obs, config) / 4.0
    return clamp01(fingers * 0.75 + thumb_open(obs, config) * 0.25)
