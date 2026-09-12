"""Eigene Loop Beats aus dem Ordner `loops/`.

Hier wird aus WAV Dateien die Bibliothek, die die Spieler links im Bild
greifen. Es gibt genau eine Stelle zum Eintragen: den Ordner. Keine
Liste im Code, keine Konfiguration - wer eine Datei hineinlegt, hat beim
naechsten Start einen neuen Beat.

Ablage
------

    loops/
        kick.wav                 ein Loop, eine Datei
        bassline/                ein Loop mit vier Dichtestufen
            1_sparse.wav         1 Finger
            2.wav                2 Finger
            3.wav                3 Finger
            4_full.wav           4 Finger
        loops.json               optional, nur fuer Feinheiten

Ein Ordner ist der bessere Weg: die vier Dateien darin sind dieselbe
Spur in vier Dichten, und die Fingerzahl waehlt zwischen ihnen. Liegen
weniger als vier Dateien darin, werden sie gleichmaessig auf die vier
Stufen verteilt; bei einer einzigen Datei bleibt der Loop unter allen
Fingerzahlen derselbe.

Tempo
-----

Die Dateien muessen nicht im Tempo des Instruments liegen. Aus der Laenge
wird die Anzahl Takte geschaetzt und die Datei danach exakt auf das
Raster gezogen - ein Loop mit 140 BPM laeuft danach sauber mit den
anderen auf 128. Wer es genau wissen will, traegt Tempo und Taktzahl in
`loops.json` ein:

    {
      "bassline": { "label": "DEEP BASS", "short": "BAS",
                    "bpm": 140, "bars": 2, "colour": "#7AA2FF" }
    }

Alle Felder sind freiwillig. `bpm` sagt, in welchem Tempo die Datei
aufgenommen wurde, `bars` wie viele Takte sie umfasst.
"""

from __future__ import annotations

import json
import os
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import PROJECT_ROOT
from ..mapping.presets import LOOP_COLOURS, Preset
from .synth import load_wav_mono, normalise

LOOPS_DIR = os.path.join(PROJECT_ROOT, "loops")
META_FILE = "loops.json"

# Ohne Angabe wird die Taktzahl aus der Laenge geschaetzt und auf einen
# dieser Werte gerundet. Alles dazwischen waere fast immer ein Messfehler.
BAR_CHOICES = (1, 2, 4, 8)
LOOP_PEAK = 0.88
FADE_SECONDS = 0.004   # kurze Blende an beiden Enden gegen Knackser

# Nach diesen Woertern im Namen bekommt ein Loop seine Familie. Die
# Familie steuert nur die Sortierung in der Bibliothek.
FAMILY_HINTS = (
    ("drums", ("kick", "drum", "beat", "clap", "snare", "hat", "perc", "top")),
    ("bass", ("bass", "sub", "808", "acid")),
    ("lead", ("lead", "melo", "arp", "pluck", "synth")),
    ("chord", ("chord", "stab", "pad", "key")),
    ("fx", ("fx", "sweep", "riser", "noise", "atmo", "vox")),
)
FAMILY_ORDER = {"drums": 0, "bass": 1, "chord": 2, "lead": 3, "fx": 4, "other": 5}


@dataclass(frozen=True)
class LoopLibrary:
    """Was der Ordner hergibt: Presets fuer die Anzeige, Buffer fuer den Ton."""

    presets: List[Preset]
    bank: Dict[str, List[np.ndarray]]
    problems: List[str]

    def __bool__(self) -> bool:
        return bool(self.presets)


_CACHE: Dict[Tuple[str, int, float], LoopLibrary] = {}


# ----------------------------------------------------------------------
def load_library(
    samplerate: int, bpm: float, directory: Optional[str] = None
) -> LoopLibrary:
    """Liest den Loop Ordner ein. Das Ergebnis wird zwischengespeichert,
    damit Anzeige und Sound Engine dieselben Buffer benutzen."""
    directory = directory or LOOPS_DIR
    key = (os.path.abspath(directory), int(samplerate), round(float(bpm), 4))
    cached = _CACHE.get(key)
    if cached is None:
        cached = _scan(directory, int(samplerate), float(bpm))
        _CACHE[key] = cached
    return cached


def clear_cache() -> None:
    _CACHE.clear()


# ----------------------------------------------------------------------
def _scan(directory: str, samplerate: int, bpm: float) -> LoopLibrary:
    presets: List[Preset] = []
    bank: Dict[str, List[np.ndarray]] = {}
    problems: List[str] = []

    if not os.path.isdir(directory):
        return LoopLibrary(presets, bank, problems)

    meta = _read_meta(directory, problems)
    sources = _collect_sources(directory)

    for order, (name, paths) in enumerate(sources):
        entry = meta.get(name, {})
        clips: List[np.ndarray] = []
        bars = 0
        for path in paths[:4]:
            try:
                clip, clip_bars = _load_clip(path, samplerate, bpm, entry)
            except (FileNotFoundError, wave.Error, ValueError, OSError) as exc:
                problems.append(f"{os.path.relpath(path, PROJECT_ROOT)}: {exc}")
                continue
            clips.append(clip)
            bars = max(bars, clip_bars)
        if not clips:
            continue
        # Alle Varianten eines Loops muessen exakt gleich lang sein, sonst
        # laufen die Dichtestufen beim Umschalten auseinander.
        clips = [_fit_length(clip, len(clips[0])) for clip in clips]
        bank[name] = clips
        presets.append(_make_preset(name, entry, order, bars, len(clips)))

    presets.sort(key=lambda item: (FAMILY_ORDER.get(item.family, 9), item.label))
    return LoopLibrary(presets, bank, problems)


def _read_meta(directory: str, problems: List[str]) -> Dict[str, dict]:
    path = os.path.join(directory, META_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        problems.append(f"{META_FILE}: {exc}")
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def _collect_sources(directory: str) -> List[Tuple[str, List[str]]]:
    """Jede WAV Datei direkt im Ordner ist ein Loop, jeder Unterordner
    ein Loop mit mehreren Dichtestufen."""
    out: List[Tuple[str, List[str]]] = []
    for name in sorted(os.listdir(directory)):
        full = os.path.join(directory, name)
        if os.path.isdir(full):
            files = sorted(
                os.path.join(full, item)
                for item in os.listdir(full)
                if item.lower().endswith(".wav") and not item.startswith(".")
            )
            if files:
                out.append((_slug(name), files))
        elif name.lower().endswith(".wav") and not name.startswith("."):
            out.append((_slug(os.path.splitext(name)[0]), [full]))
    return out


# ----------------------------------------------------------------------
def _load_clip(
    path: str, samplerate: int, bpm: float, entry: dict
) -> Tuple[np.ndarray, int]:
    data = load_wav_mono(path, samplerate)
    if len(data) < 16:
        raise ValueError("Datei ist zu kurz")

    source_bpm = _positive(entry.get("bpm")) or bpm
    bars = int(entry.get("bars") or 0)
    if bars <= 0:
        bars = _guess_bars(len(data) / samplerate, source_bpm)

    target = int(round(bars * 4.0 * 60.0 / bpm * samplerate))
    clip = _resample(data, target)
    clip = _fade_edges(clip, samplerate)
    return normalise(clip, LOOP_PEAK), bars


def _positive(value: Optional[object]) -> Optional[float]:
    """Eine Zahl aus `loops.json`, sofern sie brauchbar ist."""
    try:
        number = float(value)   # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0.0 else None


def _guess_bars(seconds: float, bpm: float) -> int:
    beats = seconds * bpm / 60.0
    raw = max(beats / 4.0, 1e-3)
    return min(BAR_CHOICES, key=lambda bars: abs(bars - raw))


def _resample(data: np.ndarray, target: int) -> np.ndarray:
    """Zieht den Loop auf exakt `target` Samples.

    Lineare Interpolation reicht: die Korrektur ist meist klein, und eine
    saubere Kante zum Takt hin ist wichtiger als das letzte Prozent
    Klangtreue.
    """
    target = max(target, 16)
    if len(data) == target:
        return data.astype(np.float32)
    source = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
    wanted = np.linspace(0.0, 1.0, num=target, endpoint=False)
    return np.interp(wanted, source, data).astype(np.float32)


def _fit_length(clip: np.ndarray, length: int) -> np.ndarray:
    if len(clip) == length:
        return clip
    out = np.zeros(length, dtype=np.float32)
    take = min(len(clip), length)
    out[:take] = clip[:take]
    return out


def _fade_edges(clip: np.ndarray, samplerate: int) -> np.ndarray:
    n = min(int(FADE_SECONDS * samplerate), len(clip) // 8)
    if n < 2:
        return clip
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    clip = clip.copy()
    clip[:n] *= ramp
    clip[-n:] *= ramp[::-1]
    return clip


# ----------------------------------------------------------------------
def _make_preset(
    name: str, entry: dict, order: int, bars: int, variants: int
) -> Preset:
    label = str(entry.get("label") or name.replace("_", " ").upper())
    short = str(entry.get("short") or _short_of(name)).upper()[:4]
    family = str(entry.get("family") or _family_of(name))
    colour = _colour(entry.get("colour"), order)
    gain = float(entry.get("gain") or 1.0)
    return Preset(
        id=name,
        label=label,
        short=short,
        family=family,
        bank=name,
        note_mode="loop",
        # Ein Loop braucht keine Schlagmuster. Die vier vollen Raster
        # stehen nur da, damit alles, was ein Preset anfasst - Anzeige,
        # OSC, Tests - keine Sonderfaelle kennen muss.
        patterns=tuple(tuple(range(16)) for _ in range(4)),
        colour=colour,
        gain=max(0.05, min(gain, 2.0)),
        kind="loop",
        bars=max(bars, 1),
        variants=max(variants, 1),
    )


def _slug(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "loop"


def _short_of(name: str) -> str:
    parts = [part for part in name.split("_") if part]
    if len(parts) >= 2:
        return (parts[0][:2] + parts[1][:1]).upper()
    return name[:3].upper()


def _family_of(name: str) -> str:
    for family, words in FAMILY_HINTS:
        if any(word in name for word in words):
            return family
    return "other"


def _colour(value: Optional[object], order: int) -> Tuple[int, int, int]:
    """Farbangabe aus `loops.json` lesen, sonst aus der Reihe nehmen."""
    if isinstance(value, str) and value.strip():
        text = value.strip().lstrip("#")
        if len(text) == 6:
            try:
                r, g, b = (int(text[i:i + 2], 16) for i in (0, 2, 4))
                return (b, g, r)     # die Anzeige zeichnet in BGR
            except ValueError:
                pass
    if isinstance(value, (list, tuple)) and len(value) == 3:
        r, g, b = (max(0, min(255, int(part))) for part in value)
        return (b, g, r)
    return LOOP_COLOURS[order % len(LOOP_COLOURS)]
