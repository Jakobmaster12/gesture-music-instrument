"""Klangerzeugung fuer die interne Sound Engine.

Alle Sounds werden beim Start einmal als kurze Buffer berechnet und
spaeter nur noch abgespielt und gefiltert. Das ist billig genug, um
nebenbei zum Hand Tracking zu laufen, und braucht keine Samples auf der
Festplatte.
"""

from __future__ import annotations

import os
import wave
from typing import Dict, List

import numpy as np

from ..config import PROJECT_ROOT

# A Moll, Grundtoene der vier Takte
BAR_ROOTS_HZ = (55.00, 43.65, 32.70, 49.00)          # A1, F1, C1, G1
PENTATONIC_SEMITONES = (0, 3, 5, 7, 10, 12, 15, 17)  # A Moll Pentatonik
CHORD_SEMITONES = ((0, 3, 7), (0, 4, 7), (0, 4, 7), (0, 3, 7))


def _t(samples: int, sr: int) -> np.ndarray:
    return np.arange(samples, dtype=np.float32) / sr


def _decay(samples: int, sr: int, seconds: float, curve: float = 5.0) -> np.ndarray:
    return np.exp(-curve * _t(samples, sr) / max(seconds, 1e-4)).astype(np.float32)


def _saw(freq: float, samples: int, sr: int, detune: float = 0.0) -> np.ndarray:
    t = _t(samples, sr)
    phase = (t * freq * (1.0 + detune)) % 1.0
    return (2.0 * phase - 1.0).astype(np.float32)


def make_kick(sr: int, tight: bool = False) -> np.ndarray:
    length = 0.26 if tight else 0.45
    n = int(sr * length)
    t = _t(n, sr)
    drop = 40.0 if tight else 28.0
    pitch = (52.0 if tight else 45.0) + 95.0 * np.exp(-drop * t)
    phase = np.cumsum(pitch / sr).astype(np.float32)
    body = np.sin(2 * np.pi * phase) * _decay(n, sr, length * 0.9, curve=4.0)
    click = np.random.default_rng(1).normal(0, 1, n).astype(np.float32) * _decay(n, sr, 0.01, curve=9.0)
    return (body * 0.95 + click * (0.28 if tight else 0.15)).astype(np.float32)


def make_snare(sr: int) -> np.ndarray:
    n = int(sr * 0.28)
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 1, n).astype(np.float32) * _decay(n, sr, 0.18, curve=6.0)
    tone = np.sin(2 * np.pi * 185.0 * _t(n, sr)) * _decay(n, sr, 0.12, curve=7.0)
    return (noise * 0.75 + tone * 0.45).astype(np.float32)


def make_hat(sr: int, length: float = 0.055, seed: int = 3) -> np.ndarray:
    n = int(sr * length)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n).astype(np.float32)
    # grobe Hochpassfilterung durch Differenzieren
    noise = np.diff(noise, prepend=np.float32(0.0))
    return (noise * _decay(n, sr, length, curve=6.0) * 0.6).astype(np.float32)


def make_clap(sr: int) -> np.ndarray:
    """Drei kurze Rauschstoesse plus Nachhall, das macht den Klatscher."""
    n = int(sr * 0.30)
    rng = np.random.default_rng(11)
    noise = rng.normal(0, 1, n).astype(np.float32)
    noise = np.diff(noise, prepend=np.float32(0.0))
    out = np.zeros(n, dtype=np.float32)
    for offset, amp in ((0.000, 1.0), (0.011, 0.85), (0.022, 0.7)):
        start = int(offset * sr)
        burst = noise[: n - start] * _decay(n - start, sr, 0.012, curve=8.0)
        out[start:] += burst * amp
    tail = noise * _decay(n, sr, 0.20, curve=6.0) * 0.30
    return ((out + tail) * 0.55).astype(np.float32)


def make_rim(sr: int) -> np.ndarray:
    n = int(sr * 0.09)
    rng = np.random.default_rng(12)
    click = rng.normal(0, 1, n).astype(np.float32) * _decay(n, sr, 0.008, curve=9.0)
    tone = np.sin(2 * np.pi * 1750.0 * _t(n, sr)) * _decay(n, sr, 0.05, curve=8.0)
    return ((click * 0.4 + tone * 0.7) * 0.55).astype(np.float32)


def make_tom(freq: float, sr: int) -> np.ndarray:
    n = int(sr * 0.34)
    t = _t(n, sr)
    pitch = freq * (1.0 + 0.45 * np.exp(-14.0 * t))
    phase = np.cumsum(pitch / sr).astype(np.float32)
    body = np.sin(2 * np.pi * phase) * _decay(n, sr, 0.30, curve=4.5)
    return (body * 0.8).astype(np.float32)


def make_sub(freq: float, sr: int) -> np.ndarray:
    """808 artiger Sinusbass mit leichtem Pitch Abfall."""
    n = int(sr * 0.55)
    t = _t(n, sr)
    pitch = freq * (1.0 + 0.25 * np.exp(-22.0 * t))
    phase = np.cumsum(pitch / sr).astype(np.float32)
    env = _decay(n, sr, 0.50, curve=3.0)
    attack = np.minimum(t / 0.006, 1.0).astype(np.float32)
    return (np.sin(2 * np.pi * phase) * env * attack * 0.95).astype(np.float32)


def make_acid(freq: float, sr: int) -> np.ndarray:
    """Kurzer Saegezahn mit Filtersweep, angenaehert durch zwei Huellkurven."""
    n = int(sr * 0.22)
    bright = _saw(freq * 2.0, n, sr, 0.006) * _decay(n, sr, 0.05, curve=7.0)
    dark = _saw(freq, n, sr) * _decay(n, sr, 0.18, curve=4.0)
    attack = np.minimum(_t(n, sr) / 0.003, 1.0).astype(np.float32)
    return ((dark * 0.6 + bright * 0.5) * attack * 0.8).astype(np.float32)


def make_stab(root: float, intervals, sr: int) -> np.ndarray:
    n = int(sr * 0.34)
    out = np.zeros(n, dtype=np.float32)
    for semi in intervals:
        freq = root * (2.0 ** (semi / 12.0))
        out += _saw(freq, n, sr, 0.004) * 0.28
        out += _saw(freq, n, sr, -0.004) * 0.28
    env = _decay(n, sr, 0.24, curve=5.5)
    attack = np.minimum(_t(n, sr) / 0.004, 1.0).astype(np.float32)
    return (out * env * attack * 0.5).astype(np.float32)


def make_sweep(sr: int) -> np.ndarray:
    """Rauschen, das ueber zwei Sekunden anschwillt. Build-up Effekt."""
    n = int(sr * 2.0)
    rng = np.random.default_rng(13)
    noise = rng.normal(0, 1, n).astype(np.float32)
    noise = np.diff(noise, prepend=np.float32(0.0))
    rise = (np.linspace(0.0, 1.0, n, dtype=np.float32) ** 2.2)
    release = np.minimum((n - np.arange(n)) / (0.08 * sr), 1.0).astype(np.float32)
    return (noise * rise * release * 0.45).astype(np.float32)


def make_blip(freq: float, sr: int) -> np.ndarray:
    n = int(sr * 0.10)
    tone = np.sin(2 * np.pi * freq * _t(n, sr))
    env = _decay(n, sr, 0.07, curve=7.0)
    attack = np.minimum(_t(n, sr) / 0.002, 1.0).astype(np.float32)
    return (tone * env * attack * 0.55).astype(np.float32)


def make_pluck(freq: float, sr: int) -> np.ndarray:
    n = int(sr * 0.30)
    wave = (
        _saw(freq, n, sr) * 0.5
        + _saw(freq, n, sr, 0.008) * 0.3
        + np.sin(2 * np.pi * freq * 2 * _t(n, sr)) * 0.2
    )
    env = _decay(n, sr, 0.22, curve=5.0)
    attack = np.minimum(_t(n, sr) / 0.003, 1.0).astype(np.float32)
    return (wave * env * attack * 0.8).astype(np.float32)


def make_pad(root: float, intervals, sr: int) -> np.ndarray:
    n = int(sr * 2.0)
    out = np.zeros(n, dtype=np.float32)
    for semi in intervals:
        freq = root * (2.0 ** (semi / 12.0))
        out += _saw(freq, n, sr, 0.003) * 0.3
        out += _saw(freq, n, sr, -0.003) * 0.3
    attack = np.minimum(_t(n, sr) / 0.25, 1.0).astype(np.float32)
    release = np.minimum((n - np.arange(n)) / (0.4 * sr), 1.0).astype(np.float32)
    return (out * attack * release * 0.35).astype(np.float32)


# Zielpegel pro Klang. Rauschbasierte Buffer haben von Natur aus hohe
# Spitzen, ohne diese Stufe waeren Hats und Clap viel zu laut im Verhaeltnis.
BANK_PEAKS: Dict[str, float] = {
    "kick": 0.95, "kick_tight": 0.95, "clap": 0.60, "snare": 0.75,
    "hats": 0.45, "hats_tight": 0.40, "rim": 0.50, "tom": 0.80,
    "sub": 0.95, "acid": 0.70, "lead": 0.70, "arp": 0.60,
    "stab": 0.60, "pad": 0.50, "sweep": 0.60, "blip": 0.50,
}


def normalise(buffer: np.ndarray, peak: float) -> np.ndarray:
    top = float(np.max(np.abs(buffer)))
    if top < 1e-6:
        return buffer
    return (buffer * (peak / top)).astype(np.float32)


def build_sample_bank(sr: int) -> Dict[str, List[np.ndarray]]:
    """Erzeugt alle Buffer einmal beim Start. Index = Tonhoehe bzw. Variante."""
    sub_notes: List[np.ndarray] = []
    for root in BAR_ROOTS_HZ:
        sub_notes.append(make_sub(root, sr))
        sub_notes.append(make_sub(root * 2.0, sr))

    acid_notes = [
        make_acid(110.0 * (2.0 ** (semi / 12.0)), sr) for semi in PENTATONIC_SEMITONES
    ]
    lead_notes = [
        make_pluck(440.0 * (2.0 ** (semi / 12.0)), sr) for semi in PENTATONIC_SEMITONES
    ]
    arp_notes = [
        make_pluck(880.0 * (2.0 ** (semi / 12.0)), sr) for semi in PENTATONIC_SEMITONES
    ]
    blip_notes = [
        make_blip(1320.0 * (2.0 ** (semi / 12.0)), sr) for semi in PENTATONIC_SEMITONES
    ]
    pads = [
        make_pad(root * 4.0, intervals, sr)
        for root, intervals in zip(BAR_ROOTS_HZ, CHORD_SEMITONES)
    ]
    stabs = [
        make_stab(root * 4.0, intervals, sr)
        for root, intervals in zip(BAR_ROOTS_HZ, CHORD_SEMITONES)
    ]

    bank = {
        "kick": [make_kick(sr)],
        "kick_tight": [make_kick(sr, tight=True)],
        "clap": [make_clap(sr)],
        "snare": [make_snare(sr)],
        "hats": [make_hat(sr), make_hat(sr, length=0.16, seed=4)],
        "hats_tight": [make_hat(sr, length=0.030, seed=5)],
        "rim": [make_rim(sr)],
        "tom": [make_tom(freq, sr) for freq in (110.0, 155.0, 196.0)],
        "sub": sub_notes,
        "acid": acid_notes,
        "lead": lead_notes,
        "arp": arp_notes,
        "stab": stabs,
        "pad": pads,
        "sweep": [make_sweep(sr)],
        "blip": blip_notes,
    }
    return {
        name: [normalise(buffer, BANK_PEAKS.get(name, 0.7)) for buffer in samples]
        for name, samples in bank.items()
    }


def load_wav_mono(path: str, target_sr: int) -> np.ndarray:
    """Laedt eine WAV Datei (PCM 8/16/32 bit) als Mono Buffer.

    Bewusst ohne zusaetzliche Bibliothek: das eingebaute `wave` Modul
    reicht fuer normale Aufnahmen und Sample Packs. Mehrkanalige Dateien
    werden auf Mono gemischt, eine abweichende Samplerate wird per
    linearer Interpolation auf `target_sr` gebracht.
    """
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        framerate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"nicht unterstuetzte Bittiefe: {width * 8} bit")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if framerate != target_sr and len(data) > 1:
        duration = len(data) / framerate
        n_out = max(1, int(round(duration * target_sr)))
        x_old = np.linspace(0.0, duration, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
        data = np.interp(x_new, x_old, data)
    return data.astype(np.float32)


def apply_custom_samples(
    bank: Dict[str, List[np.ndarray]], sr: int
) -> Dict[str, List[np.ndarray]]:
    """Ersetzt einzelne Bank Eintraege durch eigene WAV Dateien.

    Was geladen wird, steht in `src/audio/sample_files.py` - dort traegt
    man den Bank Namen und die Pfade zu den WAV Dateien ein. Diese
    Funktion laeuft einmal beim Start der Sound Engine, alles andere
    (Lautstaerke, Filter, Pattern) bleibt unveraendert, weil es an der
    Bank haengt, nicht an den einzelnen Buffern.
    """
    from .sample_files import CUSTOM_SAMPLES  # spaeter Import: keine Zirkelabhaengigkeit

    for name, paths in CUSTOM_SAMPLES.items():
        loaded: List[np.ndarray] = []
        for rel_path in paths:
            full_path = os.path.join(PROJECT_ROOT, rel_path)
            try:
                buffer = load_wav_mono(full_path, sr)
            except (FileNotFoundError, wave.Error, ValueError) as exc:
                print(f"[audio] eigener Sound '{rel_path}' konnte nicht geladen werden: {exc}")
                continue
            loaded.append(normalise(buffer, BANK_PEAKS.get(name, 0.7)))
        if loaded:
            bank[name] = loaded
    return bank


class OnePoleLowpass:
    """Ein Pol Tiefpass pro Stem.

    Die Rekursion y[n] = a*x[n] + (1-a)*y[n-1] wird blockweise als Faltung
    gerechnet. Das Ergebnis ist mathematisch identisch zur Schleife, laeuft
    aber komplett in NumPy und haelt damit den Audio Callback schnell genug.
    """

    def __init__(self, samplerate: int):
        self.sr = samplerate
        self.state = 0.0
        self._alpha = None
        self._kernel = None
        self._tail = None

    def _kernels(self, alpha: float, n: int):
        if self._alpha != alpha or self._kernel is None or len(self._kernel) != n:
            idx = np.arange(n, dtype=np.float64)
            base = max(1.0 - alpha, 1e-12)
            self._kernel = np.power(base, idx).astype(np.float32)
            self._tail = np.power(base, idx + 1.0).astype(np.float32)
            self._alpha = alpha
        return self._kernel, self._tail

    def process(self, block: np.ndarray, cutoff_hz: float) -> np.ndarray:
        n = block.shape[0]
        if n == 0:
            return block
        cutoff_hz = float(min(max(cutoff_hz, 30.0), self.sr * 0.45))
        alpha = 1.0 - float(np.exp(-2.0 * np.pi * cutoff_hz / self.sr))
        kernel, tail = self._kernels(alpha, n)

        conv = np.convolve(block.astype(np.float32), kernel)[:n]
        out = (alpha * conv + tail * np.float32(self.state)).astype(np.float32)
        self.state = float(out[-1])
        return out


def cutoff_to_hz(value: float, low: float = 180.0, high: float = 12000.0) -> float:
    """Mappt 0..1 logarithmisch auf eine Filterfrequenz."""
    value = min(max(value, 0.0), 1.0)
    return float(low * (high / low) ** value)
