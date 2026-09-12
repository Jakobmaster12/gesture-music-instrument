#!/usr/bin/env python3
"""Schreibt zwei Beispiel Loops in den Ordner `loops/`.

Gedacht zum Ausprobieren: danach einmal `python main.py` starten, und in
der Bibliothek stehen statt der eingebauten Beats die beiden Dateien aus
dem Ordner. So sieht man in einer Minute, wie eigene Loops eingebunden
werden, bevor man die eigenen Dateien hineinlegt.

    python tools/make_example_loops.py          # anlegen
    python tools/make_example_loops.py --clean  # wieder entfernen

Die erzeugten Dateien sind bewusst schlicht (ein Kick Muster in vier
Dichten und eine Basslinie ueber zwei Takte) - sie zeigen die Ablage,
nicht den guten Geschmack.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.loops import LOOPS_DIR  # noqa: E402

SR = 44100
BPM = 128.0
BAR = 4.0 * 60.0 / BPM
NAMES = ("drumloop", "bassline_deep.wav")


def write_wav(path: str, data: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clipped = np.clip(data, -1.0, 1.0)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((clipped * 32000).astype("<i2").tobytes())


def drum_bar(level: int) -> np.ndarray:
    """Ein Takt Bassdrum, je hoeher `level`, desto mehr Schlaege."""
    n = int(BAR * SR)
    out = np.zeros(n)
    hits = ((0, 8), (0, 4, 8, 12), (0, 4, 8, 12, 14), tuple(range(0, 16, 2)))[level]
    step = BAR / 16.0
    for position in hits:
        start = int(position * step * SR)
        length = min(int(0.20 * SR), n - start)
        t = np.arange(length) / SR
        pitch = 48.0 + 95.0 * np.exp(-32.0 * t)
        out[start:start + length] += np.sin(2 * np.pi * pitch * t) * np.exp(-13.0 * t)
    return out * 0.8


def bass_two_bars() -> np.ndarray:
    """Zwei Takte Basslinie, acht Toene."""
    notes = (55.0, 55.0, 73.4, 65.4, 55.0, 82.4, 73.4, 65.4)
    n = int(BAR * 2 * SR)
    out = np.zeros(n)
    seg = n // len(notes)
    for index, freq in enumerate(notes):
        t = np.arange(seg) / SR
        tone = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
        out[index * seg:index * seg + seg] = tone * np.exp(-2.2 * t)
    return out * 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description="Beispiel Loops anlegen")
    parser.add_argument("--clean", action="store_true", help="Beispiele wieder loeschen")
    parser.add_argument("--dir", default=LOOPS_DIR)
    args = parser.parse_args()

    if args.clean:
        for name in NAMES:
            path = os.path.join(args.dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)
        print(f"Beispiele aus {args.dir} entfernt.")
        return 0

    for level in range(4):
        write_wav(os.path.join(args.dir, "drumloop", f"{level + 1}.wav"),
                  drum_bar(level))
    write_wav(os.path.join(args.dir, "bassline_deep.wav"), bass_two_bars())
    print(f"Beispiele in {args.dir} angelegt:")
    print("  drumloop/1..4.wav    ein Takt, vier Dichtestufen")
    print("  bassline_deep.wav    zwei Takte, eine Stufe")
    print("\nPruefen:  python tools/check_loops.py")
    print("Starten:  python main.py")
    print("Entfernen: python tools/make_example_loops.py --clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
