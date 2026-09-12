#!/usr/bin/env python3
"""Zeigt, was das Instrument aus dem Ordner `loops/` macht.

Vor einem Auftritt lohnt sich der Blick: hier steht, welche Datei als
welcher Beat erkannt wurde, ueber wie viele Takte sie laeuft, wie viele
Dichtestufen sie hat und ob sie fuers Zieltempo gedehnt werden musste.

    python tools/check_loops.py
    python tools/check_loops.py --bpm 140
    python tools/check_loops.py --dir /pfad/zu/loops
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio import loops  # noqa: E402
from src.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Eigene Loops pruefen")
    parser.add_argument("--bpm", type=float, help="Zieltempo (sonst aus config.json)")
    parser.add_argument("--dir", default=loops.LOOPS_DIR, help="Ordner mit den Loops")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config = Config.load(args.config)
    bpm = args.bpm or config.audio.bpm
    samplerate = config.audio.samplerate

    library = loops.load_library(samplerate, bpm, args.dir)
    print(f"Ordner   {os.path.abspath(args.dir)}")
    print(f"Tempo    {bpm:.1f} BPM, {samplerate} Hz\n")

    for problem in library.problems:
        print(f"  ! uebersprungen - {problem}")
    if library.problems:
        print()

    if not library.presets:
        print("Keine Loops gefunden. Das Instrument nimmt die eingebauten Beats.")
        print("Anleitung: loops/README.md")
        return 0

    header = f"{'BEAT':<16}{'KUERZEL':<9}{'TAKTE':>6}{'STUFEN':>8}{'LAENGE':>9}  FAMILIE"
    print(header)
    print("-" * len(header))
    for item in library.presets:
        clips = library.bank.get(item.bank, [])
        seconds = len(clips[0]) / samplerate if clips else 0.0
        print(
            f"{item.label[:15]:<16}{item.short:<9}{item.bars:>6}"
            f"{item.variants:>8}{seconds:>8.2f}s  {item.family}"
        )

    print(f"\n{len(library.presets)} Beats in der Bibliothek.")
    missing = [item.label for item in library.presets if item.variants < 4]
    if missing:
        print(
            "Weniger als vier Dateien (die Fingerzahl kann dort nur wenig "
            "aendern): " + ", ".join(missing)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
