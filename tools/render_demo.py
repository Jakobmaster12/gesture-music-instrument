#!/usr/bin/env python3
"""Rendert eine kurze Demo der internen Sound Engine als WAV.

Damit hoert ihr, wie das Instrument klingt, ohne Kamera und ohne
Soundkarte. Praktisch fuer die Praesentation.

Aufruf:  python tools/render_demo.py demo.wav
"""

from __future__ import annotations

import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.engine import AudioEngine  # noqa: E402
from src.config import AudioConfig  # noqa: E402

# (Sekunde, [(Slot, Preset, gain, cutoff, density)]) als kleine Choreografie.
# Jeder Slot ist ein abgelegter Loop, so wie ihn eine Hand hinterlassen wuerde.
SCRIPT = [
    (0.0, [("s1", "kick_four", 0.90, 0.55, 0.35)]),
    (2.0, [("s2", "hats_closed", 0.60, 0.95, 0.55)]),
    (4.0, [("s3", "bass_sub", 0.85, 0.40, 0.50)]),
    (6.0, [("s4", "clap", 0.80, 0.75, 0.30)]),
    (8.0, [("s5", "chord_stab", 0.55, 0.60, 0.35)]),
    (10.0, [("s6", "lead_pluck", 0.60, 0.85, 0.60)]),
    (12.0, [
        ("s1", "kick_four", 0.95, 0.70, 1.00),
        ("s2", "hats_closed", 0.70, 1.00, 0.95),
        ("s7", "bass_acid", 0.65, 0.80, 0.70),
    ]),
]


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "demo.wav"
    seconds = 16.0
    config = AudioConfig(samplerate=44100, blocksize=512, bpm=128)
    engine = AudioEngine(config)

    chunks = []
    position = 0.0
    step = 0.25
    while position < seconds:
        for at, changes in SCRIPT:
            if abs(position - at) < step / 2:
                for key, preset_id, gain, cutoff, density in changes:
                    engine.set_slot(key, preset_id, gain, cutoff, density)
        chunks.append(engine.render_offline(step))
        position += step

    audio = np.concatenate(chunks, axis=0)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")

    with wave.open(target, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(config.samplerate)
        handle.writeframes(pcm.tobytes())

    print(f"Geschrieben: {os.path.abspath(target)}  ({len(audio) / config.samplerate:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
