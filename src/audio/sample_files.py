"""Eigene Sounds statt der eingebauten Synthese.

**So fuegt ihr einen eigenen Klang hinzu:**

1. WAV Datei (PCM, jede Samplerate) in den Ordner `samples/` legen, am
   besten in einen Unterordner pro Klang, z.B. `samples/kick/mein_kick.wav`.
2. Hier unten in `CUSTOM_SAMPLES` eintragen, unter welcher Bank sie
   laufen soll (siehe Liste der Bank Namen weiter unten) und mit welchem
   Pfad, relativ zum Projektordner.
3. Beim naechsten Start (`python main.py`) wird die Datei geladen und
   ersetzt automatisch die eingebauten Buffer dieser Bank - Lautstaerke,
   Filter und Pattern eines Presets bleiben dabei unveraendert, denn die
   haengen an der Bank, nicht an der einzelnen Audiodatei.

Mehrere Pfade in einer Liste sind erlaubt, z.B. mehrere Varianten fuer
"hats" oder mehrere Toene fuer "sub" - dieselbe Auswahllogik wie bei den
eingebauten Sounds greift dann auch hier (siehe `pick_sample` in
`src/audio/engine.py`).

**Welche Bank Namen es gibt** (definiert in `src/mapping/presets.py`,
Feld `bank` jedes Presets): kick, clap, hats, sub, acid, lead, stab,
sweep. Ein neues Preset mit einem neuen Bank Namen anlegen geht genauso -
einfach in `PRESETS` in `presets.py` einen neuen Eintrag mit eigenem
`bank` Namen ergaenzen und hier die passende Datei eintragen.

Beispiel:

    CUSTOM_SAMPLES = {
        "kick": ["samples/kick/mein_kick.wav"],
        "clap": ["samples/clap/clap_a.wav", "samples/clap/clap_b.wav"],
    }
"""

from __future__ import annotations

from typing import Dict, List

CUSTOM_SAMPLES: Dict[str, List[str]] = {
    # "kick": ["samples/kick/mein_kick.wav"],
}
