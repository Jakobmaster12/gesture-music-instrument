# Eigene Loop Beats

Was hier liegt, steht beim naechsten Start links in der Bibliothek und
ersetzt die eingebauten Beats. Mehr ist nicht noetig - keine Zeile Code,
kein Eintrag in einer Liste.

## Eine Datei, ein Beat

    loops/kick.wav          ->  Beat "KICK"
    loops/toploop.wav       ->  Beat "TOPLOOP"

## Ein Ordner, ein Beat mit vier Dichtestufen (empfohlen)

    loops/bassline/
        1_sparse.wav        ->  1 Finger
        2.wav               ->  2 Finger
        3.wav               ->  3 Finger
        4_full.wav          ->  4 Finger

Die Dateien werden nach Namen sortiert, die erste gehoert zur kleinsten
Dichte. Liegen weniger als vier darin, werden sie gleichmaessig auf die
vier Fingerzahlen verteilt; bei einer einzigen Datei aendert die
Fingerzahl nichts - die Karte in der Hand zeigt das dann auch an.

## Tempo

Die Dateien muessen nicht im Tempo des Instruments liegen. Aus der Laenge
wird die Taktzahl geschaetzt und die Datei exakt auf das Raster gezogen.
Wer es genau angeben will, legt `loops.json` daneben:

    {
      "bassline": {
        "label": "DEEP BASS",
        "short": "BAS",
        "bpm": 140,
        "bars": 2,
        "colour": "#7AA2FF",
        "family": "bass",
        "gain": 0.9
      }
    }

Alle Felder sind freiwillig.

## Pruefen, bevor es losgeht

    python tools/check_loops.py

zeigt fuer jede Datei, als was sie erkannt wurde - Taktzahl, Varianten,
Tempokorrektur. Mit `--bpm 140` fuer ein anderes Zieltempo.

## Format

WAV, PCM 8/16/32 bit, Mono oder Stereo, jede Samplerate. MP3 und AIFF
gehen nicht - vorher in WAV wandeln.

## Zurueck zu den eingebauten Beats

    python main.py --library builtin

Oder `"library": "builtin"` im Abschnitt `audio` der `config.json`.
