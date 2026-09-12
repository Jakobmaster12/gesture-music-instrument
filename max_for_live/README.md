# Anbindung an Ableton Live

Es gibt drei Wege, den DAW zu steuern. Der erste braucht gar nichts,
der dritte klingt am professionellsten.

## Weg 1: interne Sound Engine (kein DAW noetig)

Standard. `python main.py` startet und spielt sofort. Gut fuer den Hackathon,
weil nichts schief gehen kann und kein Rechner zusaetzlich Ableton braucht.

## Weg 2: OSC nach MIDI (jeder DAW)

```bash
pip install python-rtmidi
python tools/osc_to_midi.py          # Terminal 1
python main.py --no-audio            # Terminal 2
```

Das Skript oeffnet einen virtuellen MIDI Ausgang namens `GestureInstrument`.
In Ableton unter Einstellungen, Link/MIDI, den Eingang auf Fernsteuerung
stellen. Danach Cmd+M (Windows: Strg+M) druecken, einen Regler anklicken und
eine Hand bewegen. Fertig gemappt.

MIDI Kanaele:

| Kanal | Instrument |
|---|---|
| 1 | Kick |
| 2 | Bass |
| 3 | Snare |
| 4 | Hi-Hats |
| 5 | Synth Lead |
| 6 | Synth Pad |

CC pro Kanal: 1 = Lautstaerke, 74 = Filter, 2 = Pattern Dichte, 7 = Mute.
CC 11 auf Kanal 1 ist der Riser.

Unter Windows gibt es keine virtuellen MIDI Ports von Haus aus. Dafuer
vorher [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html)
installieren und dort einen Port anlegen.

## Weg 3: Max for Live Device selbst bauen

Eine `.amxd` Datei ist ein Binaerformat und laesst sich nicht sinnvoll als
Text mitliefern. Das Device ist aber in zwei Minuten gebaut:

1. In Ableton eine MIDI Spur anlegen, `Max Audio Effect` daraufziehen, auf
   den Stift klicken. Max oeffnet sich.
2. Alles im Patch loeschen, dann den Inhalt von `GestureOSCReceiver.txt`
   kopieren und in Max mit Cmd+V einfuegen. Max baut daraus den fertigen
   Patch.
3. Speichern als `GestureOSCReceiver.amxd`.
4. Device auf jede der sechs Spuren ziehen. Im Device oben den Slot und die
   Hand einstellen, dann per Map Button auf Lautstaerke und Filter mappen.

Der Patch hoert auf UDP Port 9000 und verteilt die Adressen
`/stem/<instrument>/...` auf Live Parameter. Die Instrumente heissen
`kick`, `bass`, `snare`, `hats`, `synth1` und `synth2`.
