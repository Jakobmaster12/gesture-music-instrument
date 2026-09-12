# Loop Forge

Ein Musikinstrument, das ohne Controller auskommt. Eine normale Webcam
filmt den Raum, erkennt bis zu sechs Haende und macht daraus einen Track,
der Schicht fuer Schicht vor dem Publikum entsteht.

Das Besondere ist nicht das Handtracking, sondern dass gespielt wird wie
auf einem Instrument: **nur die Haltung der Hand zaehlt, nie ihr Ort im
Bild**. Die Hand bleibt entspannt vor dem Koerper, niemand muss quer
durch das Bild wandern.

Mit leerer Hand greift man sich einen Beat direkt aus der **Bibliothek**
links im Bild - eine Spalte aus Faechern, ein Fach pro Beat - und nimmt
ihn mit einem **Pinch** mit. Der Loop
haengt dann an der Hand: **gestreckte Finger** bestimmen, wie dicht der
Beat schlaegt (ein Finger sparsam, vier Finger voll), die **Neigung der
Hand** die Lautstaerke, der **abgespreizte Daumen** die Klangfarbe, die
**Faust** schaltet stumm. Sitzt der Beat, dreht man das **Handgelenk um**
- Handruecken zur Kamera - und alle Werte rasten ein. Der Loop laeuft
weiter, die Haltung ist wieder frei. Zurueckdrehen gibt ihn wieder frei.

Genau dafuer gibt es das Halten: zwei gehaltene Loops, deren Haende
zusammengefuehrt werden, **verschmelzen** sofort zu einem einzigen, der
beide Beats traegt. Ein kurzer **Pinch** legt einen Loop ins **Regal**
rechts im Bild, wo er weiterspielt. Das Regal ist die zweite Spalte,
spiegelbildlich zur Bibliothek: eine leere Hand auf Hoehe eines Fachs
holt den Loop mit einem kurzen Pinch zurueck, ein langer Pinch loescht
ihn. Denselben langen Pinch mit einem Loop in der Hand: er ist weg.

Die Beats in der Bibliothek sind entweder die acht eingebauten oder
**eure eigenen Loops**: WAV Dateien in den Ordner `loops/` legen genuegt,
mehr ist nicht noetig (Abschnitt 4).

Das Ergebnis ist ein Arrangement, das im Bild sichtbar liegenbleibt, und
ein Instrument, bei dem zwei Leute ihre Loops zu einem verschmelzen
koennen - auch zwei verschiedene Personen.

Das Projekt laeuft in zwei Betriebsarten:

* **Standalone**: eingebaute Sound Engine, es wird sofort Musik gespielt.
  Kein Ableton, keine Lizenz, keine Vorbereitung.
* **DAW**: alle Werte gehen zusaetzlich per OSC raus und koennen in Ableton,
  Bitwig, Reaper oder FL Studio gemappt werden.

---

## 1. Was ihr selbst installieren muesst

| Was | Warum | Woher |
|---|---|---|
| Python 3.10 bis 3.13 | Laufzeit | python.org oder Homebrew |
| Webcam | Eingabe | eingebaut oder USB |
| Kopfhoerer oder PA | Ausgabe | vorhanden |

Alles andere zieht sich das Projekt selbst: die Python Pakete ueber
`requirements.txt` und das Hand Modell von MediaPipe beim ersten Start
(ca. 8 MB, einmalig, dafuer braucht der Rechner kurz Internet).

Optional, nur fuer den DAW Weg:

* Ableton Live (oder ein anderer DAW)
* `python-rtmidi` fuer die OSC nach MIDI Bruecke
* unter Windows zusaetzlich loopMIDI fuer virtuelle MIDI Ports

---

## 2. Start in zwei Minuten

### macOS und Linux

```bash
chmod +x start_mac_linux.sh
./start_mac_linux.sh
```

### Windows

Doppelklick auf `start_windows.bat`.

### Von Hand

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python tools/check_setup.py       # prueft Kamera, Audio, Pakete
python main.py
```

Beim ersten Start fragt macOS nach der Kameraberechtigung fuer das
Terminal. Ohne die Freigabe bleibt das Bild schwarz.

### Ohne Kamera testen

```bash
python main.py --mock             # virtuelle Haende, alles andere echt
python main.py --mock --skip-lobby     # ohne Startbildschirm
python tools/render_demo.py demo.wav   # 16 Sekunden Demo als WAV
```

---

## 3. Wie gespielt wird

### Die Haltung

Stell dich so hin, dass Oberkoerper und beide Unterarme im Bild sind. Die
Haende bleiben locker vor dem Oberkoerper, etwa auf Brusthoehe, die
Handflaechen zur Kamera, die Ellbogen am Koerper. Genau dort bleiben sie
auch: alles wird ueber die Haltung gespielt, nicht ueber die Position.
Das ist bequemer, sieht ruhiger aus und erkennt deutlich stabiler als
Herumfahren im Bild.

Zwei Dinge helfen sehr:

* **Finger klar strecken oder klar einklappen.** Halbe Sachen liest die
  Kamera als Zwischending, dann wackelt die Dichte.
* **Beim Drehen das ganze Handgelenk drehen**, nicht nur kippen. Der
  Handruecken soll flach zur Kamera stehen, so wie vorher die
  Handflaeche. Dazwischen liegt eine Totzone, in der nichts passiert -
  die Drehung darf also ruhig langsam sein.

### Steuerung pro Hand

| Geste | Wirkung |
|---|---|
| **Pinch** (Daumen und Zeigefinger beruehren sich) | leere Hand an der Bibliothek links: nimmt den Beat in ihrer Hoehe. Leere Hand am Regal rechts: holt den Loop zurueck. Hand mit Loop: legt ihn ab |
| **Hand zur Seite neigen** | nur bei einem gehaltenen Loop: Lautstaerke. Aufrecht ist die Mitte, nach aussen lauter, nach innen leiser |
| **Finger strecken (1 bis 4)** | Dichte des Beats: ein Finger sparsam, vier Finger voll |
| **Faust** | stumm, der Loop bleibt aber in der Hand |
| **Daumen abspreizen** | Klangfarbe von dumpf bis offen |
| **Handgelenk umdrehen** | haelt alle Werte fest. Zurueckdrehen gibt sie wieder frei |
| **Beide Haende zusammenfuehren** | zwei gehaltene Loops verschmelzen zu einem |
| **Kurzer Pinch mit Loop in der Hand** | legt den Loop ins Regal, die Hand ist wieder frei |
| **Pinch lange halten** | wirft weg: mit Loop in der Hand ihn selbst, am Regal das Fach, auf das die Hand zeigt. Karte bzw. Fach zeigen den Fortschritt |
| **Gehaltenen Loop in den Papierkorb tragen** | Hand mit Loop kurz auf das Muelleimer Symbol unter dem Regal halten loescht ihn (siehe unten) |
| **Alle Haende weit oeffnen / gemeinsam zur Faust** | Riser und Drop |

Fast keine dieser Gesten haengt daran, **wo** die Hand im Bild steht - nur
die beiden Spalten am Rand und der Papierkorb tun das bewusst, weil man
dort etwas Konkretes im Raum sieht und danach greift (siehe unten). Sonst
darf die Hand die ganze Zeit entspannt vor dem Koerper bleiben; sie zaehlt
Finger, neigt sich, dreht sich und pinct. Bewegt wird nur noch zum Verschmelzen,
und auch das nur die kurze Strecke, bis beide Haende sich treffen.

Der Loop wird immer mit den Werten gehalten, die **vor** der Drehung
geklungen haben. Beim Drehen kippt der Handteller kurz auf die Kante,
sonst wuerde jeder gehaltene Beat dumpfer einrasten, als er eben noch war.

### Der Kreislauf

1. **Greifen.** Leere Hand zur Bibliothek links im Bild fuehren, auf Hoehe
   des gewuenschten Beats. Das Fach hellt auf und eine Haarlinie zeigt von
   der Hand darauf.
2. **Nehmen.** Pinch, bis der Balken im Fach voll ist.
3. **Formen.** Finger zaehlen die Dichte, Neigung die Lautstaerke, Daumen
   die Klangfarbe. Die Karte neben der Hand zeigt alles mit.
4. **Halten.** Handgelenk umdrehen. Die Karte zeigt `GEHALTEN`.
5. **Verschmelzen.** Zweiten Loop genauso in die andere Hand holen und
   halten, dann beide Haende zusammenfuehren. Die Linie zwischen den
   Haenden fuellt sich, danach traegt ein Loop beide Beats.
6. **Ablegen.** Kurz pinchen und wieder oeffnen. Der Loop fliegt ins Regal
   und spielt dort weiter. Wer den Pinch geschlossen haelt, sieht die Linie
   unter der Karte rot volllaufen - dann ist der Loop weg.
7. **Zurueckholen.** Leere Hand ans Regal rechts, auf Hoehe des Fachs.
   Kurzer Pinch - auf, zu, wieder auf - holt den Loop mit allen Werten
   zurueck in die Hand. Wer den Pinch stattdessen haelt, sieht den Balken
   im Fach rot volllaufen und loescht den Loop.

Ein verschmolzener Loop behaelt die Mischung seiner Beats. Wird er wieder
aufgedreht, ist die Neigung sein Summenregler: aufrecht voll, nach innen
gekippt blendet er aus.

### Die beiden Spalten

Links die **Bibliothek** mit den Beats, die es zu greifen gibt, rechts
das **Regal** mit den Loops, die schon gebaut sind. Beide sind Spalten
aus grossen, gleich hohen Faechern an festen Plaetzen - kein Menu zum
Durchblaettern und, wichtiger, nichts, was verrutscht: ein Fach bleibt,
wo es ist, auch wenn das darueber leer wird. Wie beim Papierkorb
entscheidet hier bewusst die Position der Hand, nicht ihre Haltung: man
sieht das Fach und greift danach.

Eine leere Hand, die nah genug an eine Spalte herankommt, zeigt auf das
Fach in ihrer Hoehe. Damit die Auswahl nicht an jeder Fachgrenze
flackert, wird die Handhoehe kurz beruhigt, und die Auswahl springt erst
um, wenn die Hand deutlich im Nachbarfach steht. In der Praxis heisst
das: einmal hinhalten, kurz stehen bleiben, greifen.

* **Bibliothek links.** Ein Fach pro Beat, mit Kuerzel, Namen und - bei
  eigenen Loops - Taktzahl und Anzahl der Dichtestufen. Ein genommener
  Beat bleibt in der Bibliothek, denselben Beat koennen also mehrere
  Leute gleichzeitig halten. Die Spalte aendert sich nie waehrend des
  Spielens.
* **Regal rechts.** Fuenf Faecher (`max_parked` in der `config.json`).
  Ein abgelegter Loop laeuft dort weiter und zeigt seine Lautstaerke.
  Kurzer Pinch holt ihn zurueck in die Hand, langer Pinch loescht ihn.
  Ist das Regal voll, sagt die Karte in der Hand `REGAL VOLL`, statt den
  Pinch stillschweigend zu verschlucken.

### Der Papierkorb

Unter dem Regal rechts sitzt ein kleines Muelleimer Symbol. Wie das
Nehmen aus den Spalten geht das bewusst ueber die Position der Hand
statt ueber ihre Haltung: einen gehaltenen Loop dorthin tragen und kurz
halten loescht ihn komplett, mit einem Fortschrittsring an der Karte und
am Symbol selbst. Alternativ funktioniert weiterhin der lange Pinch aus
der Tabelle oben.

### Gruppen Macros


Beide laufen ueber die Haltung, nicht ueber die Hoehe der Haende. Niemand
muss die Arme ueber den Kopf nehmen.

* **Riser**: alle Haende, die gerade einen Loop einstellen, gemeinsam weit
  oeffnen - vier Finger gestreckt und der Daumen deutlich abgespreizt. Der
  Riser Wert steigt sichtbar an. Musikalisch passt das zusammen: die offene
  Hand ist zugleich volle Dichte und offene Klangfarbe.
* **Drop**: aus dem vollen Riser heraus ballen alle gemeinsam die Faust.
  Die Faust schaltet die Loops stumm, der Impuls setzt genau dort ein.

### Tasten im Fenster

`ESC` beenden, `f` Vollbild, `m` Spiegelung, `h` Skelett ein und aus,
`r` Bibliothek neu mischen, `Leertaste` alles Gehaltene ablegen,
`x` alles loeschen, `l` zurueck zur Auswahl.

---

## 4. Eigene Beats statt der eingebauten

Es gibt zwei Wege, und sie loesen zwei verschiedene Aufgaben.

### 4a. Eigene Loops - der normale Weg

Wer fertige Loops hat (ein Takt Drums, eine Basslinie, ein Top Loop),
legt sie einfach in den Ordner `loops/`. Sonst nichts: kein Eintrag im
Code, keine Konfiguration. Beim naechsten Start stehen sie links in der
Bibliothek und ersetzen die eingebauten Beats.

```
loops/
    toploop.wav              -> ein Beat, eine Datei
    bassline/                -> ein Beat mit vier Dichtestufen
        1_sparse.wav             1 Finger
        2.wav                    2 Finger
        3.wav                    3 Finger
        4_full.wav               4 Finger
```

Der Ordner ist der bessere Weg: die Dateien darin sind dieselbe Spur in
vier Dichten, und die Fingerzahl waehlt zwischen ihnen - genau so, wie
die Fingerzahl bei den eingebauten Beats zwischen vier Schlagmustern
waehlt. Mit weniger als vier Dateien werden die Stufen gleichmaessig
verteilt; mit einer einzigen aendert die Fingerzahl nichts, und die
Karte in der Hand zeigt das an (nur so viele Striche bei `DICHTE`, wie
es Dateien gibt).

**Tempo.** Die Dateien muessen nicht im Tempo des Instruments liegen.
Aus der Laenge wird die Taktzahl geschaetzt und die Datei exakt auf das
Raster gezogen, damit alles zusammen laeuft. Genauer geht es mit einer
`loops.json` neben den Dateien:

```json
{
  "bassline": {
    "label": "DEEP BASS", "short": "BAS",
    "bpm": 140, "bars": 2,
    "colour": "#7AA2FF", "family": "bass", "gain": 0.9
  }
}
```

Alle Felder sind freiwillig. `bpm` ist das Tempo der Aufnahme, `bars`
ihre Laenge in Takten.

**Vorher pruefen.** Dieser Befehl zeigt, als was jede Datei erkannt
wurde - Taktzahl, Dichtestufen, Laenge - und was uebersprungen wurde:

```bash
python tools/check_loops.py
python tools/check_loops.py --bpm 140      # fuer ein anderes Zieltempo
```

**Einmal ausprobieren.** Der Befehl legt zwei Beispiel Loops an, damit
man die Ablage einmal in Betrieb sieht, bevor die eigenen Dateien
hineinkommen:

```bash
python tools/make_example_loops.py          # anlegen
python tools/make_example_loops.py --clean  # wieder entfernen
```

**Format.** WAV, PCM 8/16/32 bit, Mono oder Stereo, jede Samplerate.
MP3 und AIFF gehen nicht, die vorher in WAV wandeln.

**Zurueck zu den eingebauten Beats.** `python main.py --library builtin`,
oder dauerhaft `"library": "builtin"` im Abschnitt `audio` der
`config.json`. `"auto"` (Standard) nimmt die eigenen Loops, sobald
welche da sind, sonst die eingebauten.

Wo das im Code haengt: `src/audio/loops.py` liest den Ordner,
`src/app.py` (`install_library`) setzt die Bibliothek damit, und
`src/audio/engine.py` spielt eine Loop Schicht am Stueck statt aus
Einzelschlaegen. Die Regler bleiben dieselben - Neigung Lautstaerke,
Daumen Klangfarbe, Finger Dichte.

### 4b. Einzelne Klaenge der eingebauten Beats austauschen

Die eingebauten Beats setzen sich aus einzelnen Schlaegen auf einem
16tel Raster zusammen; die Klaenge dafuer werden aus Wellenformen
berechnet (`src/audio/synth.py`). Wer nur einen davon ersetzen will -
etwa eine eigene Bassdrum, aber die eingebauten Muster behalten:

1. WAV Datei in `samples/` ablegen, z.B. `samples/kick/mein_kick.wav`.
2. In `src/audio/sample_files.py` eintragen, unter welcher "Bank" sie
   laufen soll:

   ```python
   CUSTOM_SAMPLES = {
       "kick": ["samples/kick/mein_kick.wav"],
   }
   ```

3. `python main.py` neu starten. Die Datei ersetzt dann automatisch den
   eingebauten Klang dieser Bank; Lautstaerke, Filter und Pattern eines
   Presets bleiben unveraendert, weil die an der Bank haengen und nicht
   an der einzelnen Datei.

Welche Bank Namen es gibt, steht als Feld `bank` bei jedem Preset in
`src/mapping/presets.py`. Die vollstaendige Anleitung samt Beispielen
steht direkt im Kopf von `src/audio/sample_files.py`.

---

## 5. Kommandozeile

```bash
python main.py --library builtin      # eigene Loops in loops/ ignorieren
python main.py --performers 2         # Auswahl vorbelegen
python main.py --skip-lobby           # ohne Startbildschirm direkt los
python main.py --countdown 5          # laengerer Countdown
python main.py --seed 42              # fester Zufall
python main.py --camera 1             # zweite Kamera
python main.py --bpm 140              # Tempo der internen Engine
python main.py --no-audio             # nur OSC, interne Engine aus
python main.py --no-osc               # nur interne Engine
python main.py --fullscreen           # direkt auf den Beamer
python main.py --mock --no-window --frames 100    # Testlauf
python main.py --preview bild.png     # letztes Bild speichern
```

Dauerhafte Einstellungen stehen in `config.json` und werden beim Start
gelesen. Die Kommandozeile gewinnt gegen die Datei.

---

## 6. OSC Schnittstelle

Ziel ist `127.0.0.1:9000`, alle Werte sind Floats von 0 bis 1. Es gibt
eine Adressgruppe pro Preset aus der Kiste:

```
/stem/kick_four/volume     Lautstaerke
/stem/kick_four/cutoff     Filter
/stem/kick_four/density    Pattern Dichte
/stem/kick_four/active     1 = dieser Loop ist gerade im Spiel
/global/riser              Build-up, 0 bis 1
/global/drop               kurzer Impuls
/global/hands              Anzahl erkannter Haende
/global/tokens             Anzahl Loops im Spiel
/global/placed             davon abgelegt
/global/heartbeat          laeuft immer, zum Pruefen der Verbindung
```

Analog fuer alle sechzehn Preset IDs: `kick_four`, `kick_broken`, `clap`,
`snare`, `hats_closed`, `hats_trap`, `rim`, `toms`, `bass_sub`,
`bass_acid`, `lead_pluck`, `arp`, `chord_stab`, `pad`, `sweep`, `blip`.
Laeuft dasselbe Preset in mehreren Tokens, gewinnt der lauteste.

Mitlesen, ob etwas ankommt:

```bash
python tools/osc_monitor.py 9000
```

DAW Anbindung: siehe `max_for_live/README.md`.

---

## 7. Aufbau am Veranstaltungsort

* Kamera auf Stativ, ungefaehr auf Brusthoehe, leicht nach unten geneigt.
  Die Kiste liegt im unteren Fuenftel des Bildes, dorthin muss man mit der
  Hand bequem hinunterkommen.
* Drei bis vier Meter Abstand, damit drei Leute nebeneinander mit
  ausgestreckten Armen ins Bild passen.
* Beamerbild moeglichst gross. Gespiegelt ist Standard, damit die
  Bewegung im Bild zur eigenen Bewegung passt.
* Licht von vorne. Gegenlicht ist der haeufigste Grund fuer schlechtes
  Tracking, ein Fenster im Ruecken der Spieler killt die Erkennung.
* Beamer auf den erweiterten Bildschirm, dann `f` fuer Vollbild.
* Bei schwacher Hardware: `--width 960 --height 540` reicht voellig.

---

## 8. Projektstruktur

```
gesture-music-instrument/
├── main.py                  Startpunkt
├── config.json              Voreinstellungen
├── requirements.txt
├── start_mac_linux.sh       Setup und Start in einem
├── start_windows.bat
├── src/
│   ├── app.py               Argumente, Hauptschleife, Tasten
│   ├── config.py            Konfiguration
│   ├── engine.py            Haende zu Snapshot
│   ├── layout.py            Geometrie der beiden Spalten
│   ├── vision/
│   │   ├── camera.py        Kamera in eigenem Thread
│   │   ├── tracker.py       MediaPipe Hand Landmarker und Mock
│   │   ├── hand_tracker.py  stabile IDs pro Hand ueber Frames
│   │   └── types.py         gemeinsame Datentypen
│   ├── mapping/
│   │   ├── presets.py       die eingebauten Beats und die aktive Bibliothek
│   │   ├── tokens.py        Bibliothek, Halten, Fusion, Regal
│   │   ├── gestures.py      Finger, Neigung, Daumen, Handdrehung
│   │   ├── normalizer.py    Glaettung
│   │   └── macros.py        Riser und Drop
│   ├── network/osc_bridge.py
│   ├── audio/
│   │   ├── engine.py        Sequencer mit beliebig vielen Schichten
│   │   ├── loops.py         eigene Loops aus dem Ordner loops/
│   │   ├── sample_files.py  eigene Einzelklaenge fuer die eingebauten Beats
│   │   └── synth.py         Klangerzeugung und Filter
│   └── ui/
│       ├── theme.py         Farben, Schrift, Glas - fuer beide Bildschirme
│       ├── display.py       Projektorbild: Bibliothek, Regal, Handkarten
│       └── lobby.py         Startbildschirm
├── loops/                   hier kommen eure eigenen Loop WAVs hinein
├── tools/
│   ├── check_loops.py       eigene Loops pruefen
│   ├── make_example_loops.py  zwei Beispiel Loops zum Ausprobieren
│   ├── check_setup.py       Systemcheck
│   ├── osc_monitor.py       OSC mitlesen
│   ├── osc_to_midi.py       Bruecke zu jedem DAW
│   └── render_demo.py       Demo WAV ohne Kamera
├── tests/                   111 Tests, laufen ohne Kamera
└── max_for_live/            Anleitung und Max Patch
```

---

## 9. Tests

```bash
python -m unittest discover -s tests -v
```

Die 111 Tests laufen komplett ohne Kamera, ohne Soundkarte und ohne Modell.
Abgedeckt sind die Haltungserkennung (Fingerzahl, Neigung, Daumen,
Handdrehung, Pinch) einschliesslich des Nachweises, dass die Werte von
der Position im Bild unabhaengig sind, die Wiedererkennung einzelner
Haende ueber mehrere Frames, der komplette Kreislauf aus Bibliothek, Halten,
Fusion und Regal - inklusive Herausnehmen und Loeschen aus dem Regal und
der Sperre gegen flackernde Auswahl -, alle Presets, der Weg eigener
Loops vom WAV im Ordner bis zum Ton, der OSC Versand ueber einen echten
lokalen Empfaenger und die Sound Engine inklusive Rechenzeitbudget. Dazu
kommt ein Durchlauf beider Bildschirme: Spielanzeige, Startbildschirm und
Countdown werden einmal komplett in ein Testbild gezeichnet, und die
Handkarten muessen sich auch bei sechs Haenden nicht ueberdecken.

---

## 10. Wenn etwas nicht laeuft

| Symptom | Ursache und Loesung |
|---|---|
| `Kamera 0 laesst sich nicht oeffnen` | anderer Index: `--camera 1`, oder Kamerarechte in den Systemeinstellungen |
| Schwarzes Bild unter macOS | Terminal braucht Kameraberechtigung, danach Terminal neu starten |
| Modell laedt nicht | Datei manuell laden und nach `models/hand_landmarker.task` legen, URL steht in `src/config.py` |
| Kein Ton | `pip install sounddevice`, unter Linux zusaetzlich `sudo apt install libportaudio2` |
| Ton knackst | `"blocksize": 512` in `config.json` |
| Pinch wird nicht erkannt | Finger deutlicher schliessen, oder `"pinch_max": 0.38` in `config.json` |
| Loop friert nicht ein | Handgelenk weiter drehen, oder `"flip_threshold": 0.18` in `config.json` |
| Loop wird beim Greifen versehentlich abgelegt | `"place_frames": 6` in `config.json` |
| Kurzer Pinch legt nichts ab | `"place_frames": 3` in `config.json` |
| Loop verschwindet zu schnell | `"discard_frames": 30` in `config.json` |
| Riser loest nicht aus | Daumen deutlicher abspreizen, oder `"riser_open": 0.78` |
| Fusion loest nicht aus | Haende naeher zusammen, oder `"fuse_distance": 0.20` |
| Eingefrorener Wert passt nicht | `"flip_arm": 0.45` in `config.json` |
| Werte zittern | `"smoothing": 0.5` in `config.json` |
| Schrift sieht kantig aus | Pillow fehlt: `pip install pillow` (dann echte Systemschrift statt OpenCV Strichfont) |
| Wenig FPS | `--width 960 --height 540`, Fenster kleiner ziehen |
| Nichts kommt im DAW an | erst `python tools/osc_monitor.py`, dann Port pruefen |
