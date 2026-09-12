"""Anwendungsschleife: Lobby, Countdown, Spielen."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from .audio.engine import AudioEngine
from .audio.loops import load_library
from .config import Config
from .mapping import presets as preset_library
from .mapping.presets import PRESETS
from .engine import InstrumentEngine
from .network.osc_bridge import build_bridge
from .vision.camera import Camera, CameraError, SyntheticCamera
from .vision.tracker import build_tracker

KEY_ESC = 27
STATE_LOBBY = "lobby"
STATE_COUNTDOWN = "countdown"
STATE_PLAY = "play"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gesture-music-instrument",
        description="Gestengesteuertes Multiplayer Musikinstrument",
    )
    parser.add_argument("--config", default="config.json", help="Pfad zur JSON Konfiguration")
    parser.add_argument("--performers", type=int, choices=(1, 2, 3), help="Anzahl Spieler")
    parser.add_argument("--skip-lobby", action="store_true", help="ohne Startbildschirm beginnen")
    parser.add_argument("--countdown", type=float, help="Sekunden zwischen Auswahl und Start")
    parser.add_argument("--camera", type=int, help="Kameraindex (0, 1, 2 ...)")
    parser.add_argument("--width", type=int, help="Kamerabreite")
    parser.add_argument("--height", type=int, help="Kamerahoehe")
    parser.add_argument("--bpm", type=float, help="Tempo der internen Engine")
    parser.add_argument("--osc-port", type=int, help="OSC Zielport")
    parser.add_argument("--osc-host", help="OSC Zielhost")
    parser.add_argument("--no-osc", action="store_true", help="OSC Ausgabe abschalten")
    parser.add_argument("--no-audio", action="store_true", help="interne Sound Engine aus")
    parser.add_argument(
        "--library", choices=("auto", "loops", "builtin"),
        help="woher die Beats kommen: eigene Loops aus loops/, die eingebauten, oder auto",
    )
    parser.add_argument("--no-window", action="store_true", help="ohne Anzeigefenster laufen")
    parser.add_argument("--fullscreen", action="store_true", help="Fenster im Vollbild starten")
    parser.add_argument("--no-mirror", action="store_true", help="Bild nicht spiegeln")
    parser.add_argument("--mock", action="store_true", help="virtuelle Haende statt Kamera")
    parser.add_argument("--seed", type=int, help="fester Zufall fuer die Handschuhpositionen")
    parser.add_argument("--frames", type=int, help="nach N Frames beenden (Testlauf)")
    parser.add_argument("--preview", help="letztes Bild als PNG speichern")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    config = Config.load(args.config)
    if args.performers:
        config.performers = args.performers
    if args.camera is not None:
        config.camera.device = args.camera
    if args.width:
        config.camera.width = args.width
    if args.height:
        config.camera.height = args.height
    if args.bpm:
        config.audio.bpm = args.bpm
    if args.osc_port:
        config.osc.port = args.osc_port
    if args.osc_host:
        config.osc.host = args.osc_host
    if args.no_osc:
        config.osc.enabled = False
    if args.no_audio:
        config.audio.enabled = False
    if args.library:
        config.audio.library = args.library
    if args.no_window:
        config.ui.show_window = False
    if args.fullscreen:
        config.ui.fullscreen = True
    if args.no_mirror:
        config.camera.mirror = False
    if args.countdown is not None:
        config.ui.countdown_seconds = args.countdown
    return config


def install_library(config: Config) -> str:
    """Legt fest, welche Beats in der Bibliothek liegen.

    Liegt im Ordner `loops/` mindestens eine WAV Datei, gewinnt sie: das
    ist der Weg, eigene Loops einzuspielen, ohne eine Zeile Code
    anzufassen (Anleitung in `src/audio/loops.py` und `loops/README.md`).
    Muss vor dem Bau der Engine laufen, denn die Schmiede merkt sich die
    Bibliothek beim Anlegen.
    """
    mode = (config.audio.library or "auto").lower()
    if mode == "builtin":
        preset_library.reset_presets()
        return f"{len(PRESETS)} eingebaute Beats"

    found = load_library(config.audio.samplerate, config.audio.bpm)
    for problem in found.problems:
        print(f"[loops] uebersprungen - {problem}")
    if found.presets:
        preset_library.use_presets(found.presets)
        variants = sum(item.variants for item in found.presets)
        return f"{len(found.presets)} eigene Loops ({variants} Dateien) aus loops/"
    if mode == "loops":
        print("[loops] keine eigenen Loops gefunden, es bleibt bei den eingebauten")
    preset_library.reset_presets()
    return f"{len(PRESETS)} eingebaute Beats"


def run(args: Optional[argparse.Namespace] = None) -> int:
    args = args or build_parser().parse_args()
    config = config_from_args(args)
    library_note = install_library(config)

    camera = SyntheticCamera(config.camera) if args.mock else Camera(config.camera)
    try:
        camera.open()
    except CameraError as exc:
        print(f"[fehler] {exc}")
        return 2

    try:
        tracker = build_tracker(
            config.tracking, mock=args.mock, performers=config.performers
        )
    except Exception as exc:
        camera.close()
        print(f"[fehler] Tracker konnte nicht gestartet werden: {exc}")
        return 3

    engine = InstrumentEngine(config, seed=args.seed)
    bridge = build_bridge(config.osc)
    audio = AudioEngine(config.audio)
    audio_ok = audio.start()

    want_canvas = config.ui.show_window or bool(args.preview)
    display = lobby = None
    if want_canvas:
        from .ui.display import Display
        from .ui.lobby import Lobby, countdown_overlay

        display = Display(config)
        lobby = Lobby(config)

    state = STATE_PLAY
    if want_canvas and not args.skip_lobby:
        state = STATE_LOBBY
    countdown_until = 0.0

    print(
        f"[start] {library_note} | "
        f"OSC {'an' if config.osc.enabled else 'aus'} "
        f"({config.osc.host}:{config.osc.port}) | "
        f"interne Engine {'an' if audio_ok else 'aus'}"
    )
    print(
        "[tasten] ESC beenden | f Vollbild | m Spiegelung | h Skelett | "
        "r Bibliothek mischen | Leertaste alles ablegen | x alles loeschen | "
        "l zurueck zur Auswahl"
    )

    last_canvas = None
    frames = 0
    started = time.time()

    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.003)
                continue

            observations = tracker.process(frame)
            frames += 1

            # ---------------- Lobby ----------------
            if state == STATE_LOBBY:
                choice = lobby.update(observations)
                last_canvas = lobby.render(frame.image)
                if config.ui.show_window:
                    key = display.show(last_canvas)
                    if key in (KEY_ESC, ord("q")):
                        break
                    if key != 255:
                        choice = lobby.handle_key(key) or choice
                if choice:
                    engine.set_performers(choice)
                    lobby.reset()
                    countdown_until = time.time() + config.ui.countdown_seconds
                    state = STATE_COUNTDOWN
                    print(f"[info] {choice} Spieler, {len(PRESETS)} Beats in der Bibliothek")
                if args.frames and frames >= args.frames:
                    break
                continue

            # ---------------- Countdown ----------------
            if state == STATE_COUNTDOWN:
                remaining = countdown_until - time.time()
                snapshot = engine.update(observations)
                base = display.render(frame.image, snapshot, "START")
                last_canvas = countdown_overlay(base, remaining)
                if config.ui.show_window:
                    key = display.show(last_canvas)
                    if key in (KEY_ESC, ord("q")):
                        break
                if remaining <= 0:
                    state = STATE_PLAY
                if args.frames and frames >= args.frames:
                    break
                continue

            # ---------------- Spielen ----------------
            snapshot = engine.update(observations)
            bridge.publish(snapshot)
            if audio_ok:
                audio.update_from_snapshot(snapshot)

            if display is not None:
                status = "OSC" if config.osc.enabled else ""
                status += " AUDIO" if audio_ok else ""
                last_canvas = display.render(frame.image, snapshot, status.strip())

            if display is not None and config.ui.show_window:
                key = display.show(last_canvas)
                if key in (KEY_ESC, ord("q")):
                    break
                if key == ord("f"):
                    display.toggle_fullscreen()
                elif key == ord("m"):
                    config.camera.mirror = not config.camera.mirror
                    camera.config.mirror = config.camera.mirror
                elif key == ord("h"):
                    config.ui.draw_landmarks = not config.ui.draw_landmarks
                elif key == ord("r"):
                    engine.shuffle_library()
                    print("[info] Bibliothek neu gemischt")
                elif key == ord(" "):
                    engine.place_all()
                    print("[info] alle Loops ins Regal gelegt")
                elif key == ord("x"):
                    engine.clear()
                    print("[info] alles geloescht")
                elif key == ord("l") and lobby is not None:
                    engine.clear()
                    state = STATE_LOBBY

            if args.frames and frames >= args.frames:
                break
    except KeyboardInterrupt:
        print("\n[stop] Abbruch durch Benutzer")
    finally:
        if args.preview and last_canvas is not None:
            from .ui.display import save_preview

            save_preview(last_canvas, args.preview)
            print(f"[info] Vorschau gespeichert: {os.path.abspath(args.preview)}")

        duration = max(time.time() - started, 1e-6)
        print(f"[stop] {frames} Frames in {duration:.1f} s ({frames / duration:.1f} FPS)")
        tracker.close()
        camera.close()
        bridge.close()
        audio.stop()
        if display is not None:
            display.close()
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:  # pragma: no cover
        print(f"[fehler] {exc}", file=sys.stderr)
        return 1
