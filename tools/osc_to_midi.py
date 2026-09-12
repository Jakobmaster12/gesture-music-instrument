#!/usr/bin/env python3
"""Bruecke von OSC nach MIDI CC.

Damit laesst sich jeder DAW steuern, auch ohne Max for Live: das Skript
oeffnet einen virtuellen MIDI Ausgang und schickt die Handwerte als
Control Change Nachrichten. Im DAW einfach MIDI Learn benutzen.

Benoetigt zusaetzlich:  pip install python-rtmidi

Aufruf:  python tools/osc_to_midi.py [--port 9000] [--channel 1]

CC Belegung (MIDI Kanal 1 bis 6 = die sechs Instrumente):
    CC 1  Lautstaerke
    CC 74 Filter Cutoff
    CC 2  Pattern Dichte
    CC 7  Mute (0 oder 127)
    CC 11 Riser (nur Kanal 1)
"""

from __future__ import annotations

import argparse
import threading
import time

CHANNEL_MAP = {
    "/stem/kick": 0,
    "/stem/bass": 1,
    "/stem/snare": 2,
    "/stem/hats": 3,
    "/stem/synth1": 4,
    "/stem/synth2": 5,
}
CC_MAP = {"volume": 1, "cutoff": 74, "density": 2, "mute": 7}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--name", default="GestureInstrument")
    args = parser.parse_args()

    try:
        import rtmidi
    except ImportError:
        print("python-rtmidi fehlt.  pip install python-rtmidi")
        return 1

    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import ThreadingOSCUDPServer

    midi_out = rtmidi.MidiOut()
    try:
        midi_out.open_virtual_port(args.name)
        print(f"Virtueller MIDI Ausgang offen: {args.name}")
    except Exception:
        ports = midi_out.get_ports()
        if not ports:
            print("Kein MIDI Ausgang verfuegbar. Unter Windows zuerst loopMIDI installieren.")
            return 2
        midi_out.open_port(0)
        print(f"MIDI Ausgang: {ports[0]}")

    last: dict[tuple[int, int], int] = {}

    def send_cc(channel: int, cc: int, value: float) -> None:
        midi_value = int(max(0.0, min(1.0, value)) * 127)
        if last.get((channel, cc)) == midi_value:
            return
        last[(channel, cc)] = midi_value
        midi_out.send_message([0xB0 | (channel & 0x0F), cc, midi_value])

    def handle(address: str, *osc_args) -> None:
        if not osc_args:
            return
        value = float(osc_args[0])
        if address == "/global/riser":
            send_cc(0, 11, value)
            return
        prefix, _, param = address.rpartition("/")
        channel = CHANNEL_MAP.get(prefix)
        cc = CC_MAP.get(param)
        if channel is None or cc is None:
            return
        send_cc(channel, cc, value)

    dispatcher = Dispatcher()
    dispatcher.set_default_handler(handle)
    server = ThreadingOSCUDPServer(("127.0.0.1", args.port), dispatcher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Hoere auf OSC 127.0.0.1:{args.port}. Strg+C beendet.")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nEnde.")
    finally:
        server.shutdown()
        server.server_close()
        midi_out.close_port()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
