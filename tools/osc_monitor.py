#!/usr/bin/env python3
"""Zeigt alle eingehenden OSC Nachrichten an.

Gut zum Pruefen, ob das Mapping ankommt, bevor der DAW dran ist.

Aufruf:  python tools/osc_monitor.py [port]
Beenden: Strg+C
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

values: dict[str, float] = defaultdict(float)
counter = {"n": 0}


def handle(address: str, *args) -> None:
    if args:
        values[address] = float(args[0])
    counter["n"] += 1


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(handle)
    server = ThreadingOSCUDPServer(("127.0.0.1", port), dispatcher)

    import threading

    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"OSC Monitor hoert auf 127.0.0.1:{port}. Strg+C beendet.\n")

    try:
        while True:
            time.sleep(0.25)
            interesting = {
                k: v
                for k, v in sorted(values.items())
                if not k.endswith("heartbeat")
            }
            lines = [f"{k:34s} {v:5.2f}" for k, v in interesting.items()]
            print("\033[2J\033[H", end="")
            print(f"Nachrichten gesamt: {counter['n']}\n")
            print("\n".join(lines[:40]))
    except KeyboardInterrupt:
        print("\nEnde.")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
