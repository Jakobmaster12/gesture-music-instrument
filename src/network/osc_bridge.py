"""OSC Ausgabe an den DAW.

Eine Adressgruppe pro Preset aus der Kiste, zum Beispiel kick_four:

    /stem/kick_four/volume     float 0..1
    /stem/kick_four/cutoff     float 0..1
    /stem/kick_four/density    float 0..1
    /stem/kick_four/active     float 0 oder 1 (liegt gerade als Loop im Bild)

Klingt dasselbe Preset in mehreren Tokens, gewinnt der lauteste.

Global:
    /global/riser         float 0..1
    /global/drop          float 0 oder 1 (Impuls)
    /global/hands         float Anzahl aktiver Haende
    /global/tokens        float Anzahl Loops im Spiel
    /global/placed        float davon abgelegt
    /global/heartbeat     float steigender Zaehler
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

from ..config import OscConfig
from ..mapping.presets import PRESET_IDS
from ..mapping.tokens import PLACED
from ..vision.types import EngineSnapshot


class OscBridge:
    def __init__(self, config: OscConfig):
        self.config = config
        self.client = None
        self._last: Dict[str, float] = {}
        self._last_send = 0.0
        self._heartbeat = 0.0
        self.sent_messages = 0

        if not config.enabled:
            return
        try:
            from pythonosc.udp_client import SimpleUDPClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "python-osc fehlt. `pip install -r requirements.txt`"
            ) from exc
        self.client = SimpleUDPClient(config.host, config.port)

    # ------------------------------------------------------------------
    def _send(self, address: str, value: float, force: bool = False) -> None:
        if self.client is None:
            return
        previous = self._last.get(address)
        if (
            not force
            and previous is not None
            and abs(previous - value) < self.config.min_delta
        ):
            return
        self._last[address] = value
        self.client.send_message(address, float(value))
        self.sent_messages += 1

    def publish(self, snapshot: EngineSnapshot, force: bool = False) -> bool:
        """Sendet den aktuellen Zustand. Gibt True zurueck, wenn gesendet wurde."""
        if self.client is None:
            return False

        now = time.perf_counter()
        interval = 1.0 / max(self.config.send_rate, 1)
        if not force and (now - self._last_send) < interval:
            return False
        self._last_send = now

        loudest: Dict[str, Tuple[float, float, float]] = {}
        for slot in snapshot.slots:
            best = loudest.get(slot.preset_id)
            if best is None or slot.gain > best[0]:
                loudest[slot.preset_id] = (slot.gain, slot.cutoff, slot.density)

        for preset_id in PRESET_IDS:
            prefix = f"/stem/{preset_id}"
            gain, cutoff, density = loudest.get(preset_id, (0.0, 0.0, 0.0))
            self._send(f"{prefix}/volume", gain, force)
            self._send(f"{prefix}/cutoff", cutoff, force)
            self._send(f"{prefix}/density", density, force)
            self._send(f"{prefix}/active", 1.0 if preset_id in loudest else 0.0, force)

        self._send("/global/riser", snapshot.macros.riser, force)
        self._send("/global/drop", snapshot.macros.drop, force)
        self._send("/global/hands", float(snapshot.macros.active_hands), force)
        self._send("/global/tokens", float(len(snapshot.tokens)), force)
        self._send(
            "/global/placed",
            float(sum(1 for t in snapshot.tokens if t.state == PLACED)),
            force,
        )

        self._heartbeat = (self._heartbeat + 1.0) % 1000.0
        self._send("/global/heartbeat", self._heartbeat, force=True)
        return True

    def send_reset(self) -> None:
        """Alle Kanaele auf Null, z.B. beim Beenden."""
        if self.client is None:
            return
        for address in list(self._last.keys()):
            if address.endswith("/volume") or address.endswith("/active"):
                self.client.send_message(address, 0.0)
            elif address.endswith("/mute"):
                self.client.send_message(address, 1.0)

    def close(self) -> None:
        self.send_reset()
        self.client = None


class NullBridge:
    """Platzhalter, wenn OSC abgeschaltet ist."""

    sent_messages = 0

    def publish(self, snapshot: EngineSnapshot, force: bool = False) -> bool:
        return False

    def send_reset(self) -> None:
        return None

    def close(self) -> None:
        return None


def build_bridge(config: OscConfig):
    return OscBridge(config) if config.enabled else NullBridge()
