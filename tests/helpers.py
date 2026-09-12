"""Hilfsmittel, um Haende ohne Kamera durch das Bild zu fuehren.

Gespielt wird ueber die Haltung, deshalb beschreibt eine Testhand vor
allem ihre Pose: wie viele Finger stehen, wie sie geneigt ist, wie weit
der Daumen absteht und auf welcher Seite das Handgelenk steht.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.config import Config  # noqa: E402
from src.engine import InstrumentEngine  # noqa: E402
from src.vision.tracker import synthetic_hand  # noqa: E402

FRAME_TIME = 1.0 / 30.0

POSE = {"fingers": 2, "thumb": 1.0, "tilt": 0.0, "roll": 1.0, "pinch": False}


def hand_at(x, y, handedness="right", **pose):
    """Hand, deren getrackte Mitte genau auf (x, y) liegt.

    Der Versatz wird pro Pose neu bestimmt, denn Neigung und Drehung
    verschieben die Handmitte gegenueber der Handwurzel.
    """
    probe = synthetic_hand(0.0, 0.0, handedness, **pose)
    centre = probe.center
    shift = np.array([x - float(centre[0]), y - float(centre[1]), 0.0], dtype=np.float32)
    probe.landmarks += shift
    return probe


def test_config(performers=3):
    config = Config()
    config.performers = performers
    config.mapping.smoothing = 0.0
    config.osc.enabled = False
    config.audio.enabled = False
    return config


class Driver:
    """Fuehrt benannte Haende durch das Bild und treibt die Engine dabei mit.

    Die feste Bildzeit haelt die Blaettergeschwindigkeit reproduzierbar -
    sonst wuerde die reale Schleifengeschwindigkeit in die Tests
    einfliessen.
    """

    def __init__(self, config=None, seed=1):
        self.config = config or test_config()
        self.engine = InstrumentEngine(self.config, seed=seed)
        self.hands = {}
        self.snapshot = None

    # ------------------------------------------------------------------
    @property
    def forge(self):
        return self.engine.forge

    @property
    def tokens(self):
        return self.engine.forge.tokens

    def entry_index(self, preset_id):
        """Fach des Presets in der linken Spalte."""
        for index, entry in enumerate(self.forge.library_entries()):
            if entry.preset_ids == [preset_id]:
                return index
        raise LookupError(preset_id)

    def rack_index(self, token_id):
        """Fach des abgelegten Loops in der rechten Spalte."""
        for index, entry in enumerate(self.forge.rack_entries()):
            if entry is not None and entry.token_id == token_id:
                return index
        raise LookupError(token_id)

    # ------------------------------------------------------------------
    def place(self, name, x=0.5, y=0.5, handedness="right", **pose):
        hand = {"x": x, "y": y, "handedness": handedness}
        hand.update(POSE)
        hand.update(pose)
        self.hands[name] = hand
        return hand

    def set(self, name, **pose):
        self.hands[name].update(pose)

    def remove(self, name):
        self.hands.pop(name, None)

    def step(self, frames=1):
        for _ in range(frames):
            observations = [
                hand_at(
                    hand["x"], hand["y"], hand["handedness"],
                    fingers=hand["fingers"], thumb=hand["thumb"],
                    tilt=hand["tilt"], roll=hand["roll"], pinch=hand["pinch"],
                )
                for hand in self.hands.values()
            ]
            self.snapshot = self.engine.update(observations, dt=FRAME_TIME)
        return self.snapshot

    def move(self, name, x, y, frames=10):
        hand = self.hands[name]
        start_x, start_y = hand["x"], hand["y"]
        for index in range(1, frames + 1):
            hand["x"] = start_x + (x - start_x) * index / frames
            hand["y"] = start_y + (y - start_y) * index / frames
            self.step()
        return self.snapshot

    # ------------------------------------------------------------------
    def browser(self, name):
        """Blaetterzustand der Hand, die unter diesem Namen laeuft."""
        track_id = self.track_id(name)
        return self.forge.browsers.get(track_id)

    def track_id(self, name):
        position = list(self.hands).index(name)
        return sorted(self.engine.hands.tracks)[position]

    def reach(self, name, column, index, x):
        """Fuehrt die Hand vor das Fach `index` einer Spalte."""
        self.move(name, x, column.slot_y(index), frames=12)
        browser = self.browser(name)
        if browser is not None:
            browser.cooldown = 0
            browser.latched = False
        return browser

    def take(self, name, index):
        """Fuehrt die Hand zum Fach `index` der Bibliothek und greift es."""
        cfg = self.config.forge
        x = min(self.hands[name]["x"], cfg.library_capture_x)
        self.reach(name, self.forge.library_column, index, x)
        self.set(name, pinch=True)
        self.step(cfg.take_frames + 1)
        self.set(name, pinch=False)
        self.step()
        return self.snapshot

    def take_from_rack(self, name, index):
        """Kurzer Pinch am Regal: der Loop kommt zurueck in die Hand."""
        cfg = self.config.forge
        x = max(self.hands[name]["x"], cfg.rack_capture_x + 0.02)
        self.reach(name, self.forge.rack_column, index, x)
        self.set(name, pinch=True)
        self.step(cfg.take_frames + 1)
        self.set(name, pinch=False)
        self.step(2)
        return self.snapshot

    def delete_from_rack(self, name, index):
        """Pinch am Regal lange halten: der Loop wird geloescht."""
        cfg = self.config.forge
        x = max(self.hands[name]["x"], cfg.rack_capture_x + 0.02)
        self.reach(name, self.forge.rack_column, index, x)
        self.set(name, pinch=True)
        self.step(cfg.discard_frames + 2)
        self.set(name, pinch=False)
        self.step(2)
        return self.snapshot

    def grab(self, name, preset_id, handedness="right", **pose):
        """Nimmt ein Preset aus dem Browser in eine frische Hand."""
        if name not in self.hands:
            self.place(name, 0.35 if handedness == "right" else 0.65, 0.45,
                       handedness=handedness, **pose)
        self.take(name, self.entry_index(preset_id))
        return self.tokens[-1]

    def hold(self, name):
        """Handgelenk umdrehen: der Loop haelt seine Werte."""
        self.set(name, roll=-1.0)
        self.step(self.config.mapping.flip_frames + 2)
        return self.snapshot

    def release_hold(self, name):
        """Handflaeche wieder zeigen: der Loop folgt wieder der Hand."""
        self.set(name, roll=1.0)
        self.step(self.config.mapping.flip_frames + 2)
        return self.snapshot

    def put_away(self, name):
        """Kurzer Pinch: der Loop wandert ins Regal."""
        self.step(self.config.forge.min_hold_frames + 2)
        self.set(name, pinch=True)
        self.step(self.config.forge.place_frames + 1)
        self.set(name, pinch=False)
        self.step()
        return self.snapshot

    def throw_away(self, name):
        """Pinch lange halten: der Loop wird weggeworfen."""
        self.step(self.config.forge.min_hold_frames + 2)
        self.set(name, pinch=True)
        self.step(self.config.forge.discard_frames + 1)
        self.set(name, pinch=False)
        self.step()
        return self.snapshot

    def settle(self, frames=30):
        return self.step(frames)
