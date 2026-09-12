"""Tests fuer Presets, die OSC Ausgabe und die interne Sound Engine."""

import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import Driver  # noqa: E402
from src.audio import synth  # noqa: E402
from src.audio.engine import AudioEngine  # noqa: E402
from src.config import AudioConfig, OscConfig  # noqa: E402
from src.mapping.presets import PRESETS  # noqa: E402
from src.network.osc_bridge import OscBridge, build_bridge  # noqa: E402


def playing_snapshot():
    """Ein abgelegter und ein gehaltener Loop, beide klingend."""
    driver = Driver(seed=11)
    driver.grab("a", "kick_four")
    driver.set("a", fingers=3, tilt=-40, thumb=0.8)
    driver.step(8)
    driver.hold("a")
    driver.put_away("a")
    driver.settle(20)
    driver.remove("a")
    driver.step(driver.config.forge.release_frames + 4)

    driver.place("b", 0.60, 0.45, handedness="left")
    driver.grab("b", "bass_sub", handedness="left")
    driver.set("b", fingers=2, tilt=30, thumb=0.6)
    return driver.step(8)


class TestPresets(unittest.TestCase):
    def setUp(self):
        self.bank = synth.build_sample_bank(22050)

    def test_every_preset_has_samples(self):
        for preset in PRESETS:
            with self.subTest(preset=preset.id):
                self.assertIn(preset.bank, self.bank)
                self.assertTrue(self.bank[preset.bank])

    def test_every_preset_has_four_usable_patterns(self):
        for preset in PRESETS:
            with self.subTest(preset=preset.id):
                self.assertEqual(len(preset.patterns), 4)
                for pattern in preset.patterns:
                    self.assertTrue(pattern)
                    self.assertTrue(all(0 <= step < 16 for step in pattern))

    def test_density_selects_rising_pattern_levels(self):
        preset = PRESETS[0]
        self.assertEqual(preset.pattern_for(0.0), preset.patterns[0])
        self.assertEqual(preset.pattern_for(1.0), preset.patterns[3])
        self.assertEqual(preset.pattern_for(0.5), preset.patterns[1])

    def test_preset_ids_are_unique(self):
        ids = [preset.id for preset in PRESETS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_samples_stay_in_range(self):
        for name, samples in self.bank.items():
            for index, buffer in enumerate(samples):
                with self.subTest(bank=name, index=index):
                    self.assertGreater(len(buffer), 0)
                    self.assertLessEqual(float(np.max(np.abs(buffer))), 1.0)
                    self.assertFalse(bool(np.any(np.isnan(buffer))))


class TestOscBridge(unittest.TestCase):
    def test_messages_reach_a_local_receiver(self):
        from pythonosc.dispatcher import Dispatcher
        from pythonosc.osc_server import ThreadingOSCUDPServer

        received = {}
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(
            lambda address, *args: received.__setitem__(address, args[0] if args else None)
        )
        server = ThreadingOSCUDPServer(("127.0.0.1", 0), dispatcher)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            snapshot = playing_snapshot()
            bridge = OscBridge(OscConfig(host="127.0.0.1", port=port, send_rate=1000))
            self.assertTrue(bridge.publish(snapshot, force=True))
            time.sleep(0.35)
        finally:
            server.shutdown()
            server.server_close()

        self.assertIn("/stem/kick_four/volume", received)
        self.assertIn("/stem/bass_sub/cutoff", received)
        self.assertIn("/global/riser", received)
        self.assertIn("/global/placed", received)
        self.assertGreater(received["/stem/kick_four/volume"], 0.0)
        self.assertEqual(received["/stem/kick_four/active"], 1.0)
        # Ungenutzte Presets melden sich als stumm, nicht gar nicht.
        self.assertEqual(received["/stem/sweep/active"], 0.0)
        self.assertGreaterEqual(len(received), len(PRESETS) * 4)

    def test_rate_limit_blocks_fast_calls(self):
        snapshot = playing_snapshot()
        bridge = OscBridge(OscConfig(host="127.0.0.1", port=59999, send_rate=5))
        self.assertTrue(bridge.publish(snapshot, force=True))
        self.assertFalse(bridge.publish(snapshot))

    def test_disabled_bridge_is_a_null_object(self):
        bridge = build_bridge(OscConfig(enabled=False))
        self.assertFalse(bridge.publish(playing_snapshot()))
        bridge.close()


class TestAudioEngine(unittest.TestCase):
    def _engine(self):
        return AudioEngine(AudioConfig(samplerate=22050, blocksize=256, bpm=128))

    def test_silence_without_any_slot(self):
        audio = self._engine()
        block = audio.render_offline(1.0)
        self.assertEqual(block.shape, (22050, 2))
        self.assertLess(float(np.max(np.abs(block))), 1e-4)

    def test_every_preset_makes_sound_on_its_own(self):
        for preset in PRESETS:
            with self.subTest(preset=preset.id):
                audio = self._engine()
                audio.set_slot("one", preset.id, gain=0.9, cutoff=0.9, density=0.8)
                block = audio.render_offline(2.0)
                self.assertGreater(float(np.max(np.abs(block))), 0.01)
                self.assertFalse(bool(np.any(np.isnan(block))))

    def test_many_layers_stay_within_range(self):
        audio = self._engine()
        for index, preset in enumerate(PRESETS):
            audio.set_slot(f"s{index}", preset.id, gain=0.9, cutoff=0.8, density=0.9)
        block = audio.render_offline(2.0)
        peak = float(np.max(np.abs(block)))
        self.assertGreater(peak, 0.05)
        self.assertLessEqual(peak, 1.0)

    def test_removing_a_slot_silences_it(self):
        audio = self._engine()
        audio.set_slot("one", "kick_four", gain=0.9, cutoff=0.9, density=1.0)
        audio.render_offline(1.0)
        self.assertIn("one", audio.layers)

        audio.update_from_snapshot(_empty_snapshot())
        block = audio.render_offline(1.5)
        self.assertEqual(audio.layers, {})
        self.assertLess(float(np.max(np.abs(block))), 1e-4)

    def test_cutoff_reduces_high_frequency_energy(self):
        def energy(cutoff):
            audio = self._engine()
            audio.set_slot("one", "hats_closed", gain=1.0, cutoff=cutoff, density=1.0)
            block = audio.render_offline(2.0)[:, 0]
            spectrum = np.abs(np.fft.rfft(block))
            freqs = np.fft.rfftfreq(len(block), 1 / 22050)
            return float(np.sum(spectrum[freqs > 4000]))

        self.assertLess(energy(0.0), energy(1.0))

    def test_snapshot_drives_the_engine(self):
        snapshot = playing_snapshot()
        audio = self._engine()
        audio.update_from_snapshot(snapshot)
        block = audio.render_offline(1.5)
        self.assertGreater(float(np.max(np.abs(block))), 0.01)

    def test_realtime_budget(self):
        audio = self._engine()
        for index, preset in enumerate(PRESETS):
            audio.set_slot(f"s{index}", preset.id, gain=0.8, cutoff=0.6, density=1.0)
        audio.render_offline(0.5)          # warmlaufen
        start = time.perf_counter()
        audio.render_offline(2.0)
        elapsed = time.perf_counter() - start
        # Deutlich schneller als Echtzeit, sonst knackst die Ausgabe.
        self.assertLess(elapsed, 0.8)


def _empty_snapshot():
    from src.vision.types import EngineSnapshot, MacroState

    return EngineSnapshot(tokens=[], slots=[], library=[], rack=[], hands=[],
                          macros=MacroState())


if __name__ == "__main__":
    unittest.main(verbosity=2)
