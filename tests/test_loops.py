"""Tests fuer eigene Loop Beats aus dem Ordner `loops/`.

Der Weg vom WAV im Ordner bis zum Ton: einlesen, aufs Taktraster
ziehen, als Preset in die Bibliothek stellen, in der Sound Engine
abspielen. Die Dateien entstehen hier im Test, damit nichts im
Projektordner liegen muss.
"""

import os
import shutil
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.audio import loops  # noqa: E402
from src.audio.engine import AudioEngine  # noqa: E402
from src.config import AudioConfig  # noqa: E402
from src.mapping import presets as preset_library  # noqa: E402

SR = 22050
BPM = 128.0
BAR_SECONDS = 4.0 * 60.0 / BPM     # ein Takt bei 128 BPM


def write_wav(path, seconds, freq=220.0, samplerate=SR, channels=1):
    """Eine kurze Testaufnahme: ein Sinus mit Pulsen auf jeder Viertel."""
    n = int(seconds * samplerate)
    t = np.arange(n, dtype=np.float64) / samplerate
    tone = np.sin(2 * np.pi * freq * t) * 0.6
    data = (tone * 32767).astype("<i2")
    if channels == 2:
        data = np.repeat(data, 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(samplerate)
        handle.writeframes(data.tobytes())
    return path


class LoopFolder(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="loops-")
        loops.clear_cache()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(loops.clear_cache)
        self.addCleanup(preset_library.reset_presets)

    def load(self, bpm=BPM):
        return loops.load_library(SR, bpm, self.dir)


class TestScanning(LoopFolder):
    def test_an_empty_folder_yields_nothing(self):
        library = self.load()
        self.assertFalse(library)
        self.assertEqual(library.presets, [])

    def test_a_single_file_becomes_a_beat(self):
        write_wav(os.path.join(self.dir, "toploop.wav"), BAR_SECONDS)
        library = self.load()
        self.assertEqual(len(library.presets), 1)
        found = library.presets[0]
        self.assertEqual(found.id, "toploop")
        self.assertEqual(found.kind, "loop")
        self.assertEqual(found.bars, 1)
        self.assertEqual(found.variants, 1)
        self.assertIn("toploop", library.bank)

    def test_a_folder_becomes_a_beat_with_four_density_steps(self):
        for index in range(4):
            write_wav(os.path.join(self.dir, "bassline", f"{index}.wav"), BAR_SECONDS)
        library = self.load()
        self.assertEqual(len(library.presets), 1)
        self.assertEqual(library.presets[0].variants, 4)
        self.assertEqual(len(library.bank["bassline"]), 4)

    def test_every_density_step_picks_its_own_file(self):
        for index in range(4):
            write_wav(os.path.join(self.dir, "bassline", f"{index}.wav"), BAR_SECONDS)
        found = self.load().presets[0]
        picked = [found.variant_for(d) for d in (0.125, 0.375, 0.625, 0.875)]
        self.assertEqual(picked, [0, 1, 2, 3])

    def test_fewer_files_spread_over_the_four_steps(self):
        for name in ("a.wav", "b.wav"):
            write_wav(os.path.join(self.dir, "pad", name), BAR_SECONDS)
        found = self.load().presets[0]
        picked = [found.variant_for(d) for d in (0.125, 0.375, 0.625, 0.875)]
        self.assertEqual(picked, [0, 0, 1, 1])

    def test_a_single_file_stays_itself_at_every_density(self):
        write_wav(os.path.join(self.dir, "one.wav"), BAR_SECONDS)
        found = self.load().presets[0]
        self.assertEqual({found.variant_for(d) for d in (0.1, 0.4, 0.6, 0.9)}, {0})

    def test_variants_of_one_beat_have_the_same_length(self):
        write_wav(os.path.join(self.dir, "mix", "a.wav"), BAR_SECONDS)
        write_wav(os.path.join(self.dir, "mix", "b.wav"), BAR_SECONDS * 0.98)
        clips = self.load().bank["mix"]
        self.assertEqual(len({len(clip) for clip in clips}), 1)

    def test_two_bars_are_recognised(self):
        write_wav(os.path.join(self.dir, "long.wav"), BAR_SECONDS * 2)
        self.assertEqual(self.load().presets[0].bars, 2)

    def test_a_broken_file_is_reported_and_skipped(self):
        with open(os.path.join(self.dir, "kaputt.wav"), "wb") as handle:
            handle.write(b"nicht wirklich eine wav datei")
        write_wav(os.path.join(self.dir, "gut.wav"), BAR_SECONDS)
        library = self.load()
        self.assertEqual([item.id for item in library.presets], ["gut"])
        self.assertTrue(any("kaputt" in problem for problem in library.problems))

    def test_stereo_and_other_rates_are_accepted(self):
        write_wav(os.path.join(self.dir, "stereo.wav"), BAR_SECONDS,
                  samplerate=44100, channels=2)
        library = self.load()
        self.assertEqual(len(library.presets), 1)
        self.assertGreater(float(np.max(np.abs(library.bank["stereo"][0]))), 0.1)


class TestGrid(LoopFolder):
    def _length(self, seconds, bpm=BPM, meta=None):
        write_wav(os.path.join(self.dir, "beat.wav"), seconds)
        if meta is not None:
            import json

            with open(os.path.join(self.dir, "loops.json"), "w", encoding="utf-8") as f:
                json.dump({"beat": meta}, f)
        library = self.load(bpm)
        return len(library.bank["beat"][0]) / SR

    def test_a_loop_lands_exactly_on_the_bar(self):
        self.assertAlmostEqual(self._length(BAR_SECONDS), BAR_SECONDS, places=3)

    def test_a_slightly_off_recording_is_pulled_onto_the_grid(self):
        self.assertAlmostEqual(self._length(BAR_SECONDS * 1.04), BAR_SECONDS, places=3)

    def test_a_recording_at_another_tempo_is_fitted(self):
        """Zwei Takte bei 140 BPM muessen bei 128 BPM zwei Takte bleiben."""
        seconds = 2 * 4 * 60.0 / 140.0
        length = self._length(seconds, meta={"bpm": 140, "bars": 2})
        self.assertAlmostEqual(length, 2 * BAR_SECONDS, places=3)

    def test_the_metadata_file_sets_name_and_colour(self):
        write_wav(os.path.join(self.dir, "beat.wav"), BAR_SECONDS)
        import json

        with open(os.path.join(self.dir, "loops.json"), "w", encoding="utf-8") as f:
            json.dump({"beat": {"label": "MEIN BEAT", "short": "MB",
                                "colour": "#102030", "family": "bass"}}, f)
        found = self.load().presets[0]
        self.assertEqual(found.label, "MEIN BEAT")
        self.assertEqual(found.short, "MB")
        self.assertEqual(found.colour, (0x30, 0x20, 0x10))   # BGR
        self.assertEqual(found.family, "bass")

    def test_clips_stay_inside_the_range(self):
        write_wav(os.path.join(self.dir, "beat.wav"), BAR_SECONDS)
        clip = self.load().bank["beat"][0]
        self.assertLessEqual(float(np.max(np.abs(clip))), 1.0)
        self.assertFalse(bool(np.any(np.isnan(clip))))


class TestPlayback(LoopFolder):
    def _engine(self, gain=0.9, density=0.5):
        library = self.load()
        preset_library.use_presets(library.presets)
        audio = AudioEngine(AudioConfig(samplerate=SR, blocksize=256, bpm=BPM))
        audio.bank.update(library.bank)
        audio.set_slot("one", library.presets[0].id, gain=gain, cutoff=0.9,
                       density=density)
        return audio

    def test_a_loop_sounds(self):
        write_wav(os.path.join(self.dir, "beat.wav"), BAR_SECONDS)
        block = self._engine().render_offline(BAR_SECONDS * 1.5)
        self.assertGreater(float(np.max(np.abs(block))), 0.01)
        self.assertFalse(bool(np.any(np.isnan(block))))

    def test_a_loop_starts_at_once_instead_of_waiting_for_the_bar(self):
        """Wer einen Loop greift, will ihn hoeren - nicht erst im naechsten Takt."""
        write_wav(os.path.join(self.dir, "beat.wav"), BAR_SECONDS)
        audio = self._engine()
        audio.render_offline(BAR_SECONDS * 0.5)     # mitten in den Takt laufen
        audio.set_slot("two", "beat", gain=0.9, cutoff=0.9, density=0.5)
        block = audio.render_offline(0.1)
        self.assertGreater(float(np.max(np.abs(block))), 0.01)

    def test_a_loop_keeps_repeating(self):
        write_wav(os.path.join(self.dir, "beat.wav"), BAR_SECONDS)
        audio = self._engine()
        audio.render_offline(BAR_SECONDS * 2)
        tail = audio.render_offline(BAR_SECONDS)
        self.assertGreater(float(np.max(np.abs(tail))), 0.01)

    def test_the_density_switches_the_variant(self):
        write_wav(os.path.join(self.dir, "beat", "a.wav"), BAR_SECONDS, freq=220.0)
        write_wav(os.path.join(self.dir, "beat", "b.wav"), BAR_SECONDS, freq=880.0)
        audio = self._engine(density=0.1)
        audio.render_offline(BAR_SECONDS)
        low = audio.layers["one"].voices[0].buffer
        audio.set_slot("one", "beat", gain=0.9, cutoff=0.9, density=0.9)
        audio.render_offline(BAR_SECONDS)
        high = audio.layers["one"].voices[0].buffer
        self.assertIsNot(low, high)

    def test_a_quiet_loop_stays_in_time(self):
        """Leise heisst nicht angehalten - sonst setzt er versetzt wieder ein."""
        write_wav(os.path.join(self.dir, "beat.wav"), BAR_SECONDS)
        audio = self._engine(gain=0.0)
        audio.render_offline(BAR_SECONDS * 1.5)
        self.assertTrue(audio.layers["one"].voices)

    def test_loops_and_builtin_beats_can_sound_together(self):
        write_wav(os.path.join(self.dir, "beat.wav"), BAR_SECONDS)
        library = self.load()
        preset_library.use_presets(library.presets + preset_library.BUILTIN_PRESETS)
        audio = AudioEngine(AudioConfig(samplerate=SR, blocksize=256, bpm=BPM))
        audio.bank.update(library.bank)
        audio.set_slot("loop", "beat", gain=0.8, cutoff=0.9, density=0.5)
        audio.set_slot("steps", "kick_four", gain=0.8, cutoff=0.9, density=0.5)
        block = audio.render_offline(BAR_SECONDS)
        self.assertGreater(float(np.max(np.abs(block))), 0.01)
        self.assertLessEqual(float(np.max(np.abs(block))), 1.0)


class TestLibrarySwap(LoopFolder):
    def test_own_loops_replace_the_builtin_library(self):
        from src.app import install_library
        from src.config import Config

        write_wav(os.path.join(self.dir, "beat.wav"), BAR_SECONDS)
        config = Config()
        config.audio.samplerate = SR
        config.audio.bpm = BPM
        original = loops.LOOPS_DIR
        loops.LOOPS_DIR = self.dir
        try:
            note = install_library(config)
        finally:
            loops.LOOPS_DIR = original
        self.assertIn("eigene Loops", note)
        self.assertEqual(preset_library.PRESET_IDS, ["beat"])

    def test_builtin_mode_keeps_the_builtin_library(self):
        from src.app import install_library
        from src.config import Config

        write_wav(os.path.join(self.dir, "beat.wav"), BAR_SECONDS)
        config = Config()
        config.audio.library = "builtin"
        note = install_library(config)
        self.assertIn("eingebaute", note)
        self.assertEqual(len(preset_library.PRESETS),
                         len(preset_library.BUILTIN_PRESETS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
