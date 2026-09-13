"""Tests fuer Haltungserkennung, Handverfolgung und die Beat Schmiede."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import Driver, hand_at, test_config  # noqa: E402
from src.config import Config, MappingConfig, TrackConfig  # noqa: E402
from src.mapping import gestures  # noqa: E402
from src.mapping import tokens as tk  # noqa: E402
from src.mapping.macros import MacroDetector  # noqa: E402
from src.mapping.presets import PRESET_IDS  # noqa: E402
from src.vision.hand_tracker import HandTracker  # noqa: E402


class TestGestures(unittest.TestCase):
    def setUp(self):
        self.cfg = MappingConfig()

    def test_finger_count_matches_the_pose(self):
        for count in range(5):
            with self.subTest(count=count):
                hand = hand_at(0.5, 0.5, fingers=count)
                self.assertEqual(gestures.finger_count(hand, self.cfg), count)

    def test_closed_hand_is_a_fist(self):
        self.assertTrue(gestures.is_fist(hand_at(0.5, 0.5, fingers=0), self.cfg))
        self.assertFalse(gestures.is_fist(hand_at(0.5, 0.5, fingers=1), self.cfg))

    def test_density_rises_with_the_finger_count(self):
        values = [gestures.density_from_count(n) for n in (1, 2, 3, 4)]
        self.assertEqual(values, sorted(values))
        self.assertEqual(len(set(values)), 4)

    def test_upright_hand_sits_in_the_middle_of_the_volume_range(self):
        self.assertAlmostEqual(
            gestures.lean(hand_at(0.5, 0.5), self.cfg), 0.5, delta=0.02
        )

    def test_leaning_outward_is_louder_for_both_hands(self):
        # Nach aussen kippen heisst fuer die rechte Hand nach links im
        # ungespiegelten Bild und fuer die linke Hand nach rechts.
        right_out = gestures.lean(hand_at(0.5, 0.5, "right", tilt=-50), self.cfg)
        right_in = gestures.lean(hand_at(0.5, 0.5, "right", tilt=50), self.cfg)
        left_out = gestures.lean(hand_at(0.5, 0.5, "left", tilt=50), self.cfg)
        left_in = gestures.lean(hand_at(0.5, 0.5, "left", tilt=-50), self.cfg)
        self.assertGreater(right_out, 0.9)
        self.assertLess(right_in, 0.1)
        self.assertAlmostEqual(right_out, left_out, delta=0.02)
        self.assertAlmostEqual(right_in, left_in, delta=0.02)

    def test_posture_values_ignore_where_the_hand_is(self):
        """Der Kern der Steuerung: die Hand darf stehen bleiben."""
        near = hand_at(0.2, 0.8, tilt=-30, thumb=0.6, fingers=3)
        far = hand_at(0.9, 0.15, tilt=-30, thumb=0.6, fingers=3)
        self.assertAlmostEqual(
            gestures.lean(near, self.cfg), gestures.lean(far, self.cfg), delta=0.01
        )
        self.assertAlmostEqual(
            gestures.thumb_open(near, self.cfg),
            gestures.thumb_open(far, self.cfg),
            delta=0.01,
        )
        self.assertEqual(
            gestures.finger_count(near, self.cfg), gestures.finger_count(far, self.cfg)
        )

    def test_thumb_opens_continuously(self):
        values = [
            gestures.thumb_open(hand_at(0.5, 0.5, thumb=amount), self.cfg)
            for amount in (0.0, 0.5, 1.0)
        ]
        self.assertEqual(values, sorted(values))
        self.assertLess(values[0], 0.1)
        self.assertGreater(values[-1], 0.9)

    def test_turning_the_wrist_flips_the_palm(self):
        for handedness in ("right", "left"):
            with self.subTest(handedness=handedness):
                front = hand_at(0.5, 0.5, handedness, roll=1.0)
                back = hand_at(0.5, 0.5, handedness, roll=-1.0)
                edge = hand_at(0.5, 0.5, handedness, roll=0.05)
                self.assertGreater(gestures.palm_facing(front), 0.8)
                self.assertLess(gestures.palm_facing(back), -0.8)
                self.assertAlmostEqual(gestures.palm_facing(edge), 0.0, delta=0.2)

    def test_palm_direction_survives_tilting(self):
        turned = hand_at(0.5, 0.5, tilt=45, roll=-1.0)
        self.assertLess(gestures.palm_facing(turned), -0.8)

    def test_pinch_is_not_a_fist(self):
        self.assertTrue(gestures.is_pinch(hand_at(0.5, 0.5, pinch=True), self.cfg))
        self.assertFalse(gestures.is_pinch(hand_at(0.5, 0.5, fingers=0), self.cfg))
        self.assertFalse(gestures.is_pinch(hand_at(0.5, 0.5, fingers=4), self.cfg))


class TestHandTracker(unittest.TestCase):
    def test_id_stays_the_same_while_moving(self):
        tracker = HandTracker(TrackConfig())
        first = tracker.update([hand_at(0.2, 0.5)])[0]
        for step in range(1, 12):
            tracks = tracker.update([hand_at(0.2 + step * 0.03, 0.5)])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].id, first.id)

    def test_two_hands_keep_separate_ids(self):
        tracker = HandTracker(TrackConfig())
        tracker.update([hand_at(0.2, 0.5), hand_at(0.8, 0.5)])
        tracks = tracker.update([hand_at(0.23, 0.5), hand_at(0.77, 0.5)])
        self.assertEqual(len({t.id for t in tracks}), 2)

    def test_track_disappears_after_enough_missing_frames(self):
        cfg = TrackConfig()
        tracker = HandTracker(cfg)
        tracker.update([hand_at(0.5, 0.5)])
        for _ in range(cfg.max_missing_frames + 2):
            tracks = tracker.update([])
        self.assertEqual(tracks, [])

    def test_new_hand_gets_a_new_id(self):
        tracker = HandTracker(TrackConfig())
        first = tracker.update([hand_at(0.2, 0.2)])[0]
        second = tracker.update([hand_at(0.2, 0.2), hand_at(0.9, 0.9)])
        self.assertIn(first.id, {t.id for t in second})
        self.assertEqual(len({t.id for t in second}), 2)


class TestBrowser(unittest.TestCase):
    def test_library_offers_every_preset(self):
        driver = Driver()
        entries = driver.forge.library_entries()
        self.assertEqual([e.preset_ids[0] for e in entries], PRESET_IDS)

    def test_hand_height_picks_the_nearest_slot(self):
        cfg = test_config().forge
        driver = Driver()
        driver.place("a", 0.1, cfg.library_top)
        driver.aim("a", 0.1, cfg.library_top)
        driver.step(5)
        self.assertEqual(driver.browser("a").index, 0)

    def test_the_selection_follows_the_grabbing_fingers(self):
        """Gezielt wird mit den Fingern, nicht mit dem Handteller.

        Die getrackte Handmitte liegt rund eine Fachhoehe unter dem
        Punkt, an dem Daumen und Zeigefinger zugreifen. Wuerde die Spalte
        nach der Mitte auswaehlen, bekaeme man das Fach unter dem, auf
        das man sichtbar zeigt.
        """
        driver = Driver()
        column = driver.forge.library_column
        driver.place("a", 0.1, 0.5)
        driver.aim("a", 0.1, column.slot_y(2))
        driver.step(4)
        self.assertEqual(driver.browser("a").index, 2)
        self.assertGreater(driver.hands["a"]["y"], column.slot_y(2))

    def test_moving_down_selects_a_later_slot(self):
        cfg = test_config().forge
        driver = Driver()
        count = len(driver.forge.library_entries())
        driver.place("a", 0.1, cfg.library_top)
        driver.step(5)
        driver.aim("a", 0.1, cfg.library_bottom, frames=16)
        self.assertEqual(driver.browser("a").index, count - 1)

    def test_a_small_wobble_does_not_change_the_selection(self):
        """Die Auswahl muss stehen bleiben, sonst greift man daneben."""
        driver = Driver()
        column = driver.forge.library_column
        driver.place("a", 0.1, column.slot_y(3))
        driver.aim("a", 0.1, column.slot_y(3))
        driver.step(8)
        chosen = driver.browser("a").index
        self.assertEqual(chosen, 3)
        for offset in (0.35, -0.35, 0.3, -0.3):
            driver.aim("a", 0.1, column.slot_y(3) + column.pitch * offset, frames=3)
            self.assertEqual(driver.browser("a").index, chosen)

    def test_a_full_slot_step_does_change_the_selection(self):
        driver = Driver()
        column = driver.forge.library_column
        driver.place("a", 0.1, column.slot_y(2))
        driver.aim("a", 0.1, column.slot_y(2))
        driver.step(8)
        driver.aim("a", 0.1, column.slot_y(4), frames=10)
        self.assertEqual(driver.browser("a").index, 4)

    def test_the_grip_locks_the_slot_it_closes_on(self):
        """Was beim Zugreifen unter den Fingern liegt, wird auch genommen.

        Frueher zeigte die Spalte nach dem Glaetten noch auf das Fach
        davor. Der Griff schnappt deshalb ohne Sperre auf die Hoehe der
        Finger und haelt die Auswahl fest, solange er zugeht.
        """
        config = test_config()
        config.forge.select_settle = 10.0   # Sperre bleibt voll aktiv
        driver = Driver(config)
        column = driver.forge.library_column
        driver.place("a", 0.1, column.slot_y(3))
        driver.aim("a", 0.1, column.slot_y(3))
        driver.step(6)
        driver.aim("a", 0.1, column.slot_y(3) + column.pitch * 0.6, frames=6)
        self.assertEqual(driver.browser("a").index, 3)   # Sperre haelt noch
        driver.set("a", pinch=True)
        driver.step()
        self.assertEqual(driver.browser("a").index, 4)

    def test_the_library_does_not_shift_when_the_rack_fills_up(self):
        """Frueher lagen beide in einer Liste - jede Ablage hat alles verschoben."""
        driver = Driver()
        before = [e.preset_ids[0] for e in driver.forge.library_entries()]
        driver.grab("a", "kick_four")
        driver.step(8)
        driver.hold("a")
        driver.put_away("a")
        driver.settle(20)
        after = [e.preset_ids[0] for e in driver.forge.library_entries()]
        self.assertEqual(before, after)

    def test_a_hand_outside_the_library_zone_does_not_browse(self):
        cfg = test_config().forge
        driver = Driver()
        driver.place("a", cfg.library_capture_x + 0.1, 0.5)
        driver.step(10)
        self.assertIsNone(driver.browser("a"))

    def test_pinching_takes_the_selected_loop(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        self.assertEqual(token.state, tk.LIVE)
        self.assertEqual(token.preset_ids, ["kick_four"])
        self.assertIsNotNone(token.hand_id)

    def test_a_preset_stays_available_after_being_taken(self):
        """Ein Loop verschwindet nicht aus der Bibliothek, wenn ihn jemand nimmt."""
        driver = Driver()
        driver.grab("a", "kick_four")
        driver.place("b", 0.7, 0.5, handedness="left")
        driver.grab("b", "kick_four", handedness="left")
        self.assertEqual(len(driver.tokens), 2)
        self.assertEqual([t.preset_ids for t in driver.tokens],
                         [["kick_four"], ["kick_four"]])

    def test_the_browser_stops_handing_out_above_the_limit(self):
        config = test_config()
        config.forge.max_tokens = 1
        driver = Driver(config)
        driver.grab("a", "kick_four")
        driver.place("b", 0.7, 0.5, handedness="left")
        driver.grab("b", "clap", handedness="left")
        self.assertEqual(len(driver.tokens), 1)

    def test_a_hand_that_already_holds_a_loop_takes_nothing(self):
        driver = Driver()
        driver.grab("a", "kick_four")
        driver.set("a", pinch=True)
        driver.step(driver.config.forge.take_frames + 6)
        self.assertEqual(len(driver.tokens), 1)


class TestShaping(unittest.TestCase):
    def test_fingers_set_the_density(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        seen = []
        for count in (1, 2, 3, 4):
            driver.set("a", fingers=count)
            driver.step(6)
            seen.append(token.recipes[0].density)
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(len(set(seen)), 4)

    def test_leaning_sets_the_volume_and_the_thumb_the_tone(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.set("a", tilt=50, thumb=0.0)
        driver.step(6)
        quiet, dark = token.recipes[0].volume, token.recipes[0].cutoff
        driver.set("a", tilt=-50, thumb=1.0)
        driver.step(6)
        self.assertGreater(token.recipes[0].volume, quiet + 0.5)
        self.assertGreater(token.recipes[0].cutoff, dark + 0.5)

    def test_values_stay_put_while_the_hand_wanders(self):
        """Das Ziel der Steuerung: nur die Haltung zaehlt, nie der Ort."""
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.set("a", fingers=3, tilt=-30, thumb=0.5)
        driver.step(8)
        before = (token.recipes[0].volume, token.recipes[0].cutoff,
                  token.recipes[0].density)
        driver.move("a", 0.85, 0.15, frames=16)
        driver.move("a", 0.15, 0.75, frames=16)
        after = (token.recipes[0].volume, token.recipes[0].cutoff,
                 token.recipes[0].density)
        for value, other in zip(before, after):
            self.assertAlmostEqual(value, other, delta=0.01)

    def test_a_fist_mutes_the_loop(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.set("a", fingers=0)
        driver.step(6)
        self.assertTrue(token.muted)
        self.assertTrue(all(s.gain == 0.0 for s in driver.snapshot.slots))
        driver.set("a", fingers=2)
        driver.step(6)
        self.assertFalse(token.muted)

    def test_a_single_shaky_frame_does_not_switch_the_density(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.set("a", fingers=4)
        driver.step(6)
        steady = token.recipes[0].density
        driver.set("a", fingers=1)
        driver.step(1)
        self.assertEqual(token.recipes[0].density, steady)


class TestHold(unittest.TestCase):
    def test_turning_the_wrist_holds_the_values(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.set("a", fingers=3, tilt=-40, thumb=0.8)
        driver.step(8)
        shaped = (token.recipes[0].volume, token.recipes[0].cutoff,
                  token.recipes[0].density)

        driver.hold("a")
        self.assertEqual(token.state, tk.FROZEN)
        held = (token.recipes[0].volume, token.recipes[0].cutoff,
                token.recipes[0].density)
        for value, other in zip(shaped, held):
            self.assertAlmostEqual(value, other, delta=0.02)

        driver.set("a", fingers=1, tilt=40, thumb=0.0)
        driver.step(10)
        after = (token.recipes[0].volume, token.recipes[0].cutoff,
                 token.recipes[0].density)
        self.assertEqual(held, after)

    def test_turning_back_lets_the_loop_follow_again(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.step(6)
        driver.hold("a")
        self.assertEqual(token.state, tk.FROZEN)
        driver.release_hold("a")
        self.assertEqual(token.state, tk.LIVE)

    def test_a_short_wobble_does_not_hold_the_loop(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.step(6)
        driver.set("a", roll=-1.0)
        driver.step(1)
        driver.set("a", roll=1.0)
        driver.step(4)
        self.assertEqual(token.state, tk.LIVE)

    def test_a_held_loop_keeps_sounding(self):
        driver = Driver()
        driver.grab("a", "kick_four")
        driver.step(6)
        driver.hold("a")
        self.assertEqual(len(driver.snapshot.slots), 1)
        self.assertGreater(driver.snapshot.slots[0].gain, 0.0)


class TestFusion(unittest.TestCase):
    def _two_held(self, driver):
        left = driver.grab("a", "kick_four")
        driver.set("a", fingers=3, tilt=-30)
        driver.step(8)
        driver.hold("a")
        driver.place("b", 0.70, 0.50, handedness="left")
        right = driver.grab("b", "bass_sub", handedness="left")
        driver.set("b", fingers=2, tilt=20)
        driver.step(8)
        driver.hold("b")
        return left, right

    def test_hands_together_fuse_the_two_loops(self):
        driver = Driver()
        left, _right = self._two_held(driver)
        driver.move("a", 0.46, 0.50, frames=6)
        driver.move("b", 0.54, 0.50, frames=6)
        driver.step(driver.config.forge.fuse_frames + 2)
        driver.step(driver.config.forge.fuse_duration + 2)

        self.assertEqual(len(driver.tokens), 1)
        self.assertEqual(driver.tokens[0].preset_ids, ["kick_four", "bass_sub"])
        self.assertEqual(driver.tokens[0].state, tk.FROZEN)
        self.assertIs(driver.tokens[0], left)

    def test_fusion_happens_quickly(self):
        """Zusammenfuehren soll sich sofort anfuehlen, nicht nach Warten."""
        driver = Driver()
        self._two_held(driver)
        driver.move("a", 0.46, 0.50, frames=6)
        driver.move("b", 0.54, 0.50, frames=6)
        for _ in range(8):
            driver.step()
            if len(driver.tokens) == 1:
                break
        self.assertEqual(len(driver.tokens), 1)

    def test_fusion_keeps_the_values_of_both_loops(self):
        driver = Driver()
        left, right = self._two_held(driver)
        before = [
            (r.preset_id, r.volume, r.cutoff, r.density)
            for r in (left.recipes[0], right.recipes[0])
        ]
        driver.move("a", 0.46, 0.50, frames=6)
        driver.move("b", 0.54, 0.50, frames=6)
        driver.step(driver.config.forge.fuse_frames + driver.config.forge.fuse_duration + 4)

        after = [(r.preset_id, r.volume, r.cutoff, r.density)
                 for r in driver.tokens[0].recipes]
        self.assertEqual(before, after)

    def test_the_approach_is_visible_before_it_snaps(self):
        config = test_config()
        # Deutlich laenger als der Weg der beiden Haende: sonst haengt der
        # Test daran, ab welchem Frame die Haende einander nahe kommen.
        config.forge.fuse_frames = 40
        driver = Driver(config)
        self._two_held(driver)
        driver.move("a", 0.46, 0.50, frames=6)
        driver.move("b", 0.54, 0.50, frames=6)
        driver.step(2)
        self.assertEqual(len(driver.tokens), 2)
        self.assertGreater(max(t.approach for t in driver.tokens), 0.0)

    def test_loops_that_still_follow_the_hand_do_not_fuse(self):
        driver = Driver()
        driver.grab("a", "kick_four")
        driver.place("b", 0.70, 0.50, handedness="left")
        driver.grab("b", "bass_sub", handedness="left")
        driver.move("a", 0.47, 0.50, frames=6)
        driver.move("b", 0.53, 0.50, frames=6)
        driver.step(driver.config.forge.fuse_frames + 4)
        self.assertEqual(len(driver.tokens), 2)

    def test_fusion_stops_at_the_recipe_limit(self):
        config = test_config()
        config.forge.max_recipes = 1
        driver = Driver(config)
        self._two_held(driver)
        driver.move("a", 0.46, 0.50, frames=6)
        driver.move("b", 0.54, 0.50, frames=6)
        driver.step(config.forge.fuse_frames + 4)
        self.assertEqual(len(driver.tokens), 2)

    def test_a_fused_loop_keeps_its_mix_and_gets_a_master(self):
        """Aufgetaut bleibt die Mischung stehen, die Neigung blendet nur aus."""
        driver = Driver()
        self._two_held(driver)
        driver.move("a", 0.46, 0.50, frames=6)
        driver.move("b", 0.54, 0.50, frames=6)
        driver.step(driver.config.forge.fuse_frames + driver.config.forge.fuse_duration + 4)
        fused = driver.tokens[0]
        mix = [r.volume for r in fused.recipes]

        driver.release_hold("a")
        self.assertEqual(fused.state, tk.LIVE)
        self.assertEqual([r.volume for r in fused.recipes], mix)
        self.assertAlmostEqual(fused.master, 1.0, delta=0.05)

        loud = sorted(s.gain for s in driver.snapshot.slots)
        driver.set("a", tilt=50)          # nach innen kippen blendet aus
        driver.step(8)
        self.assertLess(fused.master, 0.1)
        quiet = sorted(s.gain for s in driver.snapshot.slots)
        for before, after in zip(loud, quiet):
            self.assertLess(after, before * 0.2)

        driver.set("a", tilt=0)
        driver.step(8)
        driver.hold("a")
        self.assertEqual(fused.state, tk.FROZEN)
        self.assertEqual(fused.master, 1.0)


class TestRack(unittest.TestCase):
    def test_a_pinch_puts_the_loop_into_the_rack(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.step(8)
        driver.hold("a")
        driver.put_away("a")
        self.assertEqual(token.state, tk.FLYING)
        driver.settle()
        self.assertEqual(token.state, tk.PLACED)
        self.assertIsNone(token.hand_id)

    def test_a_parked_loop_keeps_sounding(self):
        driver = Driver()
        driver.grab("a", "kick_four")
        driver.step(8)
        driver.hold("a")
        driver.put_away("a")
        snapshot = driver.settle()
        self.assertEqual(len(snapshot.slots), 1)
        self.assertGreater(snapshot.slots[0].gain, 0.0)

    def test_a_loop_that_still_follows_the_hand_holds_on_the_way_out(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.step(8)
        driver.put_away("a")
        driver.settle()
        self.assertEqual(token.state, tk.PLACED)
        self.assertGreater(token.recipes[0].volume, 0.0)

    def test_parked_loops_line_up_inside_the_frame(self):
        driver = Driver()
        for preset_id in ("kick_four", "clap", "hats_closed"):
            driver.grab("a", preset_id)
            driver.step(8)
            driver.hold("a")
            driver.put_away("a")
            driver.settle(20)
            driver.move("a", 0.35, 0.45, frames=4)
        parked = driver.forge.parked
        self.assertEqual(len(parked), 3)
        ys = [t.y for t in parked]
        self.assertEqual(ys, sorted(ys))
        for index, token in enumerate(parked):
            self.assertEqual(token.rack_index, index)
            self.assertAlmostEqual(token.x, driver.config.forge.rack_x, places=3)
            self.assertTrue(0.0 < token.y < 1.0)

    def test_a_freed_slot_does_not_move_the_others(self):
        """Ein Fach bleibt, wo es ist - sonst muesste man jedes Mal suchen."""
        driver = Driver()
        parked = []
        for preset_id in ("kick_four", "clap", "hats_closed"):
            token = driver.grab("a", preset_id)
            driver.step(8)
            driver.hold("a")
            driver.put_away("a")
            driver.settle(20)
            driver.move("a", 0.35, 0.45, frames=4)
            parked.append(token)

        places = {token.id: (token.rack_index, token.y) for token in parked}
        driver.delete_from_rack("a", parked[0].rack_index)
        driver.settle(4)
        self.assertNotIn(parked[0], driver.tokens)
        for token in parked[1:]:
            self.assertEqual((token.rack_index, token.y), places[token.id])

    def test_a_long_pinch_throws_the_loop_away(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.step(8)
        driver.hold("a")
        driver.throw_away("a")
        self.assertNotIn(token, driver.tokens)
        self.assertEqual(driver.tokens, [])

    def test_moving_the_hand_keeps_the_loop_in_the_hand(self):
        """Der Kern der Umstellung: wandern legt nichts mehr ab."""
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.step(8)
        driver.hold("a")
        driver.step(driver.config.forge.min_hold_frames + 2)
        driver.move("a", 0.85, 0.15, frames=6)
        driver.move("a", 0.15, 0.80, frames=6)
        self.assertEqual(token.state, tk.FROZEN)
        self.assertIsNotNone(token.hand_id)

    def test_the_grabbing_pinch_does_not_throw_the_loop_away(self):
        """Ein langer Pinch nimmt den Loop und behaelt ihn auch."""
        driver = Driver()
        driver.place("a", 0.1, 0.45)
        driver.step()
        y = driver.forge.library_column.slot_y(driver.entry_index("kick_four"))
        driver.move("a", 0.1, y, frames=10)
        driver.set("a", pinch=True)
        driver.step(driver.config.forge.discard_frames
                    + driver.config.forge.min_hold_frames + 10)
        self.assertEqual(len(driver.tokens), 1)
        self.assertTrue(driver.tokens[0].held)

    def test_a_pinch_that_opens_at_once_keeps_the_loop(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.step(8)
        driver.hold("a")
        driver.step(driver.config.forge.min_hold_frames + 2)
        driver.set("a", pinch=True)
        driver.step()
        driver.set("a", pinch=False)
        driver.step(2)
        self.assertEqual(token.state, tk.FROZEN)
        self.assertIsNotNone(token.hand_id)

    def test_a_lost_hand_leaves_the_loop_in_the_rack(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.step(8)
        driver.hold("a")
        driver.remove("a")
        driver.step(driver.config.forge.release_frames + 6)
        self.assertEqual(token.state, tk.PLACED)
        self.assertIsNone(token.hand_id)

    def test_a_lost_hand_with_a_full_rack_does_not_leave_a_ghost(self):
        """Sonst klingt ein Loop weiter, den niemand mehr erreichen kann."""
        config = test_config()
        config.forge.max_parked = 1
        driver = Driver(config)
        driver.grab("a", "kick_four")
        driver.step(8)
        driver.hold("a")
        driver.put_away("a")
        driver.settle(20)
        driver.move("a", 0.35, 0.45, frames=4)

        ghost = driver.grab("a", "clap")
        driver.step(8)
        driver.remove("a")
        driver.step(driver.config.forge.release_frames + 6)
        self.assertNotIn(ghost, driver.tokens)
        self.assertEqual(len(driver.forge.parked), 1)

    def test_a_parked_loop_can_be_taken_back(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.set("a", fingers=4, tilt=-40)
        driver.step(8)
        driver.hold("a")
        shaped = token.recipes[0].volume
        driver.put_away("a")
        driver.settle(20)

        slot = driver.rack_index(token.id)
        self.assertEqual(driver.forge.rack_entries()[slot].kind, "loop")
        driver.take_from_rack("a", slot)
        self.assertEqual(token.state, tk.FROZEN)
        self.assertIsNotNone(token.hand_id)
        self.assertEqual(token.recipes[0].volume, shaped)

    def test_a_long_pinch_at_the_rack_deletes_the_loop(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.step(8)
        driver.hold("a")
        driver.put_away("a")
        driver.settle(20)

        driver.delete_from_rack("a", driver.rack_index(token.id))
        self.assertNotIn(token, driver.tokens)
        self.assertEqual(driver.forge.parked, [])

    def test_the_rack_grip_shows_how_close_the_deletion_is(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.step(8)
        driver.hold("a")
        driver.put_away("a")
        driver.settle(20)

        cfg = driver.config.forge
        driver.reach("a", driver.forge.rack_column, driver.rack_index(token.id),
                     cfg.rack_capture_x + 0.02)
        driver.set("a", pinch=True)
        driver.step(cfg.discard_frames - 4)
        self.assertIn(token, driver.tokens)
        self.assertGreater(driver.browser("a").delete_progress, 0.3)

    def test_a_hand_between_the_columns_browses_nothing(self):
        driver = Driver()
        driver.grab("a", "kick_four")
        driver.step(8)
        driver.hold("a")
        driver.put_away("a")
        driver.settle(20)
        driver.move("a", 0.5, 0.5, frames=8)
        self.assertIsNone(driver.browser("a"))

    def test_the_rack_does_not_overflow(self):
        config = test_config()
        config.forge.max_parked = 1
        driver = Driver(config)
        first = driver.grab("a", "kick_four")
        driver.step(8)
        driver.hold("a")
        driver.put_away("a")
        driver.settle(20)
        driver.move("a", 0.35, 0.45, frames=4)

        second = driver.grab("a", "clap")
        driver.step(8)
        driver.hold("a")
        driver.put_away("a")
        driver.settle(20)
        self.assertEqual(first.state, tk.PLACED)
        self.assertEqual(second.state, tk.FROZEN)
        self.assertTrue(second.held)


class TestEngine(unittest.TestCase):
    def test_empty_frame_has_no_loops_but_a_full_library(self):
        driver = Driver()
        snapshot = driver.step()
        self.assertEqual(snapshot.tokens, [])
        self.assertEqual(snapshot.slots, [])
        self.assertEqual(len(snapshot.library), len(PRESET_IDS))
        self.assertEqual(snapshot.rack, [None] * driver.config.forge.max_parked)
        self.assertEqual(snapshot.hands, [])

    def test_hands_reach_the_display(self):
        driver = Driver()
        driver.place("a", 0.35, 0.5)
        snapshot = driver.step(3)
        self.assertEqual(len(snapshot.hands), 1)
        self.assertFalse(snapshot.hands[0].holding)
        self.assertIsNotNone(snapshot.hands[0].landmarks)

    def test_changing_players_clears_the_table(self):
        driver = Driver()
        driver.grab("a", "kick_four")
        self.assertEqual(len(driver.tokens), 1)
        driver.engine.set_performers(1)
        self.assertEqual(driver.tokens, [])

    def test_shuffle_keeps_every_preset(self):
        driver = Driver()
        before = {e.preset_ids[0] for e in driver.forge.library_entries()}
        driver.engine.shuffle_library()
        self.assertEqual(
            {e.preset_ids[0] for e in driver.forge.library_entries()}, before
        )

    def test_place_all_puts_everything_into_the_rack(self):
        driver = Driver()
        token = driver.grab("a", "kick_four")
        driver.step(8)
        driver.engine.place_all()
        self.assertEqual(token.state, tk.PLACED)


class TestMacros(unittest.TestCase):
    def test_open_hands_then_fists_give_riser_and_drop(self):
        cfg = MappingConfig()
        detector = MacroDetector(cfg)
        for _ in range(cfg.riser_frames + 40):
            state = detector.update([1.0, 0.95, 0.9])
        self.assertGreater(state.riser, 0.75)
        for _ in range(cfg.drop_frames + 2):
            state = detector.update([0.05, 0.0, 0.1])
        self.assertGreater(state.drop, 0.5)

    def test_no_riser_with_half_open_hands(self):
        detector = MacroDetector(MappingConfig())
        for _ in range(40):
            state = detector.update([0.5, 0.55, 0.6])
        self.assertLess(state.riser, 0.05)

    def test_opening_the_hand_feeds_the_macro(self):
        """Die Haende bleiben, wo sie sind - nur die Haltung zaehlt."""
        driver = Driver()
        driver.grab("a", "kick_four", fingers=4, thumb=1.0)
        driver.set("a", fingers=4, thumb=1.0)
        snapshot = driver.step(6)
        self.assertGreater(snapshot.macros.group_open, 0.9)
        driver.set("a", fingers=0, thumb=0.0)
        snapshot = driver.step(8)
        self.assertLess(snapshot.macros.group_open, 0.1)


class TestLobby(unittest.TestCase):
    def test_hand_over_the_button_starts_the_party(self):
        from src.ui.lobby import Lobby

        lobby = Lobby(Config(), dwell_frames=5)
        rx, ry, rw, rh = lobby.rect
        x, y = rx + rw / 2, ry + rh / 2

        result = None
        for _ in range(8):
            result = lobby.update([hand_at(x, y)]) or result
        self.assertEqual(result, 1)

    def test_keyboard_start(self):
        from src.ui.lobby import Lobby

        lobby = Lobby(Config())
        self.assertIsNone(lobby.handle_key(ord("x")))
        self.assertEqual(lobby.handle_key(13), 1)

    def test_progress_decays_without_a_hand(self):
        from src.ui.lobby import Lobby

        lobby = Lobby(Config(), dwell_frames=20)
        rx, ry, rw, rh = lobby.rect
        x, y = rx + rw / 2, ry + rh / 2
        for _ in range(5):
            lobby.update([hand_at(x, y)])
        before = lobby.progress
        for _ in range(10):
            lobby.update([])
        self.assertLess(lobby.progress, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSurface(unittest.TestCase):
    """Die Oberflaeche einmal komplett zeichnen.

    Gerendert wird ohne Fenster in ein Testbild. Der Test prueft keine
    Pixel - er faengt ab, dass ein Bildschirm ueberhaupt nicht mehr
    durchlaeuft, etwa weil eine Zeichenfunktion umgebaut wurde.
    """

    def setUp(self):
        import numpy as np

        from src.ui.display import Display
        from src.ui.lobby import Lobby, countdown_overlay

        self.np = np
        self.Display = Display
        self.Lobby = Lobby
        self.countdown_overlay = countdown_overlay
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8)

    def _busy_snapshot(self):
        """Ein Bild mit allem drin: Regal, Karte in der Hand, Browser."""
        driver = Driver(test_config(2))
        for preset_id in ("kick_four", "hats_closed"):
            driver.grab("a", preset_id)
            driver.step(6)
            driver.hold("a")
            driver.put_away("a")
            driver.settle(20)
        driver.grab("a", "chord_stab")
        driver.place("b", 0.7, 0.5, handedness="left")
        driver.step(8)
        return driver, driver.step(2)

    def test_the_stage_view_renders(self):
        driver, snapshot = self._busy_snapshot()
        canvas = self.Display(driver.config).render(self.image, snapshot, "LIVE")
        self.assertEqual(canvas.shape, self.image.shape)
        self.assertGreater(int(canvas.max()), 0)

    def test_the_stage_view_renders_while_a_loop_is_being_placed(self):
        driver, _ = self._busy_snapshot()
        driver.set("a", pinch=True)
        snapshot = driver.step(driver.config.forge.place_frames + 4)
        canvas = self.Display(driver.config).render(self.image, snapshot, "LIVE")
        self.assertGreater(int(canvas.max()), 0)

    def test_six_hands_get_cards_that_do_not_overlap(self):
        driver = Driver(test_config(3))
        for index in range(6):
            name = f"h{index}"
            driver.place(name, 0.14, 0.30 + 0.09 * index,
                         handedness="right" if index % 2 else "left")
            driver.grab(name, PRESET_IDS[index])
            driver.move(name, 0.40 + 0.06 * index, 0.32 + 0.07 * index, frames=4)
        snapshot = driver.step(4)
        self.assertEqual(len(driver.tokens), 6)
        display = self.Display(driver.config)
        display.u = 1.0
        by_token = {token.id: token for token in snapshot.tokens}
        rects = [rect for _hand, _token, rect, _unit
                 in display._layout(snapshot, by_token, 1280, 720)]
        self.assertEqual(len(rects), 6)
        for first in range(len(rects)):
            for second in range(first + 1, len(rects)):
                ax, ay, aw, ah = rects[first]
                bx, by, bw, bh = rects[second]
                overlap = ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah
                self.assertFalse(overlap, f"{rects[first]} deckt {rects[second]}")

    def test_the_lobby_and_the_countdown_render(self):
        lobby = self.Lobby(Config())
        lobby.progress = 0.5
        canvas = lobby.render(self.image)
        self.assertEqual(canvas.shape, self.image.shape)
        self.assertGreater(int(canvas.max()), 0)
        self.assertGreater(int(self.countdown_overlay(canvas, 2.4).max()), 0)

    def test_text_falls_back_without_a_font(self):
        """Ohne Pillow muss die Anzeige weiterlaufen, nur schlichter."""
        from src.ui import theme

        setter = theme.Typeset()
        setter.available = False
        canvas = self.np.zeros((60, 400, 3), dtype=self.np.uint8)
        setter.draw(canvas, "PROBE", 10, 40, 22, theme.INK)
        self.assertGreater(int(canvas.max()), 0)
        self.assertGreater(setter.width("PROBE", 22), 0)
