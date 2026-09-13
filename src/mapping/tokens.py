"""Die Beat Schmiede.

Ein Loop ist ein Beat in der Hand. Gespielt wird ausschliesslich mit der
Haltung der Hand - die Hand darf dabei entspannt vor dem Koerper bleiben
und muss nirgendwo hin wandern:

    Bibliothek (links)    leere Hand auf Hoehe eines Fachs, Pinch nimmt
    LIVE                  Finger zaehlen die Dichte, Neigung die
                          Lautstaerke, der Daumen die Klangfarbe
    FROZEN                Handgelenk umdrehen haelt die Werte fest,
                          zurueckdrehen gibt sie wieder frei
    FUSING                zwei gehaltene Loops zusammenfuehren
    FLYING / PLACED       ein kurzer Pinch legt den Loop ins Regal, wo er
                          weiterlaeuft
    Regal (rechts)        leere Hand auf Hoehe eines Fachs: kurzer Pinch
                          holt den Loop zurueck in die Hand, langer Pinch
                          loescht ihn

Die beiden Spalten haben feste Faecher an festen Plaetzen. Vorher lagen
Bibliothek und Regal in einer gemeinsamen Liste; jeder abgelegte Loop hat
alles darin verschoben und verkleinert, bis Treffen zum Gluecksspiel
wurde. Getrennte Spalten mit fester Fachhoehe und einer Sperre gegen das
Umspringen (`Column.pick`) machen das Zielen wieder verlaesslich.

Gezielt wird dabei mit `gestures.grip_point`, also mit dem Punkt, an dem
Daumen und Zeigefinger zugreifen - nicht mit der getrackten Handmitte,
die rund eine Fachhoehe darunter liegt. Und die Sperre gilt nur der
ruhenden Hand: sobald die Hand sichtbar wandert, loesen sich Glaettung
und Sperre, und der zugehende Pinch schnappt ohne beides auf die Hoehe
der Finger. Sonst zeigt die Spalte noch auf ein Fach, an dem die Hand
laengst vorbei ist, und man greift das falsche daneben.

Ein Loop kann mehrere Rezepte tragen. Jedes Rezept ist ein Preset mit
seinen eigenen eingefrorenen Werten, deshalb klingt eine Fusion genau so
wie die beiden Einzelloops davor.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

from ..config import ForgeConfig, MappingConfig
from ..layout import Column
from ..vision.hand_tracker import HandTrack
from ..vision.types import BrowserEntry, HandView, SoundSlot
from . import gestures
from .normalizer import ChannelSmoother
from .presets import PRESET_IDS, preset

LIVE = "live"
FROZEN = "frozen"
FUSING = "fusing"
FLYING = "flying"
PLACED = "placed"

# Die beiden Spalten am Bildrand
LIBRARY = "library"
RACK = "rack"


@dataclass
class Recipe:
    """Ein Preset mit seinen Werten. Der Key bleibt ueber Fusionen stabil,
    damit die Sound Engine die Stimme nicht neu anlegen muss."""

    key: str
    preset_id: str
    volume: float = 0.5
    cutoff: float = 0.55
    density: float = 0.375


@dataclass
class Token:
    id: int
    recipes: List[Recipe]
    state: str = LIVE
    x: float = 0.5
    y: float = 0.5
    hand_id: Optional[int] = None
    track: Optional[HandTrack] = field(default=None, repr=False)
    muted: bool = False
    hold_frames: int = 0
    fingers: int = 2                # zuletzt uebernommene Fingerzahl
    freeze_progress: float = 0.0    # 0 = offen, 1 = gehalten
    flip_frames: int = 0            # wie lange die Handseite schon stimmt
    count_candidate: int = -1
    count_frames: int = 0
    approach: float = 0.0           # Naehe zum Fusionspartner, 0..1
    fuse_progress: float = 0.0      # Animationsfortschritt, 0..1
    fuse_parts: List[Tuple[float, float, str]] = field(default_factory=list)
    fuse_center: Tuple[float, float] = (0.5, 0.5)
    pinch_frames: int = 0           # wie lange der Ablege-Pinch schon steht
    place_progress: float = 0.0     # 0 = offen, 1 = gleich weggeworfen
    pinch_latched: bool = True      # der Greif-Pinch muss erst aufgehen
    trash_hold_frames: int = 0      # wie lange der Loop schon im Muelleimer liegt
    trash_progress: float = 0.0     # 0 = weg vom Muelleimer, 1 = gleich geloescht
    rack_index: int = 0
    park_blocked: int = 0           # Restframes der Meldung "Regal voll"
    fly_progress: float = 0.0
    fly_from: Tuple[float, float] = (0.5, 0.5)
    # Summenregler: nur bei fusionierten Loops im Einsatz.
    master: float = 1.0
    # Werte aus dem Moment, in dem die Hand noch flach zur Kamera stand.
    hold_values: Tuple[float, float, float] = (0.5, 0.55, 0.375)
    hold_master: float = 1.0
    smoother: ChannelSmoother = field(default_factory=ChannelSmoother, repr=False)

    @property
    def held(self) -> bool:
        return self.hand_id is not None

    @property
    def visible_hand(self) -> bool:
        return self.track is not None and self.track.visible

    @property
    def fused(self) -> bool:
        return len(self.recipes) > 1

    @property
    def preset_ids(self) -> List[str]:
        return [recipe.preset_id for recipe in self.recipes]


@dataclass
class BrowserHand:
    """Der Zustand einer leeren Hand an einer der beiden Spalten."""

    hand_id: int
    zone: str = LIBRARY
    index: int = 0
    pinch_frames: int = 0
    cooldown: int = 0
    latched: bool = False           # Pinch muss einmal geoeffnet werden
    pinching: bool = False          # steht der Griff schon seit dem letzten Frame?
    smooth_y: float = -1.0          # beruhigte Zielhoehe fuer die Auswahl
    delete_progress: float = 0.0    # 0 = herausnehmen, 1 = gleich geloescht


class TokenForge:
    def __init__(
        self,
        forge: ForgeConfig,
        mapping: MappingConfig,
        seed: Optional[int] = None,
    ):
        self.cfg = forge
        self.mapping = mapping
        self.rng = random.Random(seed)
        self.dt = 1.0 / 30.0
        self.tokens: List[Token] = []
        self.library: List[str] = list(PRESET_IDS)
        self.events: List[str] = []
        self.hands: List[HandView] = []
        self.browsers: Dict[int, BrowserHand] = {}
        self._next_token = 1
        self._next_recipe = 1
        self._fuse_hold: Dict[Tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    def shuffle_library(self) -> None:
        self.rng.shuffle(self.library)

    def place_all(self) -> None:
        """Alle gehaltenen Loops wandern ins Regal."""
        for token in list(self.tokens):
            if token.held:
                self._park(token, instant=True)

    def clear(self) -> None:
        self.tokens.clear()
        self.browsers.clear()
        self._fuse_hold.clear()
        self.hands = []

    # ------------------------------------------------------------------
    @property
    def library_column(self) -> Column:
        """Die linke Spalte: ein Fach pro Beat in der Bibliothek."""
        cfg = self.cfg
        return Column.packed(cfg.library_x, cfg.library_top, cfg.library_bottom,
                             len(self.library))

    @property
    def rack_column(self) -> Column:
        """Die rechte Spalte: immer `max_parked` Faecher.

        Die Zahl der Faecher haengt bewusst nicht davon ab, wie viele
        Loops darin liegen. Ein Fach bleibt dort, wo es war, auch wenn
        das darueber frei wird - sonst muesste man beim Zugreifen jedes
        Mal neu suchen.
        """
        cfg = self.cfg
        return Column.packed(cfg.rack_x, cfg.rack_top, cfg.rack_bottom,
                             cfg.max_parked)

    def library_entries(self) -> List[BrowserEntry]:
        """Die Beats, die es frisch zu greifen gibt."""
        return [
            BrowserEntry(kind="preset", preset_ids=[preset_id])
            for preset_id in self.library
        ]

    def rack_entries(self) -> List[BrowserEntry]:
        """Ein Eintrag pro Fach im Regal, leere Faecher als None."""
        slots: List[Optional[BrowserEntry]] = [None] * max(self.cfg.max_parked, 1)
        for token in self.parked:
            if 0 <= token.rack_index < len(slots):
                slots[token.rack_index] = BrowserEntry(
                    kind="loop", preset_ids=token.preset_ids, token_id=token.id
                )
        return slots

    def entries(self) -> List[BrowserEntry]:
        """Alles, was greifbar ist - Bibliothek zuerst. Nur fuer Ueberblick
        und Tests; gespielt wird ueber die beiden getrennten Spalten."""
        return self.library_entries() + [
            entry for entry in self.rack_entries() if entry is not None
        ]

    @property
    def parked(self) -> List[Token]:
        return [t for t in self.tokens if t.state == PLACED]

    def token_by_id(self, token_id: Optional[int]) -> Optional[Token]:
        return next((t for t in self.tokens if t.id == token_id), None)

    def _free_rack_slot(self) -> int:
        """Das oberste freie Fach. -1, wenn das Regal voll ist."""
        taken = {token.rack_index for token in self.parked}
        taken.update(
            token.rack_index for token in self.tokens if token.state == FLYING
        )
        for index in range(max(self.cfg.max_parked, 1)):
            if index not in taken:
                return index
        return -1

    # ------------------------------------------------------------------
    def update(self, tracks: List[HandTrack], dt: float = 1.0 / 30.0) -> List[Token]:
        self.events.clear()
        self.dt = max(dt, 1e-3)
        by_id = {track.id: track for track in tracks}
        busy: Set[int] = set()

        # 1. gehaltene Loops an ihre Hand koppeln
        for token in list(self.tokens):
            if not token.held:
                continue
            track = by_id.get(token.hand_id)
            token.track = track
            if track is None or track.missing > self.cfg.release_frames:
                self._park(token, instant=True)
                if token.held:
                    # Die Hand ist weg und im Regal war kein Fach frei.
                    # Der Loop haette sonst unsichtbar weitergeklungen,
                    # ohne dass ihn noch jemand erreichen kann.
                    self._release_hand(token)
                    self._remove(token)
                    self.events.append("discard")
                continue
            busy.add(track.id)
            token.hold_frames += 1
            if track.visible:
                token.x = float(track.position[0])
                token.y = float(track.position[1])

        # 2. Zustand pro Loop weiterdrehen
        for token in list(self.tokens):
            token.park_blocked = max(0, token.park_blocked - 1)
            if token.state == LIVE:
                self._update_live(token)
            elif token.state == FROZEN:
                self._update_frozen(token)
            elif token.state == FUSING:
                self._update_fusing(token)
            elif token.state == FLYING:
                self._update_flying(token)

        # 3. Muelleimer: gehaltenen Loop in die Ecke tragen loescht ihn
        self._update_trash()

        # 4. Fusion pruefen
        self._update_fusion()

        # 5. freie Haende blaettern im Browser
        free = [track for track in tracks if track.visible and track.id not in busy]
        self._update_browser(free, {track.id for track in tracks})

        # 6. Regal neu ausrichten und Anzeige fuettern
        self._layout_rack()
        self._collect_hands(tracks, by_id)
        return self.tokens

    # ------------------------------------------------------------------
    # Loop in der Hand
    # ------------------------------------------------------------------
    def _update_live(self, token: Token) -> None:
        if not token.visible_hand:
            return
        obs = token.track.observation
        # Beim Ablege-Pinch klappt der Zeigefinger ein. Wuerden die Regler
        # weiterlaufen, saenke die Dichte im selben Moment - der Loop
        # klaenge beim Ablegen anders als beim Spielen. Deshalb ruhen
        # waehrend des Pinch alle Werte.
        if self._check_release(token):
            return
        count = self._stable_count(token, obs)
        token.muted = count == 0
        lean = gestures.lean(obs, self.mapping)

        if token.fused:
            # Ein fusionierter Loop behaelt die Mischung seiner Beats. Die
            # Neigung wird zum Summenregler: aufrecht ist voll, nach innen
            # gekippt blendet aus. Aufrecht heisst also "nichts aendert
            # sich" - beim Auftauen springt darum nichts.
            token.master = token.smoother.smooth("master", min(1.0, lean * 2.0))
        else:
            recipe = token.recipes[0]
            recipe.volume = token.smoother.smooth("volume", lean)
            recipe.cutoff = token.smoother.smooth(
                "cutoff", gestures.thumb_open(obs, self.mapping)
            )
            recipe.density = gestures.density_from_count(count)

        # Solange die Handflaeche klar zur Kamera zeigt, gelten die Werte
        # als gespielt. Sobald sich das Handgelenk dreht, bleibt dieser
        # Stand stehen - er ist es, der beim Halten einrastet.
        if gestures.palm_facing(obs) > self.mapping.flip_arm:
            first = token.recipes[0]
            token.hold_values = (first.volume, first.cutoff, first.density)
            token.hold_master = token.master

        # Handgelenk umdrehen: Werte festhalten.
        if self._flip_holds(token, obs, want_back=True):
            self._freeze(token)
            return

        token.freeze_progress = max(
            0.0, min(0.9, token.flip_frames / max(self.cfg.freeze_frames, 1))
        )

    def _update_frozen(self, token: Token) -> None:
        if not token.held or not token.visible_hand:
            return
        obs = token.track.observation
        # Handflaeche wieder zeigen gibt den Loop wieder frei. Bei einer
        # Fusion bleiben die einzelnen Beats, wie sie sind - dort greift
        # nur noch der Summenregler.
        if self._flip_holds(token, obs, want_back=False):
            token.state = LIVE
            token.freeze_progress = 0.0
            token.master = 1.0
            token.hold_master = 1.0
            token.hold_values = (
                token.recipes[0].volume,
                token.recipes[0].cutoff,
                token.recipes[0].density,
            )
            token.smoother.reset_to(token.recipes[0].volume, token.recipes[0].cutoff)
            token.smoother.filters.pop("master", None)
            self.events.append("unfreeze")
            return
        token.freeze_progress = 1.0
        self._check_release(token)

    def _update_fusing(self, token: Token) -> None:
        token.fuse_progress += 1.0 / max(self.cfg.fuse_duration, 1)
        if token.fuse_progress >= 1.0:
            token.fuse_progress = 1.0
            token.state = FROZEN
            token.fuse_parts = []

    def _update_flying(self, token: Token) -> None:
        token.fly_progress += 1.0 / max(self.cfg.fly_frames, 1)
        target = self._rack_position(token.rack_index)
        ease = 1.0 - (1.0 - min(token.fly_progress, 1.0)) ** 3
        token.x = token.fly_from[0] + (target[0] - token.fly_from[0]) * ease
        token.y = token.fly_from[1] + (target[1] - token.fly_from[1]) * ease
        if token.fly_progress >= 1.0:
            token.state = PLACED
            token.x, token.y = target

    # ------------------------------------------------------------------
    def _stable_count(self, token: Token, obs) -> int:
        """Fingerzahl mit kurzer Beruhigung.

        Ohne diese Sperre wuerde ein einzelner Frame mit halb gestrecktem
        Finger die Dichte kurz umschalten - das hoert man sofort.
        """
        count = gestures.finger_count(obs, self.mapping)
        if count == token.fingers:
            token.count_candidate = -1
            token.count_frames = 0
            return token.fingers
        if count != token.count_candidate:
            token.count_candidate = count
            token.count_frames = 0
        token.count_frames += 1
        if token.count_frames >= max(self.mapping.finger_hold_frames, 1):
            token.fingers = count
            token.count_candidate = -1
            token.count_frames = 0
        return token.fingers

    def _flip_holds(self, token: Token, obs, want_back: bool) -> bool:
        """Zaehlt, wie lange das Handgelenk schon auf der gesuchten Seite steht."""
        facing = gestures.palm_facing(obs)
        threshold = self.mapping.flip_threshold
        matched = facing < -threshold if want_back else facing > threshold
        if matched:
            token.flip_frames += 1
        else:
            token.flip_frames = 0
        return token.flip_frames >= max(self.mapping.flip_frames, 1)

    def _freeze(self, token: Token) -> None:
        """Werte einrasten lassen - mit dem Stand von vor der Drehung.

        Beim Drehen kippt der Handteller zur Kante. Der sichtbare
        Daumenabstand schrumpft dabei, und mit ihm die Klangfarbe. Ohne
        diesen Rueckgriff wuerde jeder gehaltene Loop dumpfer klingen als
        der, den man gerade noch gespielt hat.
        """
        if token.fused:
            for recipe in token.recipes:
                recipe.volume = min(1.0, recipe.volume * token.hold_master)
            token.master = 1.0
            token.hold_master = 1.0
        else:
            recipe = token.recipes[0]
            recipe.volume, recipe.cutoff, recipe.density = token.hold_values
        token.state = FROZEN
        token.freeze_progress = 1.0
        token.flip_frames = 0
        token.muted = False
        self.events.append("freeze")

    # ------------------------------------------------------------------
    # Ablegen
    # ------------------------------------------------------------------
    def _check_release(self, token: Token) -> bool:
        """Pinch mit einem Loop in der Hand: kurz = ablegen, lang = wegwerfen.

        Ausgeloest wird beim Oeffnen der Finger, nicht beim Schliessen.
        So sieht man an der Karte noch, wohin die Reise geht, und kann
        einen versehentlich begonnenen Pinch abbrechen, indem man ihn
        sofort wieder loslaesst.

        Der Pinch, mit dem der Loop gerade erst geholt wurde, zaehlt
        nicht mit: er ist gesperrt, bis die Finger einmal aufgehen.
        Sonst wuerde ein einziger langer Pinch den Loop nehmen und im
        selben Zug wieder wegwerfen.

        Rueckgabe: True, solange ein Pinch steht - dann ruhen die Regler,
        damit der eingeklappte Zeigefinger die Dichte nicht verstellt.
        """
        if not token.visible_hand:
            token.pinch_frames = 0
            token.place_progress = 0.0
            return False

        pinching = gestures.is_pinch(token.track.observation, self.mapping)
        if not pinching:
            held = token.pinch_frames
            latched = token.pinch_latched
            token.pinch_frames = 0
            token.place_progress = 0.0
            token.pinch_latched = False
            if not latched and held >= max(self.cfg.place_frames, 1):
                self._park(token)
                return True
            return False

        # Waehrend zwei Loops aufeinander zulaufen, gilt die Hand als mit
        # dem Zusammenfuehren beschaeftigt. Sonst rutscht der Loop beim
        # Verschmelzen aus der Hand.
        if token.pinch_latched or token.approach > 0.01 \
                or token.hold_frames < self.cfg.min_hold_frames:
            token.pinch_frames = 0
            token.place_progress = 0.0
            return True

        token.pinch_frames += 1
        span = max(self.cfg.discard_frames - self.cfg.place_frames, 1)
        token.place_progress = min(
            1.0, max(0, token.pinch_frames - self.cfg.place_frames) / span
        )
        if token.pinch_frames >= max(self.cfg.discard_frames, 1):
            self._release_hand(token)
            self._remove(token)
            self.events.append("discard")
        return True

    def _update_trash(self) -> None:
        """Muelleimer neben dem Regal: Loop hintragen und halten loescht ihn.

        Alles andere im Instrument liest nur die Haltung der Hand, nie
        ihren Ort im Bild - das ist bewusst so. Diese eine Aktion ist die
        Ausnahme: Loeschen soll sich anfuehlen wie etwas wegtragen, nicht
        wie eine weitere Fingerhaltung sich merken zu muessen.
        """
        cfg = self.cfg
        for token in list(self.tokens):
            if not token.held or token.state not in (LIVE, FROZEN):
                token.trash_hold_frames = 0
                token.trash_progress = 0.0
                continue
            inside = math.hypot(token.x - cfg.trash_x, token.y - cfg.trash_y) <= cfg.trash_radius
            if inside:
                token.trash_hold_frames += 1
            else:
                token.trash_hold_frames = 0
            token.trash_progress = min(
                1.0, token.trash_hold_frames / max(cfg.trash_frames, 1)
            )
            if token.trash_hold_frames >= max(cfg.trash_frames, 1):
                self._release_hand(token)
                self._remove(token)
                self.events.append("discard")

    def _park(self, token: Token, instant: bool = False) -> None:
        """Der Loop wandert ins Regal und spielt dort weiter."""
        slot = self._free_rack_slot()
        if slot < 0:
            # Kein Fach frei. Der Loop bleibt in der Hand, und die Karte
            # sagt fuer einen Moment warum - stilles Nichtstun hat sich
            # wie ein Fehler des Instruments angefuehlt.
            token.pinch_frames = 0
            token.place_progress = 0.0
            token.park_blocked = 45
            self.events.append("rack_full")
            return
        if token.state == LIVE:
            self._freeze(token)
        self._release_hand(token)
        token.rack_index = slot
        token.fly_from = (token.x, token.y)
        token.fly_progress = 0.0
        token.pinch_frames = 0
        token.place_progress = 0.0
        token.pinch_latched = True
        token.state = FLYING
        if instant:
            token.fly_progress = 1.0
            token.state = PLACED
            token.x, token.y = self._rack_position(token.rack_index)
        self.events.append("park")

    def _release_hand(self, token: Token) -> None:
        """Loop von der Hand loesen und die Hand kurz sperren.

        Ohne die Sperre wuerde die eben frei gewordene Hand im selben
        Moment im Browser landen und beim naechsten Pinch sofort wieder
        etwas greifen.
        """
        if token.hand_id is not None:
            browser = self.browsers.get(token.hand_id)
            if browser is None:
                browser = BrowserHand(hand_id=token.hand_id)
                self.browsers[token.hand_id] = browser
            browser.cooldown = self.cfg.take_cooldown
            browser.latched = True
            browser.pinch_frames = 0
        token.hand_id = None
        token.track = None
        token.hold_frames = 0

    def _remove(self, token: Token) -> None:
        if token in self.tokens:
            self.tokens.remove(token)

    def _layout_rack(self) -> None:
        for token in self.parked:
            token.x, token.y = self._rack_position(token.rack_index)

    def _rack_position(self, index: int) -> Tuple[float, float]:
        column = self.rack_column
        index = max(0, min(index, column.slots - 1))
        return column.x, column.slot_y(index)

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------
    def _update_fusion(self) -> None:
        ready = [
            token
            for token in self.tokens
            if token.state == FROZEN and token.held and token.visible_hand
        ]
        for token in self.tokens:
            token.approach = 0.0

        pairs = sorted(
            (
                (math.hypot(a.x - b.x, a.y - b.y), a, b)
                for a, b in combinations(ready, 2)
            ),
            key=lambda item: item[0],
        )

        seen: Set[Tuple[int, int]] = set()
        busy: Set[int] = set()
        for distance, a, b in pairs:
            if a.id in busy or b.id in busy:
                continue
            if distance > self.cfg.fuse_distance:
                continue
            if len(a.recipes) + len(b.recipes) > self.cfg.max_recipes:
                continue
            busy.add(a.id)
            busy.add(b.id)
            key = (min(a.id, b.id), max(a.id, b.id))
            seen.add(key)
            held = self._fuse_hold.get(key, 0) + 1
            self._fuse_hold[key] = held
            # Naehe und Haltedauer zusammen ergeben die sichtbare Anzeige.
            closeness = 1.0 - distance / max(self.cfg.fuse_distance, 1e-6)
            progress = min(1.0, held / max(self.cfg.fuse_frames, 1))
            a.approach = b.approach = max(progress, closeness * 0.6)
            if held >= self.cfg.fuse_frames:
                self._begin_fusion(a, b)

        for key in list(self._fuse_hold):
            if key not in seen:
                del self._fuse_hold[key]

    def _begin_fusion(self, a: Token, b: Token) -> None:
        keeper, donor = (a, b) if a.id <= b.id else (b, a)
        keeper.fuse_parts = [
            (keeper.x, keeper.y, keeper.recipes[0].preset_id),
            (donor.x, donor.y, donor.recipes[0].preset_id),
        ]
        keeper.fuse_center = ((keeper.x + donor.x) * 0.5, (keeper.y + donor.y) * 0.5)
        keeper.recipes = keeper.recipes + donor.recipes
        keeper.state = FUSING
        keeper.fuse_progress = 0.0
        keeper.approach = 0.0
        self._release_hand(donor)
        self._remove(donor)
        self._fuse_hold.pop((min(a.id, b.id), max(a.id, b.id)), None)
        self.events.append("fuse")

    # ------------------------------------------------------------------
    # Die beiden Spalten
    # ------------------------------------------------------------------
    def _update_browser(self, free: List[HandTrack], alive: Set[int]) -> None:
        cfg = self.cfg
        # Der Zustand haengt an der Hand, nicht daran, ob sie gerade frei
        # ist. Sonst ginge die Sperre nach dem Ablegen im selben Frame
        # wieder verloren.
        for hand_id in list(self.browsers):
            if hand_id not in alive:
                del self.browsers[hand_id]

        library = self.library_entries()
        rack = self.rack_entries()

        for track in free:
            obs = track.observation
            if obs is None:
                self.browsers.pop(track.id, None)
                continue

            # Gezielt wird mit dem Punkt, an dem Daumen und Zeigefinger
            # zugreifen, nicht mit der getrackten Handmitte. Die liegt
            # rund eine Fachhoehe tiefer - man zeigte auf ein Fach und
            # bekam das darunter.
            aim = gestures.grip_point(obs, self.mapping)
            x, y = float(aim[0]), float(aim[1])
            if x <= cfg.library_capture_x and library:
                zone, column = LIBRARY, self.library_column
            elif x >= cfg.rack_capture_x and any(rack):
                zone, column = RACK, self.rack_column
            else:
                self.browsers.pop(track.id, None)
                continue

            browser = self.browsers.get(track.id)
            if browser is None:
                browser = BrowserHand(hand_id=track.id, zone=zone)
                self.browsers[track.id] = browser
            if browser.zone != zone:
                # Seitenwechsel: alles zuruecksetzen, damit kein halber
                # Pinch von der anderen Spalte mitwandert.
                browser.zone = zone
                browser.smooth_y = -1.0
                browser.pinch_frames = 0
                browser.pinching = False
                browser.delete_progress = 0.0
                browser.latched = True
            browser.cooldown = max(0, browser.cooldown - 1)

            # Die Zielhoehe zeigt auf ein Fach - dieselbe Formel wie beim
            # Zeichnen. Glaettung und Sperre halten eine ruhende Hand auf
            # ihrem Fach; eine Hand, die sichtbar wandert, sollen sie
            # nicht bremsen. Deshalb loesen sich beide mit der
            # Geschwindigkeit: `step` ist der Weg seit dem letzten Bild,
            # `settled` faellt von 1 (Hand steht) auf 0 (Hand zieht
            # durch).
            if browser.smooth_y < 0.0:
                browser.smooth_y = y
                browser.index = column.nearest(y)
            step = abs(y - browser.smooth_y)
            settled = max(0.0, 1.0 - step / max(cfg.select_settle, 1e-6))
            keep = min(max(cfg.select_smoothing, 0.0), 0.95) * settled
            browser.smooth_y += (y - browser.smooth_y) * (1.0 - keep)

            pinching = gestures.is_pinch(obs, self.mapping)
            # Ein Griff, der wirklich etwas ausloest. Im Moment, in dem
            # er zugeht, zaehlt allein, was unter den Fingern liegt: ohne
            # Sperre, ohne Nachlauf. Danach steht die Auswahl fest,
            # solange der Griff haelt - sonst wandert das Fach noch
            # weg, waehrend man es schon greift.
            armed = pinching and not browser.latched and browser.cooldown <= 0
            if armed and not browser.pinching:
                browser.smooth_y = y
                browser.index = column.nearest(y)
            elif not armed:
                browser.index = column.pick(
                    browser.smooth_y, browser.index,
                    cfg.select_hysteresis * settled,
                )
            browser.pinching = armed

            if zone == LIBRARY:
                self._browse_library(browser, library, track, pinching)
            else:
                self._browse_rack(browser, rack, track, pinching)

    def _browse_library(
        self, browser: BrowserHand, entries: List[BrowserEntry],
        hand: HandTrack, pinching: bool,
    ) -> None:
        """Links: ein Pinch nimmt den Beat sofort mit."""
        if not pinching:
            browser.latched = False
            browser.pinch_frames = 0
            return
        if browser.latched or browser.cooldown > 0:
            return
        browser.pinch_frames += 1
        if browser.pinch_frames >= max(self.cfg.take_frames, 1):
            self._take(entries[min(browser.index, len(entries) - 1)], hand, browser)

    def _browse_rack(
        self, browser: BrowserHand, slots: List[Optional[BrowserEntry]],
        hand: HandTrack, pinching: bool,
    ) -> None:
        """Rechts: kurzer Pinch holt den Loop zurueck, langer loescht ihn.

        Dieselbe Regel wie mit einem Loop in der Hand - kurz bewegt,
        lang wirft weg. Ausgeloest wird beim Oeffnen der Finger, deshalb
        laesst sich ein begonnener Griff abbrechen, indem man ihn sofort
        wieder loslaesst.
        """
        cfg = self.cfg
        entry = slots[browser.index] if browser.index < len(slots) else None
        if not pinching:
            held, latched = browser.pinch_frames, browser.latched
            browser.latched = False
            browser.pinch_frames = 0
            browser.delete_progress = 0.0
            if (
                not latched
                and browser.cooldown <= 0
                and entry is not None
                and held >= max(cfg.take_frames, 1)
            ):
                self._take(entry, hand, browser)
            return

        if browser.latched or browser.cooldown > 0 or entry is None:
            browser.pinch_frames = 0
            browser.delete_progress = 0.0
            return

        browser.pinch_frames += 1
        span = max(cfg.discard_frames - cfg.take_frames, 1)
        browser.delete_progress = min(
            1.0, max(0, browser.pinch_frames - cfg.take_frames) / span
        )
        if browser.pinch_frames >= max(cfg.discard_frames, 1):
            self._delete_parked(entry, browser)

    def _delete_parked(self, entry: BrowserEntry, browser: BrowserHand) -> None:
        browser.pinch_frames = 0
        browser.delete_progress = 0.0
        browser.latched = True
        browser.cooldown = self.cfg.take_cooldown
        token = self.token_by_id(entry.token_id)
        if token is None or token.state != PLACED:
            return
        self._remove(token)
        self.events.append("discard")

    def _take(self, entry: BrowserEntry, hand: HandTrack, browser: BrowserHand) -> None:
        browser.pinch_frames = 0
        browser.delete_progress = 0.0
        browser.latched = True
        browser.cooldown = self.cfg.take_cooldown
        if entry.kind == "loop":
            token = self.token_by_id(entry.token_id)
            if token is None or token.state != PLACED:
                return
            token.state = FROZEN
            token.hand_id = hand.id
            token.track = hand
            token.hold_frames = 0
            token.freeze_progress = 1.0
            token.flip_frames = 0
            token.pinch_frames = 0
            token.place_progress = 0.0
            token.pinch_latched = True
            token.x = float(hand.position[0])
            token.y = float(hand.position[1])
            self.events.append("pickup")
            return
        if len(self.tokens) >= self.cfg.max_tokens:
            self.events.append("full")
            return
        self._spawn(entry.preset_ids[0], hand)

    def _spawn(self, preset_id: str, hand: HandTrack) -> Token:
        recipe = Recipe(key=f"r{self._next_recipe}", preset_id=preset_id)
        self._next_recipe += 1
        token = Token(
            id=self._next_token,
            recipes=[recipe],
            state=LIVE,
            x=float(hand.position[0]),
            y=float(hand.position[1]),
            hand_id=hand.id,
            track=hand,
        )
        token.smoother.alpha = self.mapping.smoothing
        if hand.observation is not None:
            token.fingers = gestures.finger_count(hand.observation, self.mapping)
        self._next_token += 1
        self.tokens.append(token)
        self.events.append("grab")
        return token

    # ------------------------------------------------------------------
    def _collect_hands(self, tracks: List[HandTrack], by_id: Dict[int, HandTrack]) -> None:
        held = {t.hand_id: t for t in self.tokens if t.held}
        views: List[HandView] = []
        for track in tracks:
            if not track.visible or track.observation is None:
                continue
            obs = track.observation
            token = held.get(track.id)
            browser = self.browsers.get(track.id)
            aim = gestures.grip_point(obs, self.mapping)
            views.append(
                HandView(
                    hand_id=track.id,
                    x=float(track.position[0]),
                    y=float(track.position[1]),
                    aim_x=float(aim[0]),
                    aim_y=float(aim[1]),
                    handedness=track.handedness,
                    holding=token is not None,
                    token_id=token.id if token else None,
                    zone=browser.zone if browser else "",
                    selection=browser.index if browser else 0,
                    take_progress=(
                        min(1.0, browser.pinch_frames / max(self.cfg.take_frames, 1))
                        if browser
                        else 0.0
                    ),
                    delete_progress=browser.delete_progress if browser else 0.0,
                    fingers=token.fingers if token else gestures.finger_count(obs, self.mapping),
                    facing=gestures.palm_facing(obs),
                    lean=gestures.lean(obs, self.mapping),
                    thumb=gestures.thumb_open(obs, self.mapping),
                    landmarks=obs.landmarks,
                )
            )
        self.hands = views

    # ------------------------------------------------------------------
    def slots(self) -> List[SoundSlot]:
        out: List[SoundSlot] = []
        for token in self.tokens:
            gain = 0.0 if token.muted else token.master
            for recipe in token.recipes:
                out.append(
                    SoundSlot(
                        key=recipe.key,
                        preset_id=recipe.preset_id,
                        gain=recipe.volume * gain * preset(recipe.preset_id).gain,
                        cutoff=recipe.cutoff,
                        density=recipe.density,
                    )
                )
        return out

    @property
    def live_openness(self) -> List[float]:
        """Handoeffnung der Spieler, die gerade einen Loop formen.

        Daraus bauen die Gruppenmacros ihren Build-up: alle Haende
        gemeinsam weit oeffnen ist ein Riser, alle gemeinsam zur Faust
        ist der Drop. Die Position im Bild spielt nirgends mehr eine
        Rolle.
        """
        return [
            gestures.openness(token.track.observation, self.mapping)
            for token in self.tokens
            if token.state == LIVE and not token.muted and token.visible_hand
        ]

    @property
    def placed_count(self) -> int:
        return len(self.parked)
