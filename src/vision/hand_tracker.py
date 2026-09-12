"""Wiedererkennung einzelner Haende ueber mehrere Frames.

MediaPipe liefert pro Bild nur eine Liste von Haenden, ohne Identitaet.
Fuer die Handschuhe brauchen wir aber eine stabile ID: ein Handschuh
klebt an genau der Hand, die ihn gegriffen hat, auch wenn diese quer
durch das Bild laeuft.

Das Verfahren ist bewusst einfach gehalten: jede bekannte Hand sagt ueber
ihre letzte Geschwindigkeit voraus, wo sie jetzt sein sollte. Danach
werden Vorhersagen und neue Beobachtungen nach kuerzester Distanz
gepaart. Das reicht bei 30 bis 60 Bildern pro Sekunde locker aus, weil
sich eine Hand zwischen zwei Bildern nur wenige Zentimeter bewegt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import TrackConfig
from .types import HandObservation


@dataclass
class HandTrack:
    id: int
    observation: HandObservation
    position: np.ndarray            # geglaettete Mitte, shape (2,)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    handedness: str = "right"
    missing: int = 0
    age: int = 0

    @property
    def visible(self) -> bool:
        return self.missing == 0

    @property
    def predicted(self) -> np.ndarray:
        return self.position + self.velocity


class HandTracker:
    def __init__(self, config: TrackConfig):
        self.config = config
        self.tracks: Dict[int, HandTrack] = {}
        self._next_id = 1

    # ------------------------------------------------------------------
    def update(self, observations: List[HandObservation]) -> List[HandTrack]:
        centers = [np.asarray(obs.center[:2], dtype=np.float32) for obs in observations]
        unmatched_obs = set(range(len(observations)))
        unmatched_tracks = set(self.tracks.keys())

        pairs = []
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            for obs_index in unmatched_obs:
                distance = float(np.linalg.norm(track.predicted - centers[obs_index]))
                if distance <= self.config.max_match_distance:
                    pairs.append((distance, track_id, obs_index))
        pairs.sort(key=lambda item: item[0])

        used_tracks, used_obs = set(), set()
        for _distance, track_id, obs_index in pairs:
            if track_id in used_tracks or obs_index in used_obs:
                continue
            used_tracks.add(track_id)
            used_obs.add(obs_index)
            self._advance(self.tracks[track_id], observations[obs_index], centers[obs_index])

        for obs_index in sorted(unmatched_obs - used_obs):
            self._create(observations[obs_index], centers[obs_index])

        for track_id in sorted(unmatched_tracks - used_tracks):
            track = self.tracks[track_id]
            track.missing += 1
            track.age += 1
            track.position = track.position + track.velocity
            track.velocity *= np.float32(self.config.velocity_damping)
            if track.missing > self.config.max_missing_frames:
                del self.tracks[track_id]

        return list(self.tracks.values())

    # ------------------------------------------------------------------
    def _create(self, obs: HandObservation, center: np.ndarray) -> HandTrack:
        track = HandTrack(
            id=self._next_id,
            observation=obs,
            position=center.copy(),
            handedness=obs.handedness,
        )
        self.tracks[self._next_id] = track
        self._next_id += 1
        return track

    def _advance(self, track: HandTrack, obs: HandObservation, center: np.ndarray) -> None:
        new_velocity = center - track.position
        track.velocity = track.velocity * 0.5 + new_velocity * 0.5
        track.position = center.copy()
        track.observation = obs
        track.handedness = obs.handedness
        track.missing = 0
        track.age += 1

    # ------------------------------------------------------------------
    def get(self, track_id: int) -> Optional[HandTrack]:
        return self.tracks.get(track_id)

    def reset(self) -> None:
        self.tracks.clear()
