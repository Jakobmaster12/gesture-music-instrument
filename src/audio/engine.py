"""Interne Sound Engine.

Damit laeuft das Instrument ohne Ableton. Ein 16tel Step Sequencer
spielt beliebig viele Schichten gleichzeitig - eine pro Beat, der gerade
in einer Hand liegt oder im Bild abgelegt ist. Jede Schicht bringt ihr
eigenes Preset und ihre eigenen Werte mit:

    gain    -> Lautstaerke
    cutoff  -> Tiefpassfilter
    density -> welches der vier Patterns bzw. welche Loop Variante laeuft

Es gibt zwei Arten von Schichten. Eine Schicht der Art "steps" setzt den
Beat aus einzelnen Schlaegen auf dem 16tel Raster zusammen. Eine Schicht
der Art "loop" spielt eine fertige Audiodatei ueber ihre ganze Laenge -
das sind die eigenen Beats aus dem Ordner `loops/`. Beide haengen an
derselben Uhr, deshalb laufen sie ohne Zutun synchron.

Die Engine ist thread sicher: der Audio Callback liest nur eine
Parameterkopie, die der Videoloop von aussen setzt.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..config import AudioConfig
from ..mapping.presets import Preset, preset
from ..vision.types import EngineSnapshot
from . import synth
from .loops import load_library

STEPS_PER_BAR = 16
BARS = 4


@dataclass
class SlotParams:
    preset_id: str
    gain: float = 0.0
    cutoff: float = 0.5
    density: float = 0.0


@dataclass
class Voice:
    buffer: np.ndarray
    position: int = 0
    amp: float = 1.0


@dataclass
class Layer:
    """Eine klingende Schicht, gehoert zu genau einem Beat im Bild."""

    preset: Preset
    samples: List[np.ndarray]
    filter: synth.OnePoleLowpass
    voices: List[Voice] = field(default_factory=list)
    gain: float = 0.0
    cutoff: float = 0.5
    density: float = 0.0
    current_gain: float = 0.0

    @property
    def is_loop(self) -> bool:
        return self.preset.kind == "loop"

    def trigger(self, index: int, amp: float = 1.0, offset: int = 0) -> None:
        if not self.samples:
            return
        buffer = self.samples[index % len(self.samples)]
        if offset >= len(buffer):
            return
        if len(self.voices) > 12:
            self.voices.pop(0)
        self.voices.append(Voice(buffer=buffer, position=offset, amp=amp))

    def render(self, frames: int, out: np.ndarray) -> None:
        alive: List[Voice] = []
        for voice in self.voices:
            remaining = len(voice.buffer) - voice.position
            if remaining <= 0:
                continue
            take = min(frames, remaining)
            out[:take] += voice.buffer[voice.position:voice.position + take] * voice.amp
            voice.position += take
            if voice.position < len(voice.buffer):
                alive.append(voice)
        self.voices = alive


def pick_sample(
    mode: str, count: int, local: int, bar: int, density: float
) -> Tuple[int, float]:
    """Welcher Sample Index und welche Anschlagstaerke auf diesem Step."""
    accent = 1.0 if local % 4 == 0 else 0.84
    if mode == "hat":
        return (1 if local % 8 == 4 else 0), (0.9 if local % 4 == 0 else 0.6)
    if mode == "roll":
        return 0, (1.0 if local % 4 == 0 else 0.55 + 0.3 * ((local // 2) % 2))
    if mode == "perc":
        return (local // 2) % max(count, 1), 0.9
    if mode == "bass":
        octave = 1 if (density > 0.66 and local % 4 == 2) else 0
        return bar * 2 + octave, 1.0
    if mode == "lead":
        span = 3 + int(density * 4)
        return (local * 3 + bar) % span, 0.9
    if mode == "arp":
        return local + bar, 0.85
    if mode == "bar":
        return bar, 1.0
    if mode == "blip":
        return local // 2 + bar, 0.8
    return 0, accent


class AudioEngine:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.sr = config.samplerate
        self._lock = threading.Lock()
        self._pending: Dict[str, SlotParams] = {}
        self._macro_riser = 0.0
        self._stream = None
        self.running = False

        self.bank = synth.build_sample_bank(self.sr)
        # Eigene WAV Dateien ueberschreiben einzelne Bank Eintraege, siehe
        # src/audio/sample_files.py fuer das Eintragen eigener Sounds.
        self.bank = synth.apply_custom_samples(self.bank, self.sr)
        # Und die fertigen Loops aus dem Ordner `loops/` kommen als eigene
        # Baenke dazu - eine pro Loop, mit einer Datei je Dichtestufe.
        self.bank.update(load_library(self.sr, config.bpm).bank)
        self.layers: Dict[str, Layer] = {}

        self._samples_per_step = self.sr * 60.0 / (config.bpm * 4.0)
        self._sample_counter = 0.0
        self._next_step = 0

    # ------------------------------------------------------------------
    def update_from_snapshot(self, snapshot: EngineSnapshot) -> None:
        with self._lock:
            self._pending = {
                slot.key: SlotParams(
                    preset_id=slot.preset_id,
                    gain=float(slot.gain),
                    cutoff=float(slot.cutoff),
                    density=float(slot.density),
                )
                for slot in snapshot.slots
            }
            self._macro_riser = float(snapshot.macros.riser)

    def set_slot(self, key: str, preset_id: str, gain: float, cutoff: float, density: float) -> None:
        """Direkter Zugriff ohne Snapshot, fuer Demos und Tests."""
        with self._lock:
            self._pending[key] = SlotParams(preset_id, gain, cutoff, density)

    # ------------------------------------------------------------------
    def _sync_layers(self) -> None:
        """Uebertraegt die Parameterkopie in die klingenden Schichten."""
        with self._lock:
            pending = dict(self._pending)

        for key in list(self.layers):
            if key not in pending:
                del self.layers[key]

        for key, params in pending.items():
            layer = self.layers.get(key)
            fresh = layer is None or layer.preset.id != params.preset_id
            if fresh:
                found = preset(params.preset_id)
                layer = Layer(
                    preset=found,
                    samples=self.bank.get(found.bank, [np.zeros(8, dtype=np.float32)]),
                    filter=synth.OnePoleLowpass(self.sr),
                )
                self.layers[key] = layer
            layer.gain = params.gain
            layer.cutoff = params.cutoff
            layer.density = params.density
            if fresh and layer.is_loop:
                # Ein frisch gegriffener Loop faellt mitten in den Takt.
                # Statt bis zum naechsten Taktanfang zu schweigen, setzt
                # er dort ein, wo die Uhr gerade steht - er klingt sofort
                # und liegt trotzdem im Raster.
                self._start_loop(layer, self._loop_offset(layer))

    def _trigger_step(self, step: int) -> None:
        bar = (step // STEPS_PER_BAR) % BARS
        local = step % STEPS_PER_BAR

        for layer in self.layers.values():
            if layer.is_loop:
                # Der Loop laeuft durch, auch wenn er gerade leise ist:
                # nur so bleibt er im Takt, wenn die Hand wieder aufdreht.
                if step % layer.preset.steps_per_loop == 0:
                    self._start_loop(layer, 0)
                continue
            if layer.gain <= 0.003:
                continue
            if local not in layer.preset.pattern_for(layer.density):
                continue
            index, amp = pick_sample(
                layer.preset.note_mode, len(layer.samples), local, bar, layer.density
            )
            layer.trigger(index, amp=amp)

    def _loop_offset(self, layer: Layer) -> int:
        """Wo innerhalb seiner Laenge der Loop gerade stehen muesste."""
        length = layer.preset.steps_per_loop * self._samples_per_step
        if length <= 1.0:
            return 0
        return int(self._sample_counter % length)

    def _start_loop(self, layer: Layer, offset: int) -> None:
        """Die Loopdatei von vorn (oder ab `offset`) anstossen.

        Die Dichte waehlt die Variante, aber immer nur zum Taktanfang -
        mitten im Takt umzuschneiden wuerde hoerbar stolpern.
        """
        layer.voices.clear()
        layer.trigger(layer.preset.variant_for(layer.density), amp=1.0, offset=offset)

    def _render_block(self, frames: int) -> np.ndarray:
        self._sync_layers()
        mono = np.zeros(frames, dtype=np.float32)
        layer_buffer = np.zeros(frames, dtype=np.float32)

        start = self._sample_counter
        end = start + frames
        while self._next_step * self._samples_per_step < end:
            if self._next_step * self._samples_per_step >= start:
                self._trigger_step(self._next_step)
            self._next_step += 1
        self._sample_counter = end

        with self._lock:
            riser = self._macro_riser

        sounding = 0
        for layer in self.layers.values():
            layer_buffer[:] = 0.0
            layer.render(frames, layer_buffer)
            if not np.any(layer_buffer):
                layer.current_gain += (layer.gain - layer.current_gain) * 0.2
                continue

            sounding += 1
            cutoff_hz = synth.cutoff_to_hz(0.08 + 0.92 * layer.cutoff)
            filtered = layer.filter.process(layer_buffer, cutoff_hz)
            ramp = np.linspace(layer.current_gain, layer.gain, frames, dtype=np.float32)
            layer.current_gain = layer.gain
            mono += filtered * ramp

        # Je mehr Schichten laufen, desto weiter zurueck der Summenpegel.
        mono *= np.float32(1.0 / (1.0 + 0.12 * max(0, sounding - 2)))
        if riser > 0.01:
            mono *= np.float32(1.0 + 0.25 * riser)

        mono = np.tanh(mono * np.float32(self.config.master_gain * 1.15)).astype(np.float32)
        return np.column_stack((mono, mono))

    # ------------------------------------------------------------------
    def render_offline(self, seconds: float) -> np.ndarray:
        """Rendert ohne Soundkarte, wird von den Tests benutzt."""
        frames = int(seconds * self.sr)
        block = self.config.blocksize
        chunks = []
        done = 0
        while done < frames:
            n = min(block, frames - done)
            chunks.append(self._render_block(n))
            done += n
        return np.concatenate(chunks, axis=0)

    def start(self) -> bool:
        if not self.config.enabled:
            return False
        try:
            import sounddevice as sd
        except Exception as exc:
            print(f"[audio] sounddevice nicht verfuegbar ({exc}). Interne Engine bleibt aus.")
            return False

        def callback(outdata, frames, time_info, status):  # pragma: no cover
            if status:
                pass
            outdata[:] = self._render_block(frames)

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sr,
                blocksize=self.config.blocksize,
                channels=2,
                dtype="float32",
                callback=callback,
                device=self.config.device,
            )
            self._stream.start()
        except Exception as exc:  # pragma: no cover
            print(f"[audio] Audioausgabe konnte nicht gestartet werden: {exc}")
            self._stream = None
            return False

        self.running = True
        print(f"[audio] Interne Engine laeuft, {self.config.bpm:.0f} BPM")
        return True

    def stop(self) -> None:
        self.running = False
        if self._stream is not None:  # pragma: no cover
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
