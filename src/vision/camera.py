"""Kamera Zugriff ueber OpenCV, in einem eigenen Thread.

Der Thread haelt immer nur das neueste Bild vor. Dadurch laeuft die
Verarbeitung nie einem vollen Puffer hinterher und die Latenz bleibt klein.
"""

from __future__ import annotations

import platform
import threading
import time
from typing import Optional

import cv2
import numpy as np

from ..config import CameraConfig
from .types import Frame


def _backend() -> int:
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW      # deutlich schnellerer Start als MSMF
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._running = False
        self._index = 0
        self._start_time = time.time()

    def open(self) -> "Camera":
        cap = cv2.VideoCapture(self.config.device, _backend())
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.config.device)
        if not cap.isOpened():
            raise CameraError(
                f"Kamera {self.config.device} laesst sich nicht oeffnen. "
                "Anderen Index probieren (--camera 1) oder Kamerarechte pruefen."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if self.config.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._latest = frame

    def read(self) -> Optional[Frame]:
        with self._lock:
            frame = None if self._latest is None else self._latest.copy()
        if frame is None:
            return None
        self._index += 1
        stamp = int((time.time() - self._start_time) * 1000)
        return Frame(image=frame, index=self._index, timestamp_ms=stamp)

    def wait_for_frame(self, timeout: float = 5.0) -> Frame:
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.read()
            if frame is not None:
                return frame
            time.sleep(0.01)
        raise CameraError("Kamera liefert kein Bild (Timeout).")

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()


class SyntheticCamera:
    """Ersatzkamera fuer Tests und Demos ohne Hardware."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._index = 0

    def open(self) -> "SyntheticCamera":
        return self

    def read(self) -> Frame:
        self._index += 1
        img = np.full((self.config.height, self.config.width, 3), 18, dtype=np.uint8)
        img[:, :, 2] = 30
        stamp = int(self._index * (1000.0 / max(self.config.fps, 1)))
        return Frame(image=img, index=self._index, timestamp_ms=stamp)

    def wait_for_frame(self, timeout: float = 1.0) -> Frame:
        return self.read()

    def close(self) -> None:
        return None

    def __enter__(self) -> "SyntheticCamera":
        return self

    def __exit__(self, *exc) -> None:
        return None
