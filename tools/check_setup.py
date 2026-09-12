#!/usr/bin/env python3
"""Prueft, ob auf diesem Rechner alles vorhanden ist.

Aufruf:  python tools/check_setup.py
"""

from __future__ import annotations

import os
import platform
import sys

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("GLOG_minloglevel", "2")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK = "  ok   "
WARN = " warn  "
FAIL = " fehlt "


def line(state: str, text: str) -> None:
    print(f"[{state}] {text}")


def check_python() -> bool:
    version = sys.version_info
    good = (3, 9) <= (version.major, version.minor) < (3, 14)
    line(OK if good else WARN, f"Python {platform.python_version()} ({platform.system()})")
    return good


def check_package(name: str, import_name: str | None = None) -> bool:
    try:
        module = __import__(import_name or name)
        version = getattr(module, "__version__", "?")
        line(OK, f"{name} {version}")
        return True
    except Exception as exc:
        line(FAIL, f"{name} ({exc})")
        return False


def check_model() -> bool:
    from src.config import HAND_MODEL_PATH

    if os.path.exists(HAND_MODEL_PATH) and os.path.getsize(HAND_MODEL_PATH) > 100_000:
        size = os.path.getsize(HAND_MODEL_PATH) / 1_000_000
        line(OK, f"Hand Modell vorhanden ({size:.1f} MB)")
        return True
    line(WARN, "Hand Modell fehlt, wird beim ersten Start geladen")
    return False


def check_camera(index: int = 0) -> bool:
    try:
        import cv2
    except Exception:
        line(FAIL, "OpenCV fehlt, Kamera nicht pruefbar")
        return False

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        line(FAIL, f"Kamera {index} nicht erreichbar")
        return False
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        line(FAIL, f"Kamera {index} liefert kein Bild")
        return False
    line(OK, f"Kamera {index} liefert {frame.shape[1]}x{frame.shape[0]}")
    return True


def check_audio() -> bool:
    try:
        import sounddevice as sd
    except Exception as exc:
        line(WARN, f"sounddevice nicht nutzbar ({exc}). Interne Engine bleibt aus.")
        return False
    try:
        device = sd.query_devices(kind="output")
        line(OK, f"Audioausgabe: {device['name']}")
        return True
    except Exception as exc:
        line(WARN, f"keine Audioausgabe gefunden ({exc})")
        return False


def main() -> int:
    print("Systemcheck Gesture Music Instrument\n")
    results = [
        check_python(),
        check_package("numpy"),
        check_package("opencv-python", "cv2"),
        check_package("mediapipe"),
        check_package("python-osc", "pythonosc"),
    ]
    check_audio()
    check_model()
    check_camera(int(sys.argv[1]) if len(sys.argv) > 1 else 0)

    print()
    if all(results):
        print("Alles Wichtige ist da. Start mit:  python main.py")
        return 0
    print("Es fehlt etwas. Bitte zuerst:  pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    sys.exit(main())
