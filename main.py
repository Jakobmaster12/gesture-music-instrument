#!/usr/bin/env python3
"""Startpunkt des Instruments.

Beispiele:
    python main.py                      # Kamera, 3 Spieler, OSC + interne Engine
    python main.py --performers 1       # Solo Modus zum Ueben
    python main.py --mock --no-window   # Testlauf ohne Kamera
"""

import os
import sys

# Log Rauschen von OpenCV und MediaPipe daempfen
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
