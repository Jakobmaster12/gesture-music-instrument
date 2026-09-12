"""Das Erscheinungsbild: Farben, Schrift, Flaechen.

Startbildschirm und Spielanzeige teilen sich dieses Modul. Alles, was
zweimal gleich aussehen soll - Schriftgroessen, Abstaende, Ecken, das
Glas der Karten - steht genau hier und nirgends sonst.

Zwei Dinge tragen den Eindruck:

    Schrift   Gezeichnet wird mit einer echten Schriftart ueber Pillow,
              nicht mit den eingebauten Strichfonts von OpenCV. Die
              Strichfonts kennen keine Kerning- und Formdetails; auf
              einem Projektor sieht man ihnen das sofort an.
    Licht     Karten sind Milchglas mit einer Lichtkante oben und einem
              weichen Schatten darunter. Dadurch liegen sie auf dem Bild
              statt darin zu kleben.

Wenn Pillow oder eine brauchbare Schrift fehlt, faellt die Anzeige
automatisch auf die OpenCV Strichfonts zurueck. Sie sieht dann schlichter
aus, laeuft aber weiter - auf einer Buehne ist ein haesslicher Text
allemal besser als ein Absturz.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # Pillow steckt ohnehin in den Abhaengigkeiten von mediapipe
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:  # pragma: no cover - nur auf sehr nackten Systemen
    HAVE_PIL = False


# ----------------------------------------------------------------------
# Farben
# ----------------------------------------------------------------------
def rgb(value: int) -> Tuple[int, int, int]:
    """0xRRGGBB als BGR Tupel, weil OpenCV in BGR zeichnet."""
    return ((value) & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF)


# Neutrale Reihe, leicht kuehl. Der Abstand zwischen den Stufen ist
# bewusst gross: so bleibt die Hierarchie auch auf einem blassen Beamer
# lesbar.
INK = rgb(0xF3F5F9)          # Ueberschriften, Werte
INK_SOFT = rgb(0xAEB4C2)     # normaler Text
INK_FAINT = rgb(0x71778A)    # Beschriftungen
LINE = rgb(0x333846)         # Haarlinie auf Glas
LINE_SOFT = rgb(0x22262F)    # Spur unter einem Balken
PANEL = rgb(0x14161C)        # Glasfarbe
PANEL_HI = rgb(0x1D212A)     # Glas einer ausgewaehlten Zeile
EDGE = rgb(0x3C4250)         # Lichtkante oben auf dem Glas

ACCENT = rgb(0x7AA2FF)       # ein einziger Akzent, kuehles Blau
ACCENT_DIM = rgb(0x3D5488)
DANGER = rgb(0xFF6B6B)       # Loop steht vor dem Wegwerfen
BACKDROP = rgb(0x0B0C10)     # Grund hinter allem

# Schriftgroessen in Pixel, gemessen auf 720p. Alles andere wird ueber
# den Faktor `unit()` mitskaliert, damit Vollbild und Fenster gleich
# aussehen.
TYPE_HERO = 46
TYPE_TITLE = 21
TYPE_HEAD = 16
TYPE_BODY = 14
TYPE_LABEL = 11
TYPE_MICRO = 10

TRACK_LABEL = 1.6            # Sperrung der Versalien
TRACK_TITLE = 0.6
TRACK_HERO = 2.4

RADIUS = 14                  # Eckenradius einer Karte auf 720p


def unit(height: int) -> float:
    """Skalierungsfaktor der Anzeige, 1.0 entspricht 720p."""
    return max(0.55, height / 720.0)


def scale(colour: Sequence[int], factor: float) -> Tuple[int, int, int]:
    return tuple(int(min(255, max(0, c * factor))) for c in colour)


def mix(a: Sequence[int], b: Sequence[int], t: float) -> Tuple[int, int, int]:
    t = min(1.0, max(0.0, t))
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def ease_out(value: float) -> float:
    return 1.0 - (1.0 - clamp01(value)) ** 3


# ----------------------------------------------------------------------
# Schrift
# ----------------------------------------------------------------------
# Pro Schnitt eine Liste von Kandidaten: erst die Systemschriften, die
# gut aussehen, am Ende die Schrift, die matplotlib mitbringt. Der erste
# Treffer gewinnt.
FONT_CANDIDATES: Dict[str, Tuple[Tuple[str, int], ...]] = {
    "light": (
        ("/System/Library/Fonts/HelveticaNeue.ttc", 7),
        ("/System/Library/Fonts/Avenir Next.ttc", 7),
        ("C:/Windows/Fonts/segoeuil.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ),
    "regular": (
        ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
        ("/System/Library/Fonts/Avenir Next.ttc", 7),
        ("C:/Windows/Fonts/segoeui.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ),
    "medium": (
        ("/System/Library/Fonts/HelveticaNeue.ttc", 10),
        ("/System/Library/Fonts/Avenir Next.ttc", 5),
        ("C:/Windows/Fonts/seguisb.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ),
    "bold": (
        ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
        ("/System/Library/Fonts/Avenir Next.ttc", 0),
        ("C:/Windows/Fonts/segoeuib.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
    ),
}

# OpenCV Ersatz, falls keine echte Schrift zu finden ist.
_HERSHEY = {
    "light": cv2.FONT_HERSHEY_SIMPLEX,
    "regular": cv2.FONT_HERSHEY_SIMPLEX,
    "medium": cv2.FONT_HERSHEY_SIMPLEX,
    "bold": cv2.FONT_HERSHEY_DUPLEX,
}


def _matplotlib_fonts() -> List[Tuple[str, int]]:
    """DejaVu aus matplotlib - liegt auf fast jedem Rechner mit numpy Stack."""
    try:
        import matplotlib
    except Exception:
        return []
    base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    return [(os.path.join(base, "DejaVuSans.ttf"), 0),
            (os.path.join(base, "DejaVuSans-Bold.ttf"), 0)]


class Typeset:
    """Textsatz mit Maskenspeicher.

    Jede Zeichenkette wird einmal in eine Graustufenmaske gerendert und
    danach nur noch eingefaerbt ins Bild kopiert. Das ist der Grund,
    warum echte Schrift hier nichts kostet: gerendert wird pro Text
    genau einmal, nicht pro Bild.
    """

    def __init__(self) -> None:
        self._fonts: Dict[Tuple[str, int], object] = {}
        self._masks: Dict[Tuple[str, str, int, float], Tuple[np.ndarray, int]] = {}
        self._widths: Dict[Tuple[str, str, int, float], int] = {}
        self._paths: Dict[str, Optional[Tuple[str, int]]] = {}
        self.available = HAVE_PIL and self._resolve("medium") is not None

    # ------------------------------------------------------------------
    def _resolve(self, weight: str) -> Optional[Tuple[str, int]]:
        if weight in self._paths:
            return self._paths[weight]
        found = None
        extra = _matplotlib_fonts()
        fallback = extra[1:] if weight == "bold" else extra[:1]
        for path, index in tuple(FONT_CANDIDATES[weight]) + tuple(fallback):
            if os.path.exists(path):
                found = (path, index)
                break
        self._paths[weight] = found
        return found

    def _font(self, weight: str, px: int):
        key = (weight, px)
        cached = self._fonts.get(key)
        if cached is None:
            path, index = self._resolve(weight)
            cached = ImageFont.truetype(path, px, index=index)
            self._fonts[key] = cached
        return cached

    # ------------------------------------------------------------------
    def _mask(self, text: str, weight: str, px: int, tracking: float):
        """Graustufenmaske eines Textes plus Abstand zur Grundlinie."""
        key = (text, weight, px, round(tracking, 2))
        cached = self._masks.get(key)
        if cached is not None:
            return cached

        font = self._font(weight, px)
        ascent, descent = font.getmetrics()
        advances = [font.getlength(char) for char in text]
        width = int(math.ceil(sum(advances) + tracking * max(len(text) - 1, 0))) + 4
        height = ascent + descent + 2
        image = Image.new("L", (max(width, 1), max(height, 1)), 0)
        draw = ImageDraw.Draw(image)
        cursor = 2.0
        for char, advance in zip(text, advances):
            draw.text((cursor, 1), char, font=font, fill=255, anchor="la")
            cursor += advance + tracking
        mask = np.asarray(image, dtype=np.float32) / 255.0
        cached = (mask, ascent + 1)
        # Der Speicher bleibt klein: die Anzeige zeigt immer wieder
        # dieselben Woerter.
        if len(self._masks) > 900:
            self._masks.clear()
        self._masks[key] = cached
        return cached

    # ------------------------------------------------------------------
    def width(self, text: str, px: int, weight: str = "medium",
              tracking: float = 0.0) -> int:
        """Textbreite. Gemessen wird jede Zeichenkette nur einmal.

        Die Anzeige fragt pro Bild dutzende Breiten ab, um mittig oder
        rechtsbuendig zu setzen. Ohne diesen Speicher liefe ein gutes
        Drittel der Rechenzeit in die Schriftmetrik.
        """
        if not text:
            return 0
        px = int(px)
        key = (text, weight, px, round(tracking, 2))
        cached = self._widths.get(key)
        if cached is not None:
            return cached
        if not self.available:
            (w, _), _ = cv2.getTextSize(text, _HERSHEY[weight], px / 26.0, 1)
            value = int(w + tracking * max(len(text) - 1, 0))
        else:
            font = self._font(weight, px)
            value = int(round(sum(font.getlength(c) for c in text)
                              + tracking * max(len(text) - 1, 0)))
        if len(self._widths) > 4000:
            self._widths.clear()
        self._widths[key] = value
        return value

    def draw(
        self,
        canvas: np.ndarray,
        text: str,
        x: int,
        baseline: int,
        px: int,
        colour: Sequence[int] = INK_SOFT,
        weight: str = "medium",
        tracking: float = 0.0,
        align: str = "left",
        alpha: float = 1.0,
    ) -> int:
        """Zeichnet Text auf die Grundlinie. Gibt die rechte Kante zurueck."""
        if not text:
            return x
        px = max(6, int(round(px)))
        width = self.width(text, px, weight, tracking)
        if align == "center":
            x -= width // 2
        elif align == "right":
            x -= width

        if not self.available:
            cv2.putText(canvas, text, (int(x), int(baseline)), _HERSHEY[weight],
                        px / 26.0, colour, 1, cv2.LINE_AA)
            return int(x + width)

        mask, ascent = self._mask(text, weight, px, tracking)
        self._blit(canvas, mask, int(x), int(baseline) - ascent, colour, alpha)
        return int(x + width)

    @staticmethod
    def _blit(canvas: np.ndarray, mask: np.ndarray, x: int, y: int,
              colour: Sequence[int], alpha: float) -> None:
        h_img, w_img = canvas.shape[:2]
        mh, mw = mask.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w_img, x + mw), min(h_img, y + mh)
        if x1 <= x0 or y1 <= y0:
            return
        patch = mask[y0 - y:y1 - y, x0 - x:x1 - x, None]
        if alpha < 1.0:
            patch = patch * alpha
        roi = canvas[y0:y1, x0:x1]
        tint = np.array(colour, dtype=np.float32)
        roi[:] = (roi.astype(np.float32) * (1.0 - patch) + tint * patch).astype(np.uint8)


TYPE = Typeset()


# Bequeme Kurzformen. `label` ist die gesperrte Versalzeile, die die
# ganze Oberflaeche zusammenhaelt.
def text(canvas, value, x, baseline, px=TYPE_BODY, colour=INK_SOFT,
         weight="medium", tracking=0.0, align="left", alpha=1.0) -> int:
    return TYPE.draw(canvas, value, x, baseline, px, colour, weight, tracking,
                     align, alpha)


def label(canvas, value, x, baseline, px=TYPE_LABEL, colour=INK_FAINT,
          align="left", tracking=TRACK_LABEL, weight="medium", alpha=1.0) -> int:
    return TYPE.draw(canvas, value.upper(), x, baseline, px, colour, weight,
                     tracking, align, alpha)


def text_width(value, px=TYPE_BODY, weight="medium", tracking=0.0) -> int:
    return TYPE.width(value, int(px), weight, tracking)


def label_width(value, px=TYPE_LABEL, weight="medium", tracking=TRACK_LABEL) -> int:
    return TYPE.width(value.upper(), int(px), weight, tracking)


# ----------------------------------------------------------------------
# Flaechen
# ----------------------------------------------------------------------
def rounded_path(x: int, y: int, w: int, h: int, radius: int) -> np.ndarray:
    """Polygonzug eines abgerundeten Rechtecks, fuer Flaeche und Kontur."""
    radius = int(max(0, min(radius, min(w, h) // 2)))
    if radius <= 0:
        return np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32
        )
    steps = 10
    points: List[Tuple[int, int]] = []
    corners = (
        (x + w - radius, y + radius, -90.0),
        (x + w - radius, y + h - radius, 0.0),
        (x + radius, y + h - radius, 90.0),
        (x + radius, y + radius, 180.0),
    )
    for cx, cy, start in corners:
        for step in range(steps + 1):
            angle = math.radians(start + 90.0 * step / steps)
            points.append((int(round(cx + radius * math.cos(angle))),
                           int(round(cy + radius * math.sin(angle)))))
    return np.array(points, dtype=np.int32)


class Painter:
    """Glas, Schatten und Konturen - mit Speicher fuer die Masken.

    Die Masken haengen nur an Groesse und Radius. Einmal gebaut, lassen
    sie sich Bild fuer Bild wiederverwenden; das haelt die Anzeige auch
    im Vollbild fluessig.
    """

    def __init__(self) -> None:
        self._masks: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._inverse: Dict[Tuple[int, int, int, float], np.ndarray] = {}
        self._shadows: Dict[Tuple[int, int, int, int, float], np.ndarray] = {}
        self._ramps: Dict[Tuple[int, int, Tuple[int, int, int]], np.ndarray] = {}
        self._glows: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    def mask(self, w: int, h: int, radius: int) -> np.ndarray:
        key = (w, h, radius)
        cached = self._masks.get(key)
        if cached is None:
            buffer = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(buffer, [rounded_path(0, 0, w - 1, h - 1, radius)], 255)
            # Eine Spur Weichzeichnung nimmt der Kante das Treppige.
            buffer = cv2.GaussianBlur(buffer, (3, 3), 0)
            cached = (buffer.astype(np.float32) / 255.0)[:, :, None]
            self._masks[key] = cached
        return cached

    def _shadow_factor(self, w: int, h: int, radius: int, spread: int,
                       strength: float) -> np.ndarray:
        """Fertiger Multiplikator 1 - Schatten. Nur einmal pro Kartengroesse."""
        key = (w, h, radius, spread, round(strength, 2))
        cached = self._shadows.get(key)
        if cached is None:
            buffer = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(
                buffer,
                [rounded_path(spread, spread, w - 2 * spread - 1,
                              h - 2 * spread - 1, radius)],
                255,
            )
            size = spread * 2 + 1
            buffer = cv2.GaussianBlur(buffer, (size, size), spread * 0.55)
            factor = 1.0 - (buffer.astype(np.float32) / 255.0) * strength
            cached = cv2.merge([factor, factor, factor])
            self._shadows[key] = cached
        return cached

    def _panel_base(self, w: int, h: int, tint: Sequence[int]) -> np.ndarray:
        """Die eingefaerbte Glasflaeche samt Gefaelle von oben nach unten."""
        key = (w, h, tuple(int(c) for c in tint))
        cached = self._ramps.get(key)
        if cached is None:
            ramp = np.linspace(1.16, 0.90, h, dtype=np.float32)[:, None, None]
            base = np.clip(np.array(tint, dtype=np.float32) * ramp, 0, 255)
            cached = np.broadcast_to(base, (h, w, 3)).astype(np.uint8)
            if len(self._ramps) > 120:
                self._ramps.clear()
            self._ramps[key] = cached
        return cached

    def _inverse_mask(self, w: int, h: int, radius: int, alpha: float) -> np.ndarray:
        key = (w, h, radius, round(alpha, 2))
        cached = self._inverse.get(key)
        if cached is None:
            mask = self.mask(w, h, radius)[:, :, 0] * alpha
            cached = (cv2.merge([mask, mask, mask]),
                      cv2.merge([1.0 - mask, 1.0 - mask, 1.0 - mask]))
            if len(self._inverse) > 240:
                self._inverse.clear()
            self._inverse[key] = cached
        return cached

    # ------------------------------------------------------------------
    def shadow(self, canvas: np.ndarray, rect: Tuple[int, int, int, int],
               radius: int, spread: int = 18, strength: float = 0.55,
               drop: int = 6) -> None:
        """Weicher Schatten unter einer Karte. Erst er macht das Glas plastisch."""
        x, y, w, h = rect
        x, y = x - spread, y - spread + drop
        w, h = w + 2 * spread, h + 2 * spread
        h_img, w_img = canvas.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w_img, x + w), min(h_img, y + h)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        factor = self._shadow_factor(w, h, radius + spread // 2, spread, strength)
        patch = factor[y0 - y:y1 - y, x0 - x:x1 - x]
        roi = canvas[y0:y1, x0:x1]
        cv2.multiply(roi, patch, dst=roi, dtype=cv2.CV_8U)

    def glass(
        self,
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        radius: int = RADIUS,
        tint: Sequence[int] = PANEL,
        alpha: float = 0.94,
        blur: bool = True,
        edge: bool = True,
    ) -> None:
        """Milchglasflaeche: Hintergrund weichzeichnen, abdunkeln, einsetzen."""
        h_img, w_img = canvas.shape[:2]
        x, y, w, h = rect
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w_img, x + w), min(h_img, y + h)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        roi = canvas[y0:y1, x0:x1]
        if blur:
            small = cv2.resize(
                roi, (max(2, (x1 - x0) // 12), max(2, (y1 - y0) // 12)),
                interpolation=cv2.INTER_AREA,
            )
            frosted = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        else:
            frosted = roi
        # Glasfarbe mit sanftem Gefaelle liegt fertig im Speicher, das
        # Kamerabild kommt nur noch schwach durch.
        base = self._panel_base(x1 - x0, y1 - y0, tint)
        panel = cv2.addWeighted(frosted, 0.22, base, 0.78, 0.0)
        mask, inverse = self._inverse_mask(x1 - x0, y1 - y0, radius, alpha)
        cv2.add(
            cv2.multiply(roi, inverse, dtype=cv2.CV_8U),
            cv2.multiply(panel, mask, dtype=cv2.CV_8U),
            dst=roi,
        )
        if edge:
            cv2.line(canvas, (x0 + radius, y0 + 1), (x1 - radius, y0 + 1),
                     EDGE, 1, cv2.LINE_AA)

    def outline(
        self,
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        radius: int = RADIUS,
        colour: Sequence[int] = LINE,
        thickness: int = 1,
    ) -> None:
        x, y, w, h = rect
        cv2.polylines(
            canvas, [rounded_path(x, y, w - 1, h - 1, radius)], True,
            colour, thickness, cv2.LINE_AA,
        )

    def card(
        self,
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        radius: int = RADIUS,
        border: Sequence[int] = LINE,
        tint: Sequence[int] = PANEL,
        alpha: float = 0.94,
        blur: bool = True,
        spread: int = 18,
        strength: float = 0.5,
    ) -> None:
        """Schatten, Glas und Kontur in einem Zug - so sehen alle Karten gleich aus."""
        self.shadow(canvas, rect, radius, spread=spread, strength=strength)
        self.glass(canvas, rect, radius, tint=tint, alpha=alpha, blur=blur)
        self.outline(canvas, rect, radius, border)

    def _falloff(self, radius: int) -> np.ndarray:
        """Weiches radiales Gefaelle, 1 in der Mitte, 0 am Rand.

        Frueher lag hier ein Stapel gefuellter Kreise. Auf einer glatten
        dunklen Flaeche sah man die einzelnen Ringe als Treppen - genau
        das laesst eine Oberflaeche billig wirken. Ein echtes Gefaelle
        kostet nicht mehr, weil es pro Radius nur einmal entsteht.
        """
        cached = self._glows.get(radius)
        if cached is None:
            size = radius * 2
            axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
            distance = np.sqrt(axis[None, :] ** 2 + axis[:, None] ** 2)
            fade = np.clip(1.0 - distance, 0.0, 1.0) ** 2.4
            cached = cv2.merge([fade, fade, fade])
            if len(self._glows) > 90:
                self._glows.clear()
            self._glows[radius] = cached
        return cached

    def glow(self, canvas: np.ndarray, cx: int, cy: int, radius: int,
             colour: Sequence[int], strength: float) -> None:
        """Weicher Schein, additiv ins Bild gelegt."""
        if strength <= 0.01 or radius < 3:
            return
        h, w = canvas.shape[:2]
        x, y = cx - radius, cy - radius
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, cx + radius), min(h, cy + radius)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        fade = self._falloff(radius)[y0 - y:y1 - y, x0 - x:x1 - x]
        tint = np.array(colour, dtype=np.float32) * min(strength, 1.0) * 0.75
        roi = canvas[y0:y1, x0:x1]
        cv2.add(roi, cv2.multiply(fade, tint, dtype=cv2.CV_8U), dst=roi)


# ----------------------------------------------------------------------
# Bausteine, die beide Bildschirme benutzen
# ----------------------------------------------------------------------
def meter(canvas: np.ndarray, x: int, y: int, w: int, value: float,
          colour: Sequence[int], thickness: int = 3,
          track: Sequence[int] = LINE_SOFT) -> None:
    """Schmaler Balken mit Spur. Runde Enden, sonst wirkt er abgehackt."""
    cv2.line(canvas, (x, y), (x + w, y), track, thickness, cv2.LINE_AA)
    fill = int(w * clamp01(value))
    if fill > 1:
        cv2.line(canvas, (x, y), (x + fill, y), colour, thickness, cv2.LINE_AA)
        cv2.circle(canvas, (x + fill, y), thickness // 2, colour, -1, cv2.LINE_AA)


def arc(canvas: np.ndarray, cx: int, cy: int, radius: int, progress: float,
        colour: Sequence[int], thickness: int = 2) -> None:
    """Fortschrittsring, oben beginnend."""
    if progress <= 0.005:
        return
    cv2.ellipse(canvas, (cx, cy), (radius, radius), -90.0, 0.0,
                360.0 * clamp01(progress), colour, thickness, cv2.LINE_AA)


def backdrop(canvas: np.ndarray, strength: float = 0.82) -> None:
    """Kamerabild hinter die Oberflaeche legen."""
    dark = np.full_like(canvas, BACKDROP)
    cv2.addWeighted(dark, strength, canvas, 1.0 - strength, 0, canvas)
