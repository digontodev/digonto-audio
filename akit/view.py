"""Digonto Audio - the waveform / spectrogram editor widget."""
import re

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QHBoxLayout, QScrollBar, QSizePolicy, QVBoxLayout, QWidget

from . import dsp

BG = QColor("#1A1442")
WAVE = QColor("#FF8FA8")
WAVE_DIM = QColor("#C2395C")
GRID = QColor("#3A2060")
SEL = QColor(255, 255, 255, 46)
SEL_EDGE = QColor("#FFFFFF")
CURSOR = QColor("#FFD54A")
PLAY = QColor("#FF5A4A")
MARK = QColor("#4AA8FF")
RULER_BG = QColor("#150F3A")
RULER_TX = QColor("#AC9CB8")
RULER_H = 24


def fmt_time(sec, ms=True):
    sec = max(0.0, float(sec))
    total_ms = int(round(sec * 1000))
    h, r = divmod(total_ms, 3600000)
    m, r = divmod(r, 60000)
    s, milli = divmod(r, 1000)
    if h:
        t = f"{h}:{m:02d}:{s:02d}"
    else:
        t = f"{m}:{s:02d}"
    return t + (f".{milli:03d}" if ms else "")


def parse_time(text, default=None):
    """'1:23.456' / '83.456' / '1h2m3.5s' / '250ms' -> seconds"""
    t = str(text).strip().lower().replace(",", ".")
    if not t:
        return default
    try:
        if t.endswith("ms") and re.fullmatch(r"[\d.]+ms", t):
            return float(t[:-2]) / 1000.0
        if re.fullmatch(r"[\d.]+s?", t):
            return float(t.rstrip("s"))
        m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s?)?", t)
        if m and any(m.groups()):
            return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3) or 0)
        parts = [float(p) for p in t.split(":")]
        v = 0.0
        for p in parts:
            v = v * 60 + p
        return v
    except ValueError:
        return default


def _lut():
    stops = [(0.0, (8, 10, 24)), (0.2, (40, 20, 90)), (0.4, (120, 30, 120)), (0.6, (215, 70, 70)), (0.8, (250, 170, 40)), (1.0, (255, 252, 210))]
    out = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for (a, ca), (b, cb) in zip(stops, stops[1:]):
            if a <= t <= b:
                k = (t - a) / (b - a)
                out[i] = [int(ca[j] + (cb[j] - ca[j]) * k) for j in range(3)]
                break
    return out


LUT = _lut()


class WaveView(QWidget):
    selectionChanged = Signal()
    cursorChanged = Signal(int)
    viewChanged = Signal()
    markerAdded = Signal(float)
    scrubbed = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(180)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.x = np.zeros((0, 1), np.float32)
        self.sr = 44100
        self.start = 0.0          # first visible frame
        self.spp = 100.0          # frames per pixel
        self.sel = None           # (a, b) frames, a < b
        self.cursor = 0
        self.play_pos = None
        self.markers = []         # [(seconds, name)]
        self.mode = "wave"        # wave | spectro | both
        self.zoom_y = 1.0
        self.show_db = False
        self._drag = None
        self._cache = {}
        self._ver = 0
        self.snap_zero = False

    # -- data ------------------------------------------------------------
    def set_audio(self, x, sr, keep_view=False):
        self.x, self.sr = x, sr
        self._ver += 1
        self._cache.clear()
        if not keep_view:
            self.zoom_all()
        else:
            self.start = max(0.0, min(self.start, max(0, len(x) - 1)))
        if self.sel and (self.sel[1] > len(x)):
            self.sel = (min(self.sel[0], len(x)), len(x)) if self.sel[0] < len(x) else None
        self.cursor = min(self.cursor, len(x))
        self.update()
        self.viewChanged.emit()

    def frames(self):
        return len(self.x)

    def width_px(self):
        return max(1, self.width())

    def visible(self):
        return self.width_px() * self.spp

    def zoom_all(self):
        n = max(1, len(self.x))
        self.start = 0.0
        self.spp = max(n / max(self.width_px() - 4, 1), 0.02)
        self.update()
        self.viewChanged.emit()

    def zoom_sel(self):
        if self.sel:
            a, b = self.sel
            self.start = max(0.0, a - (b - a) * 0.04)
            self.spp = max((b - a) * 1.08 / max(self.width_px(), 1), 0.02)
            self.update()
            self.viewChanged.emit()

    def zoom_by(self, f, anchor_px=None):
        if anchor_px is None:
            anchor_px = self.width_px() / 2
        f_anchor = self.start + anchor_px * self.spp
        n = max(1, len(self.x))
        self.spp = min(max(self.spp * f, 0.02), max(n / 20.0, 1.0))
        self.start = f_anchor - anchor_px * self.spp
        self._clamp()
        self.update()
        self.viewChanged.emit()

    def _clamp(self):
        n = len(self.x)
        vis = self.visible()
        if vis >= n:
            self.start = 0.0 if vis - n < vis else 0.0
        else:
            self.start = min(max(self.start, 0.0), n - vis)

    def scroll_to(self, frame):
        self.start = frame
        self._clamp()
        self.update()
        self.viewChanged.emit()

    def ensure_visible(self, frame):
        if frame < self.start or frame > self.start + self.visible():
            self.scroll_to(frame - self.visible() * 0.2)

    def f2x(self, f):
        return (f - self.start) / self.spp

    def x2f(self, px):
        return self.start + px * self.spp

    # -- selection -------------------------------------------------------
    def set_selection(self, a, b):
        n = len(self.x)
        a, b = int(max(0, min(a, n))), int(max(0, min(b, n)))
        if a > b:
            a, b = b, a
        self.sel = (a, b) if b - a >= 1 else None
        self.update()
        self.selectionChanged.emit()

    def clear_selection(self):
        self.sel = None
        self.update()
        self.selectionChanged.emit()

    def set_cursor(self, f, emit=True):
        self.cursor = int(max(0, min(f, len(self.x))))
        self.update()
        if emit:
            self.cursorChanged.emit(self.cursor)

    def _snap(self, f):
        if not self.snap_zero or len(self.x) < 3:
            return f
        r = int(max(4, 8 * self.spp))
        a, b = max(1, int(f) - r), min(len(self.x) - 1, int(f) + r)
        if b <= a:
            return f
        m = self.x[a - 1:b + 1].mean(axis=1)
        z = np.where(np.signbit(m[:-1]) != np.signbit(m[1:]))[0]
        if not len(z):
            return f
        return a + int(z[np.argmin(np.abs(z + a - f))])

    # -- geometry --------------------------------------------------------
    def lanes(self):
        """list of (kind, channel, top, height)"""
        H = self.height() - RULER_H
        ch = self.x.shape[1]
        out = []
        if self.mode == "wave":
            h = H / ch
            for c in range(ch):
                out.append(("wave", c, RULER_H + c * h, h))
        elif self.mode == "spectro":
            out.append(("spec", 0, RULER_H, H))
        else:
            hw = H * 0.55
            h = hw / ch
            for c in range(ch):
                out.append(("wave", c, RULER_H + c * h, h))
            out.append(("spec", 0, RULER_H + hw, H - hw))
        return out

    # -- painting --------------------------------------------------------
    def paintEvent(self, ev):
        p = QPainter(self)
        W, Ht = self.width(), self.height()
        p.fillRect(0, 0, W, Ht, BG)
        n = len(self.x)
        if n == 0:
            p.setPen(RULER_TX)
            p.drawText(self.rect(), Qt.AlignCenter, "Open a sound file, record, or create a tone")
            self._ruler(p)
            return
        for kind, c, top, h in self.lanes():
            if kind == "wave":
                self._wave(p, c, top, h)
            else:
                self._spectro(p, top, h)
        self._ruler(p)
        # selection
        if self.sel:
            xa, xb = self.f2x(self.sel[0]), self.f2x(self.sel[1])
            r = QRectF(xa, RULER_H, max(xb - xa, 1), Ht - RULER_H)
            p.fillRect(r, SEL)
            p.setPen(QPen(SEL_EDGE, 1))
            p.drawLine(QPointF(xa, RULER_H), QPointF(xa, Ht))
            p.drawLine(QPointF(xb, RULER_H), QPointF(xb, Ht))
        # markers
        p.setFont(QFont(self.font().family(), 8))
        for sec, name in self.markers:
            mx = self.f2x(sec * self.sr)
            if -50 < mx < W + 50:
                p.setPen(QPen(MARK, 1, Qt.DashLine))
                p.drawLine(QPointF(mx, RULER_H), QPointF(mx, Ht))
                p.setBrush(MARK)
                p.setPen(Qt.NoPen)
                p.drawPolygon(QPolygonF([QPointF(mx - 5, 0), QPointF(mx + 5, 0), QPointF(mx, 9)]))
                p.setPen(MARK)
                p.drawText(QPointF(mx + 6, 20), name)
        # cursor
        cx = self.f2x(self.cursor)
        p.setPen(QPen(CURSOR, 1))
        p.drawLine(QPointF(cx, RULER_H), QPointF(cx, Ht))
        p.setBrush(CURSOR)
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([QPointF(cx - 5, RULER_H - 9), QPointF(cx + 5, RULER_H - 9), QPointF(cx, RULER_H)]))
        if self.play_pos is not None:
            px = self.f2x(self.play_pos)
            p.setPen(QPen(PLAY, 2))
            p.drawLine(QPointF(px, RULER_H), QPointF(px, Ht))

    def _ruler(self, p):
        W = self.width()
        p.fillRect(0, 0, W, RULER_H, RULER_BG)
        p.setFont(QFont(self.font().family(), 8))
        sec_per_px = self.spp / self.sr
        steps = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600]
        step = next((s for s in steps if s / sec_per_px >= 90), 3600)
        t0 = self.start / self.sr
        k0 = int(t0 / step)
        k = k0
        while True:
            t = k * step
            px = (t - t0) / sec_per_px
            if px > W:
                break
            if px >= -50:
                p.setPen(RULER_TX)
                p.drawLine(QPointF(px, RULER_H - 7), QPointF(px, RULER_H))
                lab = fmt_time(t, ms=step < 1) if step >= 0.001 else f"{t:.4f}s"
                p.drawText(QPointF(px + 3, RULER_H - 10), lab)
                p.setPen(GRID)
                p.drawLine(QPointF(px, RULER_H), QPointF(px, self.height()))
            sub = step / 5
            for j in range(1, 5):
                sx = (t + j * sub - t0) / sec_per_px
                if 0 <= sx <= W:
                    p.setPen(RULER_TX)
                    p.drawLine(QPointF(sx, RULER_H - 3), QPointF(sx, RULER_H))
            k += 1

    def _wave(self, p, c, top, h):
        W = self.width()
        mid = top + h / 2
        amp = h / 2 * 0.92 * self.zoom_y
        p.setPen(QPen(GRID, 1))
        p.drawLine(QPointF(0, mid), QPointF(W, mid))
        a = int(max(0, self.start))
        b = int(min(len(self.x), self.start + W * self.spp + 1))
        if b <= a:
            return
        if self.spp >= 1.0:
            wpx = min(W, int((b - a) / self.spp) + 1)
            key = (self._ver, a, b, wpx)
            if key not in self._cache:
                if len(self._cache) > 6:
                    self._cache.clear()
                self._cache[key] = dsp.peaks_for(self.x, a, b, wpx)
            mins, maxs = self._cache[key]
            x0 = self.f2x(a)
            cols = len(mins)
            if self.show_db:
                f = lambda v: np.sign(v) * np.clip((np.log10(np.maximum(np.abs(v), 1e-5)) + 5) / 5, 0, 1)
                mn, mx = f(mins[:, c]), f(maxs[:, c])
            else:
                mn, mx = mins[:, c], maxs[:, c]
            xs = x0 + np.arange(cols) * ((b - a) / self.spp / max(cols, 1))
            p.setPen(QPen(WAVE, 1))
            ys0 = mid - np.clip(mx, -1, 1) * amp
            ys1 = mid - np.clip(mn, -1, 1) * amp
            for i in range(cols):
                xi = xs[i]
                y0, y1 = ys0[i], ys1[i]
                if y1 - y0 < 1:
                    y1 = y0 + 1
                p.drawLine(QPointF(xi, y0), QPointF(xi, y1))
        else:
            # sample level: draw the real samples
            idx = np.arange(a, b)
            ys = mid - np.clip(self.x[a:b, c], -1, 1) * amp
            xs = (idx - self.start) / self.spp
            pts = QPolygonF([QPointF(float(xx), float(yy)) for xx, yy in zip(xs, ys)])
            p.setPen(QPen(WAVE, 1.4))
            p.drawPolyline(pts)
            if self.spp < 0.25:
                p.setBrush(WAVE)
                for xx, yy in zip(xs, ys):
                    p.drawEllipse(QPointF(float(xx), float(yy)), 2.5, 2.5)
        p.setPen(QPen(GRID, 1))
        p.drawLine(QPointF(0, top + h), QPointF(W, top + h))

    def _spectro(self, p, top, h):
        W = self.width()
        a = int(max(0, self.start))
        b = int(min(len(self.x), self.start + W * self.spp))
        if b - a < 64 or h < 20:
            return
        n_fft = 1024
        cols = max(2, min(W, 900))
        rows = int(max(32, min(h, 360)))
        key = ("spec", self._ver, a, b, cols, rows)
        if key not in self._cache:
            if len(self._cache) > 6:
                self._cache.clear()
            m = self.x[:, 0] if self.x.shape[1] == 1 else self.x.mean(axis=1)
            centers = np.linspace(a, b, cols).astype(int)
            win = np.hanning(n_fft)
            idx = centers[:, None] + np.arange(-n_fft // 2, n_fft // 2)[None, :]
            idx = np.clip(idx, 0, len(m) - 1)
            S = np.abs(np.fft.rfft(m[idx] * win, axis=1)) / (n_fft / 4)
            D = 20 * np.log10(np.maximum(S, 1e-6))
            t = (1.0 - (np.arange(rows) + 0.5) / rows) ** 2
            bins = np.minimum((t * (D.shape[1] - 1)).astype(int), D.shape[1] - 1)
            img = D[:, bins].T                        # rows x cols
            v = np.clip((img + 95) / 80.0, 0, 1)
            rgb = LUT[(v * 255).astype(np.uint8)]
            rgb = np.ascontiguousarray(rgb)
            qi = QImage(rgb.data, cols, rows, cols * 3, QImage.Format_RGB888).copy()
            self._cache[key] = qi
        qi = self._cache[key]
        x0 = self.f2x(a)
        x1 = self.f2x(b)
        p.drawImage(QRectF(x0, top, max(x1 - x0, 1), h), qi)
        p.setPen(RULER_TX)
        p.setFont(QFont(self.font().family(), 8))
        for fq in (100, 500, 1000, 2000, 5000, 10000):
            if fq < self.sr / 2:
                t = (fq / (self.sr / 2)) ** 0.5
                y = top + h * (1 - t)
                p.drawLine(QPointF(0, y), QPointF(6, y))
                p.drawText(QPointF(8, y + 4), f"{fq // 1000}k" if fq >= 1000 else str(fq))

    # -- mouse -----------------------------------------------------------
    def _edge_at(self, px):
        if not self.sel:
            return None
        if abs(px - self.f2x(self.sel[0])) <= 5:
            return 0
        if abs(px - self.f2x(self.sel[1])) <= 5:
            return 1
        return None

    def mousePressEvent(self, e):
        if len(self.x) == 0:
            return
        px = e.position().x()
        f = self._snap(self.x2f(px))
        if e.button() in (Qt.MiddleButton, Qt.RightButton):
            self._drag = ("pan", e.position().x(), self.start)
            self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() != Qt.LeftButton:
            return
        if e.position().y() < RULER_H:
            self._drag = ("scrub",)
            self.set_cursor(f)
            self.scrubbed.emit(int(f))
            return
        edge = self._edge_at(px)
        if edge is not None:
            self._drag = ("edge", edge)
            return
        if e.modifiers() & Qt.ShiftModifier and (self.sel or self.cursor):
            anchor = self.sel[0] if self.sel and abs(f - self.sel[0]) > abs(f - self.sel[1]) else (self.sel[1] if self.sel else self.cursor)
            self._drag = ("sel", anchor)
            self.set_selection(anchor, f)
            return
        self._drag = ("sel", f)
        self.clear_selection()
        self.set_cursor(f)

    def mouseMoveEvent(self, e):
        px = e.position().x()
        f = self._snap(self.x2f(px))
        d = self._drag
        if d is None:
            if e.position().y() > RULER_H and self._edge_at(px) is not None:
                self.setCursor(Qt.SplitHCursor)
            else:
                self.unsetCursor()
            return
        if d[0] == "pan":
            self.scroll_to(d[2] - (px - d[1]) * self.spp)
        elif d[0] == "scrub":
            self.set_cursor(f)
            self.scrubbed.emit(int(f))
        elif d[0] == "edge" and self.sel:
            a, b = self.sel
            if d[1] == 0:
                self.set_selection(f, b)
            else:
                self.set_selection(a, f)
            if self.sel is None:
                self.sel = None
        elif d[0] == "sel":
            if abs(f - d[1]) >= max(1, self.spp * 2):
                self.set_selection(d[1], f)
            # auto scroll at the edges
            if px < 8:
                self.scroll_to(self.start - 20 * self.spp)
            elif px > self.width() - 8:
                self.scroll_to(self.start + 20 * self.spp)

    def mouseReleaseEvent(self, e):
        d = self._drag
        self._drag = None
        self.unsetCursor()
        if d and d[0] == "sel" and self.sel is None:
            self.set_cursor(d[1])

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton and len(self.x) and e.position().y() > RULER_H:
            self.set_selection(0, len(self.x))

    def wheelEvent(self, e):
        dy = e.angleDelta().y() or e.angleDelta().x()
        if e.modifiers() & Qt.ShiftModifier or e.angleDelta().x():
            self.scroll_to(self.start - (e.angleDelta().x() or dy) / 120.0 * self.visible() * 0.12)
        elif e.modifiers() & Qt.AltModifier:
            self.zoom_y = min(max(self.zoom_y * (1.2 if dy > 0 else 1 / 1.2), 0.25), 40.0)
            self.update()
        else:
            self.zoom_by(0.8 if dy > 0 else 1.25, e.position().x())

    def resizeEvent(self, e):
        self._clamp()
        self.viewChanged.emit()


class WaveEditor(QWidget):
    """WaveView with a scroll bar below"""

    def __init__(self):
        super().__init__()
        self.view = WaveView()
        self.bar = QScrollBar(Qt.Horizontal)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self.view, 1)
        v.addWidget(self.bar)
        self.bar.valueChanged.connect(self._bar)
        self.view.viewChanged.connect(self._sync)
        self._busy = False

    def _sync(self):
        v = self.view
        self._busy = True
        n = len(v.x)
        vis = int(v.visible())
        self.bar.setRange(0, max(0, n - vis))
        self.bar.setPageStep(max(1, vis))
        self.bar.setSingleStep(max(1, vis // 10))
        self.bar.setValue(int(v.start))
        self.bar.setVisible(n > vis)
        self._busy = False

    def _bar(self, val):
        if not self._busy:
            self.view.scroll_to(val)


def _key(self, e):
    """arrow keys move the cursor in tiny steps: Left/Right = 10 ms, Alt = 1 ms, Ctrl = 1 s, Shift = extend the selection"""
    k = e.key()
    if k not in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Home, Qt.Key_End):
        return QWidget.keyPressEvent(self, e)
    if len(self.x) == 0:
        return
    m = e.modifiers()
    step = self.sr * (0.001 if m & Qt.AltModifier else 1.0 if m & Qt.ControlModifier else 0.01)
    if k == Qt.Key_Home:
        new = 0
    elif k == Qt.Key_End:
        new = len(self.x)
    else:
        new = self.cursor + (step if k == Qt.Key_Right else -step)
    new = int(max(0, min(new, len(self.x))))
    if m & Qt.ShiftModifier:
        a = self.sel[0] if self.sel and self.cursor == self.sel[1] else (self.sel[1] if self.sel else self.cursor)
        self.set_selection(a, new)
    else:
        if self.sel:
            self.clear_selection()
    self.set_cursor(new)
    self.ensure_visible(new)
    self.update()


WaveView.keyPressEvent = _key
