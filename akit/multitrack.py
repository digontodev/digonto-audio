"""Digonto Audio - multitrack mixer: tracks with clips on a timeline, trim / split / fades, volume and pan, mix down.
Saved as a .dmix file (a zip with the mix settings and a copy of every sound that is used)."""
import copy
import json
import math
import os
import tempfile
import uuid
import zipfile

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton,
                               QScrollArea, QScrollBar, QSlider, QVBoxLayout, QWidget)

from . import aio, dsp
from .engine import Player
from .view import fmt_time

MIX_EXT = ".dmix"
LANE = 84
RULER = 26
COLORS = ["#C2395C", "#E8710A", "#1A73E8", "#12B5CB", "#7CB342", "#9334E6", "#F9AB00", "#E91E63"]


# ------------------------------------------------------------------ data
class Source:
    def __init__(self, name, x, sr, sid=None):
        self.id = sid or uuid.uuid4().hex[:10]
        self.name, self.sr = name, int(sr)
        self.x = dsp.to_stereo(x).astype(np.float32)
        self._cache = {}
        n = len(self.x)
        step = 256
        m = np.abs(self.x).max(axis=1)
        k = n // step
        self.peaks = m[:k * step].reshape(k, step).max(axis=1) if k else m.max(keepdims=True) if n else np.zeros(1, np.float32)
        self.step = step

    @property
    def seconds(self):
        return len(self.x) / self.sr

    def at(self, sr):
        if sr == self.sr:
            return self.x
        if sr not in self._cache:
            self._cache[sr] = dsp.change_samplerate(self.x, self.sr, sr).astype(np.float32)
        return self._cache[sr]


def new_track(n):
    return {"name": f"Track {n}", "vol": 0.0, "pan": 0.0, "mute": False, "solo": False, "clips": []}


class Mix:
    def __init__(self, sr=44100):
        self.sr = sr
        self.sources = {}
        self.tracks = [new_track(1), new_track(2)]

    def snap(self):
        return copy.deepcopy(self.tracks)

    def restore(self, s):
        self.tracks = copy.deepcopy(s)

    def length(self):
        e = 0.0
        for t in self.tracks:
            for c in t["clips"]:
                e = max(e, c["start"] + c["dur"])
        return e

    def clip_audio(self, c):
        s = self.sources.get(c["src"])
        if not s:
            return np.zeros((0, 2), np.float32)
        x = s.at(self.sr)
        a = int(round(c["offset"] * self.sr))
        n = int(round(c["dur"] * self.sr))
        seg = x[a:a + n].copy()
        if not len(seg):
            return seg
        seg *= 10 ** (c.get("gain", 0.0) / 20.0)
        fi, fo = int(c.get("fin", 0) * self.sr), int(c.get("fout", 0) * self.sr)
        if fi > 1:
            fi = min(fi, len(seg))
            seg[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
        if fo > 1:
            fo = min(fo, len(seg))
            seg[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)[:, None]
        return seg

    def render(self, t0=0.0, t1=None):
        t1 = self.length() if t1 is None else t1
        n0, n1 = int(t0 * self.sr), int(t1 * self.sr)
        out = np.zeros((max(0, n1 - n0), 2), np.float32)
        if not len(out):
            return out
        solo = any(t["solo"] for t in self.tracks)
        for t in self.tracks:
            if t["mute"] or (solo and not t["solo"]):
                continue
            g = 10 ** (t["vol"] / 20.0)
            ang = (t["pan"] + 1) * math.pi / 4
            gl, gr = math.cos(ang) * math.sqrt(2) * g, math.sin(ang) * math.sqrt(2) * g
            for c in t["clips"]:
                a = int(round(c["start"] * self.sr))
                if a + int(c["dur"] * self.sr) <= n0 or a >= n1:
                    continue
                seg = self.clip_audio(c)
                if not len(seg):
                    continue
                seg[:, 0] *= gl
                seg[:, 1] *= gr
                lo, hi = max(a, n0), min(a + len(seg), n1)
                if hi > lo:
                    out[lo - n0:hi - n0] += seg[lo - a:hi - a]
        return out

    # ---- file
    def save(self, path):
        used = {c["src"] for t in self.tracks for c in t["clips"]}
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
                srcs = []
                for sid in used:
                    s = self.sources[sid]
                    p = os.path.join(td, sid + ".wav")
                    aio.export(p, s.x, s.sr, "WAV (32-bit float)")
                    z.write(p, f"sources/{sid}.wav")
                    srcs.append({"id": sid, "name": s.name})
                z.writestr("mix.json", json.dumps({"app": "digonto-audio", "format": 1, "sr": self.sr, "sources": srcs, "tracks": self.tracks}))

    @classmethod
    def load(cls, path):
        m = cls()
        with zipfile.ZipFile(path) as z:
            d = json.loads(z.read("mix.json"))
            if d.get("app") != "digonto-audio":
                raise ValueError("this is not a Digonto Audio mix")
            m.sr, m.tracks = int(d["sr"]), d["tracks"]
            with tempfile.TemporaryDirectory() as td:
                for s in d["sources"]:
                    p = os.path.join(td, s["id"] + ".wav")
                    with open(p, "wb") as f:
                        f.write(z.read(f"sources/{s['id']}.wav"))
                    x, sr, _ = aio.load(p)
                    m.sources[s["id"]] = Source(s["name"], x, sr, s["id"])
        return m


# ------------------------------------------------------------------ track header
class Header(QFrame):
    changed = Signal()
    picked = Signal(int)

    def __init__(self, mix, i, color):
        super().__init__()
        self.mix, self.i = mix, i
        t = mix.tracks[i]
        self.setFixedHeight(LANE)
        self.setStyleSheet(f"Header{{background:#F7F3FA; border-bottom:1px solid #D9D0E4; border-left:5px solid {color};}}")
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 4, 6, 4)
        v.setSpacing(2)
        r = QHBoxLayout()
        self.name = QLineEdit(t["name"])
        self.name.setFixedHeight(22)
        self.name.editingFinished.connect(lambda: (t.__setitem__("name", self.name.text()), self.changed.emit()))
        self.m = QPushButton("M")
        self.s = QPushButton("S")
        for b, k, tip in ((self.m, "mute", "Mute"), (self.s, "solo", "Solo: hear only this track")):
            b.setCheckable(True)
            b.setChecked(t[k])
            b.setFixedSize(24, 22)
            b.setToolTip(tip)
            b.setStyleSheet("QPushButton{padding:0; min-width:0; border-radius:4px;} QPushButton:checked{background:#C2395C; color:white;}")
            b.toggled.connect(lambda on, k=k: (t.__setitem__(k, on), self.changed.emit()))
            r.addWidget(b)
        r.insertWidget(0, self.name, 1)
        v.addLayout(r)
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(-30, 12)
        self.vol.setValue(int(t["vol"]))
        self.vol.setToolTip("Volume (dB)")
        self.vol.valueChanged.connect(lambda x: (t.__setitem__("vol", float(x)), self.changed.emit()))
        self.pan = QSlider(Qt.Horizontal)
        self.pan.setRange(-100, 100)
        self.pan.setValue(int(t["pan"] * 100))
        self.pan.setToolTip("Left - right")
        self.pan.valueChanged.connect(lambda x: (t.__setitem__("pan", x / 100.0), self.changed.emit()))
        for lab, w in (("Vol", self.vol), ("Pan", self.pan)):
            h = QHBoxLayout()
            l = QLabel(lab)
            l.setFixedWidth(24)
            l.setStyleSheet("font-size:11px; color:#5B5278;")
            h.addWidget(l)
            h.addWidget(w, 1)
            v.addLayout(h)

    def mousePressEvent(self, e):
        self.picked.emit(self.i)


# ------------------------------------------------------------------ timeline
class MixView(QWidget):
    changed = Signal()          # clips edited
    selected = Signal()
    cursor_moved = Signal(float)
    before_edit = Signal()

    def __init__(self, mix):
        super().__init__()
        self.mix = mix
        self.pps = 60.0           # pixels per second
        self.t0 = 0.0
        self.cursor = 0.0
        self.play_t = None
        self.sel = None           # (track, clip index)
        self.drag = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumHeight(RULER + LANE * 2)

    def fit_height(self):
        self.setFixedHeight(RULER + LANE * max(1, len(self.mix.tracks)) + 2)

    def t_to_x(self, t):
        return (t - self.t0) * self.pps

    def x_to_t(self, x):
        return x / self.pps + self.t0

    def clip_rect(self, ti, c):
        return QRectF(self.t_to_x(c["start"]), RULER + ti * LANE + 4, c["dur"] * self.pps, LANE - 8)

    def hit(self, pos):
        for ti in range(len(self.mix.tracks) - 1, -1, -1):
            for ci in range(len(self.mix.tracks[ti]["clips"]) - 1, -1, -1):
                if self.clip_rect(ti, self.mix.tracks[ti]["clips"][ci]).contains(pos):
                    return ti, ci
        return None

    def selected_clip(self):
        if self.sel and self.sel[0] < len(self.mix.tracks) and self.sel[1] < len(self.mix.tracks[self.sel[0]]["clips"]):
            return self.mix.tracks[self.sel[0]]["clips"][self.sel[1]]
        return None

    # ---- painting
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#FFFFFF"))
        p.fillRect(0, 0, W, RULER, QColor("#F2EDF7"))
        # ruler
        step = 1
        for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600):
            if s * self.pps >= 70:
                step = s
                break
        else:
            step = 900
        f = QFont()
        f.setPixelSize(10)
        p.setFont(f)
        t = math.floor(self.t0 / step) * step
        while self.t_to_x(t) < W:
            x = self.t_to_x(t)
            p.setPen(QPen(QColor("#CEC8D3")))
            p.drawLine(int(x), RULER - 8, int(x), H)
            p.setPen(QColor("#5B5278"))
            p.drawText(int(x) + 3, 12, fmt_time(t, ms=step < 1))
            t += step
        for ti, tr in enumerate(self.mix.tracks):
            y = RULER + ti * LANE
            p.setPen(QPen(QColor("#E5DDEE")))
            p.drawLine(0, y + LANE, W, y + LANE)
            col = QColor(COLORS[ti % len(COLORS)])
            dim = tr["mute"]
            for ci, c in enumerate(tr["clips"]):
                r = self.clip_rect(ti, c)
                if r.right() < 0 or r.left() > W:
                    continue
                on = self.sel == (ti, ci)
                body = QColor(col)
                body.setAlpha(60 if not dim else 25)
                p.setBrush(body)
                p.setPen(QPen(col if not on else QColor("#1B1040"), 2 if on else 1))
                p.drawRoundedRect(r, 5, 5)
                s = self.mix.sources.get(c["src"])
                if s is not None:
                    self._wave(p, r, c, s, col)
                p.setPen(QColor("#1B1040"))
                p.drawText(QRectF(r.left() + 6, r.top() + 1, r.width() - 8, 14), Qt.AlignLeft | Qt.AlignVCenter, s.name if s else "?")
                for fade, left in ((c.get("fin", 0), True), (c.get("fout", 0), False)):
                    if fade > 0:
                        w = fade * self.pps
                        p.setPen(QPen(QColor("#1B1040"), 1, Qt.DashLine))
                        if left:
                            p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.left() + w, r.top()))
                        else:
                            p.drawLine(QPointF(r.right() - w, r.top()), QPointF(r.right(), r.bottom()))
        x = self.t_to_x(self.cursor)
        p.setPen(QPen(QColor("#C2395C"), 1))
        p.drawLine(int(x), 0, int(x), H)
        if self.play_t is not None:
            x = self.t_to_x(self.play_t)
            p.setPen(QPen(QColor("#1B1040"), 2))
            p.drawLine(int(x), 0, int(x), H)

    def _wave(self, p, r, c, s, col):
        x0, x1 = max(0, int(r.left())), min(self.width(), int(r.right()))
        mid = r.center().y() + 6
        amp = (r.height() - 22) / 2
        pk = s.peaks
        per_px = s.sr / self.pps / s.step          # peaks bins per pixel
        p.setPen(QPen(col, 1))
        for x in range(x0, x1):
            t = c["offset"] + (x - r.left()) / self.pps
            if t > c["offset"] + c["dur"]:
                break
            b0 = int(t * s.sr / s.step)
            b1 = max(b0 + 1, int(b0 + per_px))
            v = float(pk[b0:b1].max()) if b0 < len(pk) else 0.0
            h = min(1.0, v) * amp
            p.drawLine(x, int(mid - h), x, int(mid + h))

    # ---- mouse
    def mousePressEvent(self, e):
        self.setFocus()
        pos = e.position()
        if pos.y() < RULER:
            self.drag = ("cursor",)
            self.set_cursor(self.x_to_t(pos.x()))
            return
        h = self.hit(pos)
        self.sel = h
        self.selected.emit()
        if h and e.button() == Qt.LeftButton:
            ti, ci = h
            c = self.mix.tracks[ti]["clips"][ci]
            r = self.clip_rect(ti, c)
            mode = "left" if pos.x() - r.left() < 7 else "right" if r.right() - pos.x() < 7 else "move"
            self.before_edit.emit()
            self.drag = (mode, ti, ci, pos.x(), pos.y(), dict(c))
        else:
            self.set_cursor(self.x_to_t(pos.x()))
        self.update()

    def snap(self, t, skip=None):
        cands = [self.cursor, 0.0]
        for ti, tr in enumerate(self.mix.tracks):
            for ci, c in enumerate(tr["clips"]):
                if (ti, ci) != skip:
                    cands += [c["start"], c["start"] + c["dur"]]
        best = min(cands, key=lambda v: abs(v - t))
        return best if abs(best - t) * self.pps < 8 else t

    def mouseMoveEvent(self, e):
        pos = e.position()
        if not self.drag:
            h = self.hit(pos)
            if h:
                r = self.clip_rect(h[0], self.mix.tracks[h[0]]["clips"][h[1]])
                edge = pos.x() - r.left() < 7 or r.right() - pos.x() < 7
                self.setCursor(Qt.SizeHorCursor if edge else Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return
        if self.drag[0] == "cursor":
            self.set_cursor(self.x_to_t(pos.x()))
            return
        mode, ti, ci, x0, y0, orig = self.drag
        c = self.mix.tracks[ti]["clips"][ci]
        dt = (pos.x() - x0) / self.pps
        src = self.mix.sources.get(c["src"])
        smax = src.seconds if src else 1e9
        if mode == "move":
            c["start"] = max(0.0, self.snap(orig["start"] + dt, (ti, ci)))
            nt = min(len(self.mix.tracks) - 1, max(0, int((pos.y() - RULER) // LANE)))
            if nt != ti:
                self.mix.tracks[ti]["clips"].pop(ci)
                self.mix.tracks[nt]["clips"].append(c)
                self.drag = (mode, nt, len(self.mix.tracks[nt]["clips"]) - 1, x0, y0, orig)
                self.sel = (nt, len(self.mix.tracks[nt]["clips"]) - 1)
        elif mode == "left":
            ns = self.snap(orig["start"] + dt, (ti, ci))
            d = max(-orig["offset"], min(ns - orig["start"], orig["dur"] - 0.02))
            d = max(d, -orig["start"])
            c["start"], c["offset"], c["dur"] = orig["start"] + d, orig["offset"] + d, orig["dur"] - d
        else:
            ne = self.snap(orig["start"] + orig["dur"] + dt, (ti, ci))
            c["dur"] = max(0.02, min(ne - orig["start"], smax - orig["offset"]))
        self.update()
        self.selected.emit()

    def mouseReleaseEvent(self, e):
        if self.drag and self.drag[0] != "cursor":
            self.changed.emit()
        self.drag = None

    def wheelEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            t = self.x_to_t(e.position().x())
            k = 1.25 if e.angleDelta().y() > 0 else 0.8
            self.pps = max(2.0, min(4000.0, self.pps * k))
            self.t0 = max(0.0, t - e.position().x() / self.pps)
            self.changed.emit()
            self.update()
        else:
            self.t0 = max(0.0, self.t0 - (e.angleDelta().y() or e.angleDelta().x()) / self.pps * 0.5)
            self.update()
            self.cursor_moved.emit(self.cursor)
        e.accept()

    def set_cursor(self, t):
        self.cursor = max(0.0, t)
        self.update()
        self.cursor_moved.emit(self.cursor)


# ------------------------------------------------------------------ window
class MultitrackWindow(QMainWindow):
    def __init__(self, main, mix=None, path=None):
        super().__init__(main)
        self.main = main
        self.mix = mix or Mix()
        self.path = path
        self.undo, self.redo, self.dirty = [], [], False
        self.player = Player(self)
        self.player.position.connect(self._pos)
        self.player.stopped.connect(self._stopped)
        self._base = 0.0
        self.setWindowTitle("Multitrack - Digonto Audio")
        self.resize(1180, 620)
        self._build()
        self.rebuild()

    def _build(self):
        c = QWidget()
        v = QVBoxLayout(c)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QWidget()
        bar.setStyleSheet("QWidget#tb{background:#F2EDF7; border-bottom:1px solid #CEC8D3;}")
        bar.setObjectName("tb")
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 6, 8, 6)

        def btn(text, tip, fn, w=None):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            if w:
                b.setFixedWidth(w)
            h.addWidget(b)
            return b
        self.b_play = btn("▶", "Play / stop (Space)", self.toggle_play, 40)
        btn("⏮", "Back to the start", lambda: self.view.set_cursor(0.0), 40)
        self.lbl = QLabel("0:00.000")
        self.lbl.setStyleSheet("font-size:15px; font-weight:600; min-width:90px;")
        h.addWidget(self.lbl)
        h.addSpacing(10)
        btn("Add track", "A new empty track", self.add_track)
        self.b_add = btn("Add sound ▾", "Put a sound on the selected track at the cursor", self.add_menu)
        btn("Split", "Cut the clip at the cursor (S)", self.split)
        btn("Duplicate", "Copy the clip right behind itself", self.duplicate)
        btn("Delete", "Remove the clip (Del)", self.delete)
        h.addSpacing(10)
        btn("−", "Zoom out", lambda: self.zoom(0.8), 34)
        btn("+", "Zoom in", lambda: self.zoom(1.25), 34)
        btn("Fit", "Show the whole mix", self.fit)
        h.addStretch(1)
        btn("Mix down to a new sound", "Combine all tracks into one sound in the editor", self.mixdown)
        v.addWidget(bar)

        body = QHBoxLayout()
        body.setSpacing(0)
        self.heads = QWidget()
        self.heads.setFixedWidth(190)
        self.hl = QVBoxLayout(self.heads)
        self.hl.setContentsMargins(0, 0, 0, 0)
        self.hl.setSpacing(0)
        self.view = MixView(self.mix)
        self.view.changed.connect(self._edited)
        self.view.selected.connect(self.sync_panel)
        self.view.cursor_moved.connect(self._cursor)
        self.view.before_edit.connect(self.push)
        inner = QWidget()
        il = QHBoxLayout(inner)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(0)
        il.addWidget(self.heads, 0, Qt.AlignTop)
        il.addWidget(self.view, 1, Qt.AlignTop)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setWidget(inner)
        body.addWidget(sc, 1)
        # right panel: the clip that is selected (250 px like the other programs)
        panel = QFrame()
        panel.setFixedWidth(250)
        panel.setStyleSheet("QFrame#cp{background:white; border-left:1px solid #DDD3E9;}")
        panel.setObjectName("cp")
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(10, 10, 10, 10)
        t = QLabel("CLIP")
        t.setStyleSheet("font-weight:700; color:#5B5278; font-size:11px;")
        pl.addWidget(t)
        self.p_name = QLabel("Select a clip")
        self.p_name.setWordWrap(True)
        pl.addWidget(self.p_name)
        self.sp = {}
        for key, lab, lo, hi, step, suf in (("start", "Start", 0, 36000, 0.1, " s"), ("gain", "Volume", -40, 20, 0.5, " dB"),
                                            ("fin", "Fade in", 0, 60, 0.1, " s"), ("fout", "Fade out", 0, 60, 0.1, " s")):
            pl.addWidget(QLabel(lab))
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(2 if key != "gain" else 1)
            s.setSuffix(suf)
            s.setEnabled(False)
            s.valueChanged.connect(lambda val, k=key: self.set_clip(k, val))
            pl.addWidget(s)
            self.sp[key] = s
        pl.addStretch(1)
        tip = QLabel("Drag a clip to move it. Drag its edges to trim. Click the ruler to place the cursor. Ctrl + wheel zooms.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#6B6385; font-size:11px;")
        pl.addWidget(tip)
        body.addWidget(panel)
        v.addLayout(body, 1)
        self.setCentralWidget(c)
        mb = self.menuBar()
        fm = mb.addMenu("&Mix")
        for text, fn, key in (("Open mix…", self.open_mix, "Ctrl+O"), ("Save mix", self.save, "Ctrl+S"), ("Save mix as…", lambda: self.save(True), "Ctrl+Shift+S"),
                              ("Undo", self.do_undo, "Ctrl+Z"), ("Redo", self.do_redo, "Ctrl+Shift+Z")):
            a = fm.addAction(text)
            a.setShortcut(QKeySequence(key))
            a.triggered.connect(fn)
        for key, fn in (("Space", self.toggle_play), ("S", self.split), ("Delete", self.delete)):
            QShortcut(QKeySequence(key), self, activated=fn)

    # ---- tracks
    def rebuild(self):
        while self.hl.count():
            it = self.hl.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        sp = QWidget()
        sp.setFixedHeight(RULER)
        self.hl.addWidget(sp)
        for i in range(len(self.mix.tracks)):
            hd = Header(self.mix, i, COLORS[i % len(COLORS)])
            hd.changed.connect(self._edited)
            self.hl.addWidget(hd)
        self.hl.addStretch(1)
        self.view.fit_height()
        self.heads.setFixedHeight(RULER + LANE * len(self.mix.tracks) + 2)
        self.view.update()

    def cur_track(self):
        return self.view.sel[0] if self.view.sel else 0

    def add_track(self):
        self.push()
        self.mix.tracks.append(new_track(len(self.mix.tracks) + 1))
        self.rebuild()
        self.mark()

    def add_menu(self):
        m = QMenu(self)
        m.addAction("From a file…", self.add_file)
        docs = [d for d in self.main.docs() if d is not None]
        if docs:
            m.addSeparator()
            for d in docs:
                m.addAction("Open sound: " + d.name, lambda d=d: self.add_source(Source(d.name, d.x, d.sr)))
        m.exec(self.b_add.mapToGlobal(self.b_add.rect().bottomLeft()))

    def add_file(self):
        last = str(self.main.settings.value("last_dir", os.path.expanduser("~")))
        flt = "Sound files (" + " ".join("*." + e for e in aio.OPEN_EXTS.split()) + ");;All files (*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Add sound", last, flt)
        for p in paths:
            try:
                x, sr, _e = aio.load(p)
            except Exception as e:  # noqa
                QMessageBox.warning(self, "Add sound", f"{os.path.basename(p)}\n\n{e}")
                continue
            self.add_source(Source(os.path.splitext(os.path.basename(p))[0], x, sr))

    def add_source(self, s):
        self.push()
        if not self.mix.sources:
            self.mix.sr = max(44100, min(48000, s.sr)) if s.sr in (44100, 48000) else 44100
        self.mix.sources[s.id] = s
        ti = min(self.cur_track(), len(self.mix.tracks) - 1)
        self.mix.tracks[ti]["clips"].append({"src": s.id, "start": self.view.cursor, "offset": 0.0, "dur": s.seconds, "gain": 0.0, "fin": 0.0, "fout": 0.0})
        self.view.sel = (ti, len(self.mix.tracks[ti]["clips"]) - 1)
        self.mark()
        self.sync_panel()
        self.view.update()

    # ---- clip edits
    def split(self):
        c = self.view.selected_clip()
        if not c:
            return
        t = self.view.cursor
        if not (c["start"] + 0.02 < t < c["start"] + c["dur"] - 0.02):
            return
        self.push()
        ti = self.view.sel[0]
        d = t - c["start"]
        c2 = dict(c)
        c2["start"], c2["offset"], c2["dur"], c2["fin"] = t, c["offset"] + d, c["dur"] - d, 0.0
        c["dur"], c["fout"] = d, 0.0
        self.mix.tracks[ti]["clips"].append(c2)
        self.mark()
        self.view.update()

    def duplicate(self):
        c = self.view.selected_clip()
        if c:
            self.push()
            ti = self.view.sel[0]
            c2 = dict(c)
            c2["start"] = c["start"] + c["dur"]
            self.mix.tracks[ti]["clips"].append(c2)
            self.view.sel = (ti, len(self.mix.tracks[ti]["clips"]) - 1)
            self.mark()
            self.view.update()

    def delete(self):
        if self.view.selected_clip():
            self.push()
            ti, ci = self.view.sel
            self.mix.tracks[ti]["clips"].pop(ci)
            self.view.sel = None
            self.mark()
            self.sync_panel()
            self.view.update()

    def set_clip(self, key, val):
        c = self.view.selected_clip()
        if not c or self._syncing:
            return
        if key == "start":
            c["start"] = float(val)
        else:
            c[key] = float(val)
        self.mark()
        self.view.update()

    _syncing = False

    def sync_panel(self):
        c = self.view.selected_clip()
        self._syncing = True
        s = self.mix.sources.get(c["src"]) if c else None
        self.p_name.setText(s.name if s else "Select a clip")
        for k, sp in self.sp.items():
            sp.setEnabled(c is not None)
            sp.setValue(c.get(k, 0.0) if c else 0.0)
        self._syncing = False

    # ---- undo
    def push(self):
        self.undo.append(self.mix.snap())
        del self.undo[:-60]
        self.redo.clear()

    def do_undo(self):
        if self.undo:
            self.redo.append(self.mix.snap())
            self.mix.restore(self.undo.pop())
            self.view.sel = None
            self.rebuild()
            self.sync_panel()
            self.mark(push=False)

    def do_redo(self):
        if self.redo:
            self.undo.append(self.mix.snap())
            self.mix.restore(self.redo.pop())
            self.view.sel = None
            self.rebuild()
            self.sync_panel()
            self.mark(push=False)

    def mark(self, push=True):
        self.dirty = True
        self.setWindowTitle(("● " if self.dirty else "") + (os.path.basename(self.path) if self.path else "Untitled mix") + " - Multitrack - Digonto Audio")

    def _edited(self):
        self.mark()

    # ---- view
    def zoom(self, k):
        self.view.pps = max(2.0, min(4000.0, self.view.pps * k))
        self.view.update()

    def fit(self):
        L = max(1.0, self.mix.length())
        self.view.pps = max(2.0, (self.view.width() - 20) / L)
        self.view.t0 = 0.0
        self.view.update()

    def _cursor(self, t):
        self.lbl.setText(fmt_time(t))

    # ---- play
    def toggle_play(self):
        if self.player.playing:
            self.player.stop()
            self._stopped()
            return
        t0 = self.view.cursor
        x = self.mix.render(t0)
        if not len(x):
            return
        pk = float(np.abs(x).max())
        if pk > 1.0:
            x = x / pk
        self._base = t0
        if self.player.play(x, self.mix.sr, 0, len(x)):
            self.b_play.setText("⏸")

    def _pos(self, f):
        self.view.play_t = self._base + f / self.mix.sr
        self.lbl.setText(fmt_time(self.view.play_t))
        self.view.update()

    def _stopped(self):
        self.b_play.setText("▶")
        self.view.play_t = None
        self.view.update()

    # ---- mix down / files
    def mixdown(self):
        x = self.mix.render(0.0)
        if not len(x):
            QMessageBox.information(self, "Mix down", "There is nothing to mix yet. Add a sound first.")
            return
        pk = float(np.abs(x).max())
        note = ""
        if pk > 0.99:
            x = x * (0.98 / pk)
            note = "  (the mix was a little too loud, so it was made quieter to avoid distortion)"
        self.main.add_sound(x.astype(np.float32), self.mix.sr, "Mix")
        self.main.statusBar().showMessage("Mixed down to a new sound" + note, 8000)
        self.main.raise_()
        self.main.activateWindow()

    def save(self, as_new=False):
        p = self.path
        if as_new or not p:
            p, _ = QFileDialog.getSaveFileName(self, "Save mix", (self.path or "mix") if self.path else "mix" + MIX_EXT, "Digonto Audio mix (*.dmix)")
            if not p:
                return False
            if not p.lower().endswith(MIX_EXT):
                p += MIX_EXT
        try:
            self.mix.save(p)
        except Exception as e:  # noqa
            QMessageBox.warning(self, "Save mix", str(e))
            return False
        self.path, self.dirty = p, False
        self.mark()
        self.dirty = False
        self.setWindowTitle(os.path.basename(p) + " - Multitrack - Digonto Audio")
        return True

    def open_mix(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open mix", "", "Digonto Audio mix (*.dmix)")
        if p:
            try:
                m = Mix.load(p)
            except Exception as e:  # noqa
                QMessageBox.warning(self, "Open mix", str(e))
                return
            self.mix, self.path = m, p
            self.view.mix = m
            self.undo.clear()
            self.view.sel = None
            self.rebuild()
            self.sync_panel()
            self.dirty = False
            self.setWindowTitle(os.path.basename(p) + " - Multitrack - Digonto Audio")

    def closeEvent(self, ev):
        self.player.stop()
        if self.dirty and any(t["clips"] for t in self.mix.tracks):
            r = QMessageBox.question(self, "Close", "Save this mix first?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if r == QMessageBox.Cancel or (r == QMessageBox.Save and not self.save()):
                ev.ignore()
                return
        ev.accept()
