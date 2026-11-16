"""Digonto Audio - dialogs: effect settings with preview, export, record, cut by time, generate, statistics."""
import os

import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

from . import aio, dsp
from .view import fmt_time, parse_time

PREVIEW_SECS = 8.0


class EffectDialog(QDialog):
    """settings of one effect + Preview. run() gets (x, sr) and returns the processed piece."""

    def __init__(self, parent, title, fields, fn, x, sr, ctx, player):
        super().__init__(parent)
        self.setWindowTitle(title.rstrip("…"))
        self.fn, self.x, self.sr, self.ctx, self.player = fn, x, sr, ctx, player
        self.w = {}
        form = QFormLayout()
        for key, label, kind, default, lo, hi, extra in fields:
            if kind == "f":
                w = QDoubleSpinBox()
                w.setDecimals(2)
                w.setRange(lo, hi)
                w.setValue(default)
                w.setSingleStep(max((hi - lo) / 100.0, 0.01))
                if extra:
                    w.setSuffix(extra)
            elif kind == "i":
                w = QSpinBox()
                w.setRange(int(lo), int(hi))
                w.setValue(int(default))
                if extra:
                    w.setSuffix(extra)
            elif kind == "c":
                w = QComboBox()
                w.addItems(extra)
                w.setCurrentText(str(default))
            else:
                w = QCheckBox(label)
                w.setChecked(bool(default))
            self.w[key] = (w, kind)
            if kind == "b":
                form.addRow(w)
            else:
                form.addRow(label, w)
        v = QVBoxLayout(self)
        v.addLayout(form)
        row = QHBoxLayout()
        self.prev = QPushButton("▶ Preview")
        self.prev.setToolTip(f"Plays the first {int(PREVIEW_SECS)} seconds of the selection with this effect")
        self.prev.clicked.connect(self.preview)
        stop = QPushButton("■ Stop")
        stop.clicked.connect(lambda: self.player.stop())
        row.addWidget(self.prev)
        row.addWidget(stop)
        row.addStretch()
        v.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.setMinimumWidth(380)

    def params(self):
        out = {}
        for k, (w, kind) in self.w.items():
            out[k] = w.currentText() if kind == "c" else w.isChecked() if kind == "b" else w.value()
        return out

    def preview(self):
        n = min(len(self.x), int(self.sr * PREVIEW_SECS))
        piece = self.x[:n]
        try:
            y = self.fn(piece, self.sr, self.params(), self.ctx)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Preview", str(e))
            return
        y = np.clip(y, -1, 1)
        self.player.play(y, self.sr, 0, len(y))

    def done(self, r):
        self.player.stop()
        super().done(r)


class TimeEdit(QLineEdit):
    """time with milliseconds: 1:23.456"""

    def __init__(self, sec=0.0):
        super().__init__()
        self.setText(fmt_time(sec))
        self.setMaximumWidth(120)
        self.setAlignment(Qt.AlignRight)

    def seconds(self):
        return parse_time(self.text(), 0.0)

    def set_seconds(self, s):
        self.setText(fmt_time(s))


class CutDialog(QDialog):
    """cut / keep / copy / silence an exact piece, millisecond wise"""

    def __init__(self, parent, total_sec, a_sec, b_sec):
        super().__init__(parent)
        self.setWindowTitle("Cut by time (milliseconds)")
        self.action = None
        self.total = total_sec
        v = QVBoxLayout(self)
        v.addWidget(QLabel(f"Type exact times, for example <b>0:12.345</b> or <b>250ms</b>. Length of the sound: <b>{fmt_time(total_sec)}</b>"))
        form = QFormLayout()
        self.a, self.b = TimeEdit(a_sec), TimeEdit(b_sec)
        self.ln = TimeEdit(max(0, b_sec - a_sec))
        form.addRow("From", self.a)
        form.addRow("To", self.b)
        form.addRow("Length", self.ln)
        v.addLayout(form)
        self.a.editingFinished.connect(self._len)
        self.b.editingFinished.connect(self._len)
        self.ln.editingFinished.connect(self._end)
        grid = QVBoxLayout()
        for key, text, tip in (("delete", "Cut out this part (remove it)", "The rest joins together"),
                               ("keep", "Keep only this part", "Everything else is removed"),
                               ("select", "Just select it", ""),
                               ("copy", "Copy it", ""),
                               ("silence", "Make this part silent", ""),
                               ("new", "Make a new sound from this part", "Opens in a new tab")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self._go(k))
            grid.addWidget(b)
        v.addLayout(grid)
        c = QPushButton("Cancel")
        c.clicked.connect(self.reject)
        v.addWidget(c)

    def _len(self):
        self.ln.set_seconds(max(0.0, self.b.seconds() - self.a.seconds()))

    def _end(self):
        self.b.set_seconds(self.a.seconds() + self.ln.seconds())

    def _go(self, k):
        self.action = k
        self.accept()

    def range(self):
        a = min(max(self.a.seconds(), 0.0), self.total)
        b = min(max(self.b.seconds(), 0.0), self.total)
        return (a, b) if a <= b else (b, a)


class ExportDialog(QDialog):
    def __init__(self, parent, have_sel, n_markers, sr, ch, title=""):
        super().__init__(parent)
        self.setWindowTitle("Export / Save as…")
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.fmt = QComboBox()
        for k, (ext, _, desc) in aio.FORMATS.items():
            self.fmt.addItem(f"{k}   ({ext})", k)
        self.desc = QLabel()
        self.desc.setStyleSheet("color:#52485A;")
        self.kbps = QComboBox()
        for k in (64, 96, 128, 160, 192, 256, 320):
            self.kbps.addItem(f"{k} kbit/s", k)
        self.kbps.setCurrentIndex(4)
        self.rate = QComboBox()
        self.rate.addItem(f"Keep ({sr} Hz)", 0)
        for r in (8000, 16000, 22050, 32000, 44100, 48000, 96000):
            self.rate.addItem(f"{r} Hz", r)
        self.mono = QCheckBox("Mono (one channel)")
        self.mono.setChecked(ch == 1)
        self.mono.setEnabled(ch > 1)
        self.scope = QComboBox()
        self.scope.addItem("Everything", "all")
        if have_sel:
            self.scope.addItem("Only the selection", "sel")
            self.scope.setCurrentIndex(1)
        if n_markers:
            self.scope.addItem(f"Every part between markers ({n_markers + 1} files)", "markers")
        self.t_title, self.t_artist = QLineEdit(title), QLineEdit()
        form.addRow("Format", self.fmt)
        form.addRow("", self.desc)
        self.l_kbps = QLabel("Quality")
        form.addRow(self.l_kbps, self.kbps)
        form.addRow("Sample rate", self.rate)
        form.addRow("", self.mono)
        form.addRow("What to save", self.scope)
        self.tag_row = (QLabel("Title"), QLabel("Artist"))
        form.addRow(self.tag_row[0], self.t_title)
        form.addRow(self.tag_row[1], self.t_artist)
        v.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Save…")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.fmt.currentIndexChanged.connect(self._fmt)
        self._fmt()
        self.setMinimumWidth(420)

    def _fmt(self):
        k = self.fmt.currentData()
        self.desc.setText(aio.FORMATS[k][2])
        lossy = k in aio.LOSSY_KBPS
        self.kbps.setVisible(lossy)
        self.l_kbps.setVisible(lossy)
        tags = not k.startswith(("WAV", "AIFF"))
        self.t_title.setVisible(tags)
        self.t_artist.setVisible(tags)
        for w in self.tag_row:
            w.setVisible(tags)

    def settings(self):
        return {"fmt": self.fmt.currentData(), "kbps": self.kbps.currentData(), "rate": self.rate.currentData() or None,
                "mono": self.mono.isChecked(), "scope": self.scope.currentData(),
                "tags": {"title": self.t_title.text().strip(), "artist": self.t_artist.text().strip()}}


class NewDialog(QDialog):
    def __init__(self, parent, title="Make new sound", kinds=("silence",), tone=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.kind = QComboBox()
        self.kind.addItems(["silence", "sine", "square", "saw", "triangle", "white noise", "pink noise"] if tone else ["silence"])
        self.secs = QDoubleSpinBox()
        self.secs.setRange(0.001, 7200)
        self.secs.setDecimals(3)
        self.secs.setValue(5.0)
        self.secs.setSuffix(" s")
        self.freq = QDoubleSpinBox()
        self.freq.setRange(20, 20000)
        self.freq.setValue(440)
        self.freq.setSuffix(" Hz")
        self.level = QDoubleSpinBox()
        self.level.setRange(-60, 0)
        self.level.setValue(-6)
        self.level.setSuffix(" dB")
        self.sr = QComboBox()
        for r in (44100, 48000, 22050, 16000, 96000):
            self.sr.addItem(f"{r} Hz", r)
        self.ch = QComboBox()
        self.ch.addItem("Stereo", 2)
        self.ch.addItem("Mono", 1)
        if tone:
            form.addRow("Sound", self.kind)
            form.addRow("Pitch", self.freq)
            form.addRow("Loudness", self.level)
        form.addRow("Length", self.secs)
        self.is_new = parent is None or not hasattr(parent, "doc") or parent.doc is None
        if kinds and kinds[0] == "new":
            form.addRow("Sample rate", self.sr)
            form.addRow("Channels", self.ch)
        v.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def make(self, sr, ch):
        sr = self.sr.currentData() if self.sr.isVisibleTo(self) else sr
        ch = self.ch.currentData() if self.ch.isVisibleTo(self) else ch
        k = self.kind.currentText()
        if k == "silence":
            return dsp.silence(sr, ch, self.secs.value()), sr
        return dsp.tone(sr, ch, self.secs.value(), self.freq.value(), k, self.level.value()), sr


class StatsDialog(QDialog):
    def __init__(self, parent, st, sr, name):
        super().__init__(parent)
        self.setWindowTitle("Statistics")
        v = QVBoxLayout(self)
        lines = [("Length", fmt_time(st["seconds"])), ("Samples", f"{st['samples']:,}"), ("Sample rate", f"{sr} Hz"),
                 ("Channels", "Stereo" if st["channels"] == 2 else "Mono"),
                 ("Loudest peak", f"{st['peak_db']:.2f} dB"), ("Average loudness (RMS)", f"{st['rms_db']:.2f} dB"),
                 ("Clipped samples", f"{st['clipped']:,}" + ("   ⚠ distorted" if st["clipped"] else "")),
                 ("DC offset", "  ".join(f"{d:+.5f}" for d in st["dc"]))]
        form = QFormLayout()
        for a, b in lines:
            form.addRow(a, QLabel(f"<b>{b}</b>"))
        v.addWidget(QLabel(f"<b>{name}</b>"))
        v.addLayout(form)
        hint = ""
        if st["peak_db"] < -12:
            hint = "This sound is quiet. Try Effects > Volume > Make quiet audio louder."
        elif st["clipped"] > 10:
            hint = "The sound is clipping (too loud). Lower it with Amplify, or use Limiter."
        if hint:
            h = QLabel(hint)
            h.setWordWrap(True)
            h.setStyleSheet("color:#C2395C;")
            v.addWidget(h)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.Close).clicked.connect(self.accept)
        v.addWidget(bb)


class Meter(QWidget):
    """simple level bars (one per channel)"""

    def __init__(self):
        super().__init__()
        self.vals = [0.0, 0.0]
        self.hold = [0.0, 0.0]
        self.setMinimumHeight(46)
        self.t = QTimer(self)
        self.t.setInterval(40)
        self.t.timeout.connect(self._decay)
        self.t.start()

    def push(self, values):
        for i, v in enumerate(values[:2]):
            self.vals[i] = max(self.vals[i], float(v))
            self.hold[i] = max(self.hold[i], float(v))
        if len(values) == 1:
            self.vals[1] = self.vals[0]
            self.hold[1] = self.hold[0]
        self.update()

    def _decay(self):
        a = [v * 0.82 for v in self.vals]
        h = [v * 0.985 for v in self.hold]
        if a != self.vals or h != self.hold:
            self.vals, self.hold = a, h
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor("#1A1442"))
        bh = (H - 12) / 2
        for i in range(2):
            y = 4 + i * (bh + 4)
            v = dsp.db(self.vals[i]) if self.vals[i] > 1e-5 else -90
            frac = max(0.0, min(1.0, (v + 60) / 60.0))
            w = frac * (W - 8)
            col = QColor("#FF8FA8") if v < -12 else QColor("#FFD54A") if v < -3 else QColor("#FF5A4A")
            p.fillRect(4, int(y), int(w), int(bh), col)
            hv = dsp.db(self.hold[i]) if self.hold[i] > 1e-5 else -90
            hx = 4 + max(0.0, min(1.0, (hv + 60) / 60.0)) * (W - 8)
            p.fillRect(int(hx) - 1, int(y), 2, int(bh), QColor("#FFFFFF"))
        p.setPen(QColor("#907CA0"))
        for d in (-48, -36, -24, -12, -6, 0):
            x = 4 + (d + 60) / 60.0 * (W - 8)
            p.drawLine(int(x), H - 4, int(x), H - 1)


class RecordDialog(QDialog):
    def __init__(self, parent, recorder, devices):
        super().__init__(parent)
        self.setWindowTitle("Record")
        self.rec = recorder
        self.audio = None
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.dev = QComboBox()
        self.dev.addItem("Default microphone", None)
        for name, d in devices:
            self.dev.addItem(name, d)
        self.ch = QComboBox()
        self.ch.addItem("Mono (voice)", 1)
        self.ch.addItem("Stereo", 2)
        self.sr = QComboBox()
        for r in (44100, 48000, 22050, 16000):
            self.sr.addItem(f"{r} Hz", r)
        form.addRow("Microphone", self.dev)
        form.addRow("Channels", self.ch)
        form.addRow("Sample rate", self.sr)
        v.addLayout(form)
        self.meter = Meter()
        v.addWidget(self.meter)
        self.time = QLabel("0:00.000")
        self.time.setStyleSheet("font-size:26px; font-weight:bold; color:#C2395C;")
        self.time.setAlignment(Qt.AlignCenter)
        v.addWidget(self.time)
        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        self.msg.setStyleSheet("color:#B00020;")
        v.addWidget(self.msg)
        row = QHBoxLayout()
        self.go = QPushButton("● Record")
        self.go.setStyleSheet("QPushButton{color:#B00020; font-weight:bold;}")
        self.go.clicked.connect(self.toggle)
        row.addWidget(self.go)
        self.use = QPushButton("Use recording")
        self.use.setEnabled(False)
        self.use.clicked.connect(self.accept)
        row.addWidget(self.use)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        v.addLayout(row)
        self.rec.level.connect(lambda lv: self.meter.push([lv]))
        self.rec.failed.connect(self.msg.setText)
        self.tm = QTimer(self)
        self.tm.timeout.connect(lambda: self.time.setText(fmt_time(self.rec.frames() / max(self.rec.sr, 1))))
        self.setMinimumWidth(400)

    def toggle(self):
        if self.rec.recording:
            self.audio = self.rec.stop()
            self.tm.stop()
            self.go.setText("● Record again")
            self.use.setEnabled(len(self.audio) > 0)
            for w in (self.dev, self.ch, self.sr):
                w.setEnabled(True)
        else:
            self.msg.setText("")
            self.rec.device = self.dev.currentData()
            if self.rec.start(self.sr.currentData(), self.ch.currentData()):
                self.go.setText("■ Stop")
                self.use.setEnabled(False)
                self.tm.start(50)
                for w in (self.dev, self.ch, self.sr):
                    w.setEnabled(False)

    def done(self, r):
        if self.rec.recording:
            self.audio = self.rec.stop()
        self.tm.stop()
        try:
            self.rec.level.disconnect()
            self.rec.failed.disconnect()
        except Exception:
            pass
        super().done(r)
