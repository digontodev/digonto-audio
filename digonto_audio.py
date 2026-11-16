"""Digonto Audio - record, edit and clean up sound. Equalizer, echo, noise remover, louder, voice/music split, millisecond cuts.
Ubuntu, macOS, Windows. PySide6 + numpy (+ a private ffmpeg for MP3 / OGG / FLAC / M4A files)."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSlider, QTabWidget, QToolBar, QVBoxLayout, QWidget)

import digonto_ui as ui
from akit import __version__ as VERSION
from akit import aio, dsp
from akit.dialogs import (CutDialog, EffectDialog, ExportDialog, Meter, NewDialog, RecordDialog, StatsDialog, TimeEdit)
from akit.effects import EFFECTS
from akit.engine import Player, Recorder, devices_in, devices_out
from akit.view import WaveEditor, fmt_time, parse_time

APP_NAME = "Digonto Audio"
SLUG = "digonto-audio"
MAX_STEPS = 40


class Doc:
    """one sound (one tab) with its undo history"""

    def __init__(self, x, sr, name, markers=None, path=None, src=None):
        self.sr, self.name, self.path, self.src = sr, name, path, src
        self.x = x
        self.markers = list(markers or [])
        self.hist = [("Start", x, list(self.markers))]
        self.hi = 0
        self.saved = 0 if path else -1
        self.editor = WaveEditor()
        self.editor.doc = self

    @property
    def dirty(self):
        return self.hi != self.saved

    def commit(self, label, x, markers=None):
        del self.hist[self.hi + 1:]
        self.x = x
        if markers is not None:
            self.markers = list(markers)
        self.hist.append((label, x, list(self.markers)))
        self.hi = len(self.hist) - 1
        total = sum(h[1].nbytes for h in self.hist)
        while (len(self.hist) > MAX_STEPS or total > 2.5e9) and len(self.hist) > 2:
            total -= self.hist[0][1].nbytes
            self.hist.pop(0)
            self.hi -= 1
            if self.saved >= 0:
                self.saved -= 1

    def jump(self, i):
        i = max(0, min(i, len(self.hist) - 1))
        self.hi = i
        self.x = self.hist[i][1]
        self.markers = list(self.hist[i][2])


class EQPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        self.preset = QComboBox()
        self.preset.addItems(list(dsp.EQ_PRESETS))
        self.preset.currentTextChanged.connect(self.load_preset)
        v.addWidget(self.preset)
        row = QHBoxLayout()
        row.setSpacing(2)
        self.sl, self.lb = [], []
        for f in dsp.EQ_BANDS:
            col = QVBoxLayout()
            col.setSpacing(1)
            val = QLabel("0")
            val.setAlignment(Qt.AlignCenter)
            val.setStyleSheet("font-size:10px; color:#C2395C;")
            s = QSlider(Qt.Vertical)
            s.setRange(-15, 15)
            s.setValue(0)
            s.setMinimumHeight(110)
            s.setTickPosition(QSlider.NoTicks)
            s.valueChanged.connect(lambda v_, l=val: l.setText(f"{v_:+d}" if v_ else "0"))
            name = QLabel(f"{f // 1000}k" if f >= 1000 else str(f))
            name.setAlignment(Qt.AlignCenter)
            name.setStyleSheet("font-size:9px;")
            name.setMinimumWidth(0)
            col.addWidget(val)
            col.addWidget(s, 1, Qt.AlignHCenter)
            col.addWidget(name)
            row.addLayout(col)
            self.sl.append(s)
        v.addLayout(row)
        b = QHBoxLayout()
        pv = QPushButton("▶ Preview")
        pv.clicked.connect(self.preview)
        ap = QPushButton("Apply")
        ap.setStyleSheet("QPushButton{background:qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF6F8E, stop:1 #FFB05A); color:#2A1241; font-weight:bold; border-radius:5px; padding:5px 10px;}")
        ap.clicked.connect(self.apply)
        rs = QPushButton("Reset")
        rs.clicked.connect(lambda: self.preset.setCurrentText("Flat") or self.load_preset("Flat"))
        for w in (pv, ap, rs):
            b.addWidget(w)
        v.addLayout(b)
        hint = QLabel("Applies to the selection, or to everything when nothing is selected. dB per band.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#52485A; font-size:11px;")
        v.addWidget(hint)

    def gains(self):
        return [s.value() for s in self.sl]

    def load_preset(self, name):
        for s, g in zip(self.sl, dsp.EQ_PRESETS.get(name, [0] * 10)):
            s.setValue(int(g))

    def preview(self):
        self.main.preview_fn(lambda x, sr: dsp.eq_graphic(x, sr, self.gains()))

    def apply(self):
        g = self.gains()
        self.main.apply_effect("Equalizer", lambda x, sr: dsp.eq_graphic(x, sr, g))


class SelectionPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        g = QVBoxLayout()
        self.a, self.b, self.l = TimeEdit(), TimeEdit(), TimeEdit()
        for lab, w in (("Start", self.a), ("End", self.b), ("Length", self.l)):
            r = QHBoxLayout()
            r.addWidget(QLabel(lab))
            r.addStretch()
            r.addWidget(w)
            g.addLayout(r)
        v.addLayout(g)
        self.info = QLabel("")
        self.info.setStyleSheet("color:#52485A; font-size:11px;")
        v.addWidget(self.info)
        self.a.editingFinished.connect(lambda: self.set_from("a"))
        self.b.editingFinished.connect(lambda: self.set_from("b"))
        self.l.editingFinished.connect(lambda: self.set_from("l"))
        row = QHBoxLayout()
        for text, fn in (("Select all", main.select_all), ("Zoom to it", main.zoom_sel)):
            bt = QPushButton(text)
            bt.clicked.connect(fn)
            row.addWidget(bt)
        v.addLayout(row)
        cut = QPushButton("Cut by time…")
        cut.setToolTip("Exact start and end in milliseconds")
        cut.clicked.connect(main.cut_by_time)
        v.addWidget(cut)

    def refresh(self, doc):
        if not doc:
            return
        v = doc.editor.view
        sr = doc.sr
        if v.sel:
            a, b = v.sel
        else:
            a = b = v.cursor
        for w, val in ((self.a, a / sr), (self.b, b / sr), (self.l, (b - a) / sr)):
            w.blockSignals(True)
            w.set_seconds(val)
            w.blockSignals(False)
        self.info.setText(f"samples {a:,} – {b:,}  ({b - a:,})")

    def set_from(self, which):
        d = self.main.doc
        if not d:
            return
        sr = d.sr
        a, b = self.a.seconds(), self.b.seconds()
        if which == "l":
            b = a + self.l.seconds()
        d.editor.view.set_selection(int(a * sr), int(b * sr))
        if not d.editor.view.sel:
            d.editor.view.set_cursor(int(a * sr))


class MarkerPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        self.lst = QListWidget()
        self.lst.setMinimumHeight(90)
        self.lst.itemDoubleClicked.connect(self.go)
        v.addWidget(self.lst)
        r = QHBoxLayout()
        for text, fn in (("Add (M)", main.add_marker), ("Rename", self.rename), ("Delete", self.delete)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            r.addWidget(b)
        v.addLayout(r)

    def refresh(self, doc):
        self.lst.clear()
        if doc:
            for sec, name in sorted(doc.markers):
                it = QListWidgetItem(f"{fmt_time(sec)}   {name}")
                it.setData(Qt.UserRole, (sec, name))
                self.lst.addItem(it)

    def go(self, it):
        d = self.main.doc
        sec, _ = it.data(Qt.UserRole)
        d.editor.view.set_cursor(int(sec * d.sr))
        d.editor.view.ensure_visible(int(sec * d.sr))

    def rename(self):
        it = self.lst.currentItem()
        d = self.main.doc
        if not it or not d:
            return
        sec, name = it.data(Qt.UserRole)
        new, ok = QInputDialog.getText(self, "Marker", "Name", text=name)
        if ok:
            d.markers = [(s, new if s == sec and n == name else n) for s, n in d.markers]
            d.hist[d.hi] = (d.hist[d.hi][0], d.x, list(d.markers))
            self.main.refresh_markers()

    def delete(self):
        it = self.lst.currentItem()
        d = self.main.doc
        if not it or not d:
            return
        sec, name = it.data(Qt.UserRole)
        d.markers = [(s, n) for s, n in d.markers if not (s == sec and n == name)]
        d.hist[d.hi] = (d.hist[d.hi][0], d.x, list(d.markers))
        self.main.refresh_markers()


class HistoryPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        self.lst = QListWidget()
        self.lst.setMinimumHeight(120)
        self.lst.currentRowChanged.connect(self.pick)
        v.addWidget(self.lst)
        self._busy = False

    def refresh(self, doc):
        self._busy = True
        self.lst.clear()
        if doc:
            for i, h in enumerate(doc.hist):
                self.lst.addItem(h[0])
            self.lst.setCurrentRow(doc.hi)
        self._busy = False

    def pick(self, row):
        if not self._busy and row >= 0 and self.main.doc and row != self.main.doc.hi:
            self.main.doc.jump(row)
            self.main.refresh_audio()


class Main(QMainWindow):
    def __init__(self, paths=()):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.icon = ui.app_icon(HERE, SLUG)
        self.setWindowIcon(self.icon)
        self.setAcceptDrops(True)
        self.settings = QSettings("Digonto", "Audio")
        self.player = Player(self)
        self.player.position.connect(self.on_play_pos)
        self.player.stopped.connect(self.on_play_stop)
        self.player.error.connect(lambda m: QMessageBox.warning(self, APP_NAME, m))
        self.recorder = Recorder(self)
        self.clip = None
        self.ctx = {}
        self.loop = False
        self.untitled = 0
        self._build_ui()
        self._build_menus()
        ui.fit_window(self, 1320, 860)
        self.refresh_all()
        for p in paths:
            if os.path.isfile(p):
                self.open_path(p)

    # ------------------------------------------------------------ properties
    @property
    def doc(self):
        w = self.tabs.currentWidget()
        return getattr(w, "doc", None) if w else None

    def view(self):
        return self.doc.editor.view if self.doc else None

    # ------------------------------------------------------------ layout
    def _build_ui(self):
        hd = ui.make_header(APP_NAME, "Record, edit and clean up sound", self.icon, self.about)
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(lambda _: self.refresh_all())
        self.bar = self._build_bar()
        self.inner = QMainWindow()
        self.inner.setWindowFlags(Qt.Widget)
        self.inner.setCentralWidget(self.tabs)
        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(hd)
        cl.addWidget(self.bar)
        cl.addWidget(self.inner, 1)
        self.setCentralWidget(central)

        self.meter = Meter()
        self.sel_panel = SelectionPanel(self)
        self.eq = EQPanel(self)
        self.markers_panel = MarkerPanel(self)
        self.history_panel = HistoryPanel(self)
        self.stack = ui.PanelStack("Digonto", "Audio")
        self.sections = {}
        for title, w in (("Levels", self.meter), ("Selection", self.sel_panel), ("Equalizer", self.eq), ("Markers", self.markers_panel),
                         ("History", self.history_panel)):
            self.sections[title] = self.stack.add(title, w)
        self.stack.restore()
        ui.shrinkable(self.stack)
        self.d_side = QDockWidget("Panels", self.inner)
        self.d_side.setObjectName("Panels")
        self.d_side.setTitleBarWidget(QWidget())
        self.d_side.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.d_side.setWidget(ui.ScrollColumn(self.stack))
        self.inner.addDockWidget(Qt.RightDockWidgetArea, self.d_side)
        self.inner.resizeDocks([self.d_side], [ui.dock_width_for(self.width(), self.stack.minimumSizeHint().width())], Qt.Horizontal)

        sb = self.statusBar()
        self.lbl_cursor = QLabel("")
        self.lbl_info = QLabel("")
        sb.addWidget(self.lbl_cursor, 1)
        sb.addPermanentWidget(self.lbl_info)

    def _build_bar(self):
        bar = QWidget()
        bar.setStyleSheet("QWidget#tb{background:#F2EDF7; border-bottom:1px solid #CEC8D3;}")
        bar.setObjectName("tb")
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(4)

        def btn(text, tip, fn, w=38, checkable=False, style=""):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFixedSize(w, 32)
            b.setCheckable(checkable)
            b.setStyleSheet("QPushButton{font-size:16px; border:1px solid #C0B8C7; border-radius:5px; background:white;}"
                            "QPushButton:hover{background:#E9E2EF;} QPushButton:checked{background:#DBCDE7; border-color:#C2395C;}" + style)
            b.clicked.connect(fn)
            h.addWidget(b)
            return b
        btn("⏮", "Go to start (Home)", self.go_start)
        self.b_play = btn("▶", "Play / pause (Space). Plays the selection, or from the cursor.", self.toggle_play, 46)
        btn("■", "Stop", self.stop)
        self.b_loop = btn("⟲", "Repeat", self.set_loop, checkable=True)
        btn("⏭", "Go to end (End)", self.go_end)
        btn("●", "Record from the microphone", self.record, 38, style="QPushButton{color:#C62828;}")
        h.addSpacing(10)
        self.t_now = QLabel("0:00.000")
        self.t_now.setStyleSheet("font-size:24px; font-weight:bold; color:#C2395C; font-family:monospace;")
        self.t_now.setMinimumWidth(150)
        h.addWidget(self.t_now)
        h.addSpacing(10)
        h.addWidget(QLabel("Volume"))
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(int(float(self.settings.value("volume", 80))))
        self.vol.setFixedWidth(100)
        self.vol.valueChanged.connect(lambda v: (self.player.set_volume(v / 100.0), self.settings.setValue("volume", v)))
        self.player.set_volume(self.vol.value() / 100.0)
        h.addWidget(self.vol)
        h.addStretch()
        self.mode = QComboBox()
        self.mode.addItem("Waveform", "wave")
        self.mode.addItem("Spectrogram (frequencies)", "spectro")
        self.mode.addItem("Both", "both")
        self.mode.currentIndexChanged.connect(self.set_mode)
        h.addWidget(self.mode)
        self.c_db = QCheckBox("dB scale")
        self.c_db.toggled.connect(self.set_db)
        self.c_snap = QCheckBox("Snap to zero")
        self.c_snap.setToolTip("Selections jump to the nearest zero crossing: cleaner cuts, no clicks")
        self.c_snap.toggled.connect(lambda on: [setattr(d.editor.view, "snap_zero", on) for d in self.docs()])
        h.addWidget(self.c_db)
        h.addWidget(self.c_snap)
        for text, tip, fn in (("＋", "Zoom in", lambda: self.zoom(0.5)), ("－", "Zoom out", lambda: self.zoom(2.0)), ("▭", "Show everything", self.zoom_all)):
            btn(text, tip, fn, 32)
        return bar

    def docs(self):
        return [self.tabs.widget(i).doc for i in range(self.tabs.count())]

    # ------------------------------------------------------------ menus
    def _act(self, menu, text, fn, key=None, tip=None):
        a = QAction(text, self)
        if key:
            a.setShortcut(QKeySequence(key))
        if tip:
            a.setStatusTip(tip)
        a.triggered.connect(lambda _=False: fn())
        menu.addAction(a)
        return a

    def _build_menus(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        self._act(fm, "New sound…", self.new_sound, "Ctrl+N")
        self._act(fm, "Open…", self.open_dialog, "Ctrl+O")
        self._act(fm, "Multitrack mixer…", self.open_multitrack, "Ctrl+M", "Put several sounds on tracks and mix them")
        fm.addSeparator()
        self._act(fm, "Save project (.daud)", self.save, "Ctrl+S", "Saves everything, nothing is lost")
        self._act(fm, "Save project as…", self.save_as, "Ctrl+Shift+S")
        self._act(fm, "Export / Save as WAV, MP3, FLAC, OGG, M4A…", self.export_dialog, "Ctrl+E")
        self._act(fm, "Export selection…", lambda: self.export_dialog(sel=True), "Ctrl+Shift+E")
        fm.addSeparator()
        self._act(fm, "Close project", lambda: self.close_tab(self.tabs.currentIndex()), "Ctrl+W")
        self._act(fm, "Exit", self.close, "Ctrl+Q")
        self._act(fm, "Quit (force)", lambda: os._exit(0), "Ctrl+Shift+Q")

        em = mb.addMenu("&Edit")
        self.a_undo = self._act(em, "Undo", self.undo, "Ctrl+Z")
        self.a_redo = self._act(em, "Redo", self.redo, "Ctrl+Shift+Z")
        em.addSeparator()
        self._act(em, "Cut", lambda: self.copy(True), "Ctrl+X")
        self._act(em, "Copy", lambda: self.copy(False), "Ctrl+C")
        self._act(em, "Paste (insert)", self.paste, "Ctrl+V")
        self._act(em, "Paste as new sound", self.paste_new)
        self._act(em, "Delete", self.delete_sel, "Delete")
        em.addSeparator()
        self._act(em, "Keep only the selection (crop)", self.crop, "Ctrl+T")
        self._act(em, "Make the selection silent", self.silence_sel, "Ctrl+L")
        self._act(em, "Insert silence…", self.insert_silence)
        self._act(em, "Cut by time (milliseconds)…", self.cut_by_time, "Ctrl+K")
        self._act(em, "Remove silence at start and end", self.trim_silence)
        self._act(em, "Shorten long pauses…", self.shorten_pauses)
        em.addSeparator()
        self._act(em, "Select all", self.select_all, "Ctrl+A")
        self._act(em, "Select nothing", lambda: self.view() and self.view().clear_selection(), "Escape")
        self._act(em, "Add marker at cursor", self.add_marker, "M")

        vm = mb.addMenu("&View")
        self._act(vm, "Zoom in", lambda: self.zoom(0.5), "Ctrl+=")
        self._act(vm, "Zoom out", lambda: self.zoom(2.0), "Ctrl+-")
        self._act(vm, "Show everything", self.zoom_all, "Ctrl+0")
        self._act(vm, "Zoom to selection", self.zoom_sel)
        vm.addSeparator()
        wm = vm.addMenu("Windows")
        for t, blk in self.sections.items():
            a = QAction(t, self, checkable=True, checked=True)
            a.toggled.connect(lambda on, b=blk: b.setVisible(on))
            wm.addAction(a)

        self.fx_menu = mb.addMenu("E&ffects")
        for group, items in EFFECTS.items():
            gm = self.fx_menu.addMenu(group)
            for name, fields, fn in items:
                self._act(gm, name, lambda n=name, f=fields, fnc=fn: self.run_effect(n, f, fnc))
        self.fx_menu.addSeparator()
        self._act(self.fx_menu, "Learn noise from the selection", self.learn_noise, None,
                  "Select a part with only noise (no voice), then use this before 'Remove background noise'")
        self._act(self.fx_menu, "Equalizer (right panel)", lambda: self.sections["Equalizer"].set_collapsed(False))
        self.fx_menu.addSeparator()
        self._act(self.fx_menu, "Split into voice and music (two new sounds)…", self.split_voice_music)

        gm = mb.addMenu("&Generate")
        self._act(gm, "Silence…", lambda: self.generate(False))
        self._act(gm, "Tone or noise…", lambda: self.generate(True))
        am = mb.addMenu("&Analyze")
        self._act(am, "Statistics (loudness, clipping)…", self.stats)
        self._act(am, "Select clipped part", self.find_clip)
        rm = mb.addMenu("&Record")
        self._act(rm, "Record from the microphone…", self.record, "Ctrl+R")
        hm = mb.addMenu("&Help")
        self._act(hm, "Quick tips", self.tips)
        self._act(hm, "About", self.about)
        # keyboard: space = play
        a = QAction(self)
        a.setShortcut(QKeySequence("Space"))
        a.triggered.connect(self.toggle_play)
        self.addAction(a)

    # ------------------------------------------------------------ tabs / refresh
    def add_doc(self, doc):
        v = doc.editor.view
        v.set_audio(doc.x, doc.sr)
        v.markers = doc.markers
        v.mode = self.mode.currentData() if hasattr(self, "mode") else "wave"
        v.show_db = self.c_db.isChecked()
        v.snap_zero = self.c_snap.isChecked()
        v.selectionChanged.connect(self.on_sel)
        v.cursorChanged.connect(lambda f: self.on_sel())
        v.scrubbed.connect(lambda f: None)
        i = self.tabs.addTab(doc.editor, doc.name)
        self.tabs.setCurrentIndex(i)
        self.refresh_all()
        QTimer.singleShot(0, v.zoom_all)

    def refresh_all(self):
        d = self.doc
        has = d is not None
        self.menuBar().setEnabled(True)
        if has:
            for i in range(self.tabs.count()):
                dd = self.tabs.widget(i).doc
                self.tabs.setTabText(i, ("● " if dd.dirty else "") + dd.name)
            self.setWindowTitle(f"{d.name} - {APP_NAME}")
            self.lbl_info.setText(f"{d.sr} Hz  ·  {'Stereo' if d.x.shape[1] == 2 else 'Mono'}  ·  {fmt_time(len(d.x) / d.sr)}  ·  "
                                  f"peak {dsp.db(dsp.peak(d.x)):.1f} dB")
        else:
            self.setWindowTitle(APP_NAME)
            self.lbl_info.setText("No sound open: File > Open, drop a file here, or press the red record button")
        self.history_panel.refresh(d)
        self.markers_panel.refresh(d)
        self.sel_panel.refresh(d)
        self.on_sel()

    def refresh_audio(self):
        d = self.doc
        if not d:
            return
        v = d.editor.view
        v.markers = d.markers
        v.set_audio(d.x, d.sr, keep_view=True)
        self.refresh_all()

    def refresh_markers(self):
        d = self.doc
        if d:
            d.editor.view.markers = d.markers
            d.editor.view.update()
        self.markers_panel.refresh(d)

    def on_sel(self):
        d = self.doc
        if not d:
            self.lbl_cursor.setText("")
            self.t_now.setText("0:00.000")
            return
        v = d.editor.view
        self.sel_panel.refresh(d)
        if v.sel:
            a, b = v.sel
            self.lbl_cursor.setText(f"Selection {fmt_time(a / d.sr)} – {fmt_time(b / d.sr)}   (length {fmt_time((b - a) / d.sr)})")
            self.t_now.setText(fmt_time(a / d.sr))
        else:
            self.lbl_cursor.setText(f"Cursor {fmt_time(v.cursor / d.sr)}   (sample {v.cursor:,})")
            self.t_now.setText(fmt_time(v.cursor / d.sr))

    # ------------------------------------------------------------ files
    def add_sound(self, x, sr, base="Untitled"):
        self.untitled += 1
        self.add_doc(Doc(x, sr, f"{base} {self.untitled}"))

    def open_multitrack(self):
        from akit.multitrack import MultitrackWindow
        w = MultitrackWindow(self)
        w.setAttribute(Qt.WA_DeleteOnClose)
        self._mix_windows = [x for x in getattr(self, "_mix_windows", []) if x.isVisible()] + [w]
        w.show()

    def new_sound(self):
        dlg = NewDialog(None, "New sound", ("new",), tone=False)
        if dlg.exec():
            x, sr = dlg.make(44100, 2)
            self.untitled += 1
            self.add_doc(Doc(x, sr, f"Untitled {self.untitled}"))

    def open_dialog(self):
        last = str(self.settings.value("last_dir", os.path.expanduser("~")))
        flt = "Sound files (*.daud *.paud *.dmix " + " ".join("*." + e for e in aio.OPEN_EXTS.split()) + ");;Digonto Audio project (*.daud *.paud);;All files (*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Open sound", last, flt)
        for p in paths:
            self.open_path(p)

    def open_path(self, path):
        if path.lower().endswith(".dmix"):
            from akit.multitrack import MultitrackWindow, Mix
            try:
                w = MultitrackWindow(self, Mix.load(path), path)
            except Exception as e:  # noqa
                QMessageBox.warning(self, "Open", f"{os.path.basename(path)}\n\n{e}")
                return
            w.setAttribute(Qt.WA_DeleteOnClose)
            self._mix_windows = [x for x in getattr(self, "_mix_windows", []) if x.isVisible()] + [w]
            w.show()
            return
        for d in self.docs():
            if d.path == path:
                self.tabs.setCurrentWidget(d.editor)
                return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            x, sr, extra = aio.load(path)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Open", f"{os.path.basename(path)}\n\n{e}")
            return
        QApplication.restoreOverrideCursor()
        self.settings.setValue("last_dir", os.path.dirname(path))
        name = os.path.splitext(os.path.basename(path))[0]
        proj = path.lower().endswith((aio.PROJECT_EXT, aio.LEGACY_EXT))
        self.add_doc(Doc(x, sr, name, extra.get("markers", []), path if proj else None, src=None if proj else path))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.isLocalFile():
                self.open_path(u.toLocalFile())

    def save(self):
        d = self.doc
        if not d:
            return False
        if not d.path:
            return self.save_as()
        return self._write_project(d, d.path)

    def save_as(self):
        d = self.doc
        if not d:
            return False
        last = str(self.settings.value("last_dir", os.path.expanduser("~")))
        p, _ = QFileDialog.getSaveFileName(self, "Save project", os.path.join(last, d.name + aio.PROJECT_EXT), "Digonto Audio project (*.daud)")
        if not p:
            return False
        if not p.lower().endswith(aio.PROJECT_EXT):
            p += aio.PROJECT_EXT
        return self._write_project(d, p)

    def _write_project(self, d, p):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            aio.save_project(p, d.x, d.sr, d.markers)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Save", str(e))
            return False
        QApplication.restoreOverrideCursor()
        d.path, d.saved = p, d.hi
        d.name = os.path.splitext(os.path.basename(p))[0]
        self.settings.setValue("last_dir", os.path.dirname(p))
        self.refresh_all()
        self.statusBar().showMessage("Saved " + p, 4000)
        return True

    def export_dialog(self, sel=False):
        d = self.doc
        if not d:
            return
        v = d.editor.view
        dlg = ExportDialog(self, bool(v.sel), len(d.markers), d.sr, d.x.shape[1], d.name)
        if sel and v.sel:
            dlg.scope.setCurrentIndex(dlg.scope.findData("sel"))
        if not dlg.exec():
            return
        s = dlg.settings()
        ext = aio.FORMATS[s["fmt"]][0]
        last = str(self.settings.value("last_dir", os.path.expanduser("~")))
        try:
            if s["scope"] == "markers":
                folder = QFileDialog.getExistingDirectory(self, "Folder for the pieces", last)
                if not folder:
                    return
                cuts = [0.0] + sorted(m[0] for m in d.markers) + [len(d.x) / d.sr]
                n = 0
                QApplication.setOverrideCursor(Qt.WaitCursor)
                for i in range(len(cuts) - 1):
                    a, b = int(cuts[i] * d.sr), int(cuts[i + 1] * d.sr)
                    if b - a < 2:
                        continue
                    n += 1
                    aio.export(os.path.join(folder, f"{d.name}-{n:03d}{ext}"), d.x[a:b], d.sr, s["fmt"], s["kbps"], s["tags"], s["rate"], s["mono"])
                QApplication.restoreOverrideCursor()
                self.statusBar().showMessage(f"{n} files saved in {folder}", 6000)
                return
            p, _ = QFileDialog.getSaveFileName(self, "Export", os.path.join(last, d.name + ext), f"{s['fmt']} (*{ext})")
            if not p:
                return
            if not p.lower().endswith(ext):
                p += ext
            x = d.x[v.sel[0]:v.sel[1]] if s["scope"] == "sel" and v.sel else d.x
            QApplication.setOverrideCursor(Qt.WaitCursor)
            aio.export(p, x, d.sr, s["fmt"], s["kbps"], s["tags"], s["rate"], s["mono"])
            QApplication.restoreOverrideCursor()
            self.settings.setValue("last_dir", os.path.dirname(p))
            self.statusBar().showMessage(f"Saved {p}  ({os.path.getsize(p) / 1e6:.2f} MB)", 8000)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Export", str(e))

    def close_tab(self, i):
        if i < 0:
            return True
        w = self.tabs.widget(i)
        d = w.doc
        if d.dirty:
            r = QMessageBox.question(self, "Close", f"“{d.name}” has changes. Save the project first?",
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if r == QMessageBox.Cancel:
                return False
            if r == QMessageBox.Save:
                self.tabs.setCurrentIndex(i)
                if not self.save():
                    return False
        self.player.stop()
        self.tabs.removeTab(i)
        w.deleteLater()
        self.refresh_all()
        return True

    def closeEvent(self, ev):
        for i in range(self.tabs.count() - 1, -1, -1):
            self.tabs.setCurrentIndex(i)
            if not self.close_tab(i):
                ev.ignore()
                return
        self.player.stop()
        if self.recorder.recording:
            self.recorder.stop()
        self.settings.setValue("state", self.saveState())
        ev.accept()
        import threading
        t = threading.Timer(0.25, lambda: os._exit(0))
        t.daemon = True
        t.start()

    # ------------------------------------------------------------ editing core
    def _range(self, d, whole_if_none=True):
        v = d.editor.view
        if v.sel:
            return v.sel
        return (0, len(d.x)) if whole_if_none else (v.cursor, v.cursor)

    def replace(self, label, a, b, y, select=True):
        """x[a:b] becomes y"""
        d = self.doc
        n_old = b - a
        if y.shape[1] != d.x.shape[1]:
            y = dsp.to_mono(y) if d.x.shape[1] == 1 else dsp.to_stereo(y)
        new = np.concatenate([d.x[:a], y.astype(np.float32), d.x[b:]], axis=0)
        delta = (len(y) - n_old) / d.sr
        sec_a, sec_b = a / d.sr, b / d.sr
        mk = []
        for s, n in d.markers:
            if s >= sec_b:
                mk.append((s + delta, n))
            elif s > sec_a and len(y) < n_old:
                continue                      # a marker inside a removed piece goes away
            else:
                mk.append((s, n))
        d.commit(label, new, mk)
        v = d.editor.view
        sel = (a, a + len(y)) if select and len(y) > 0 else None
        v.markers = d.markers
        v.set_audio(d.x, d.sr, keep_view=True)
        if sel:
            v.set_selection(*sel)
        else:
            v.clear_selection()
            v.set_cursor(a)
        self.refresh_all()

    def apply_effect(self, label, fn, whole=False):
        d = self.doc
        if not d:
            return
        self.player.stop()
        a, b = self._range(d)
        piece = d.x[a:b]
        if len(piece) < 8:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            y = fn(piece, d.sr)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, label, str(e))
            return
        QApplication.restoreOverrideCursor()
        sel = d.editor.view.sel
        self.replace(label, a, b, np.asarray(y, dtype=np.float32), select=bool(sel))

    def run_effect(self, name, fields, fn):
        d = self.doc
        if not d:
            return
        label = name.rstrip("…")
        if not fields:
            return self.apply_effect(label, lambda x, sr: fn(x, sr, {}, self.ctx))
        a, b = self._range(d)
        dlg = EffectDialog(self, name, fields, fn, d.x[a:b], d.sr, self.ctx, self.player)
        if "Remove background noise" in name and "noise" not in self.ctx:
            dlg.w["use_learned"][0].setEnabled(False)
        if dlg.exec():
            p = dlg.params()
            self.apply_effect(label, lambda x, sr: fn(x, sr, p, self.ctx))

    def preview_fn(self, f):
        d = self.doc
        if not d:
            return
        a, b = self._range(d)
        n = min(b - a, int(d.sr * 8))
        try:
            y = np.clip(f(d.x[a:a + n], d.sr), -1, 1)
        except Exception as e:
            QMessageBox.warning(self, "Preview", str(e))
            return
        self.player.play(y, d.sr, 0, len(y))

    def learn_noise(self):
        d = self.doc
        v = self.view()
        if not d or not v or not v.sel or v.sel[1] - v.sel[0] < 4096:
            QMessageBox.information(self, "Learn noise", "First select a piece of the sound that has only the background noise "
                                    "(no voice or music), at least half a second. Then choose this again.")
            return
        a, b = v.sel
        self.ctx["noise"] = dsp.noise_profile(d.x[a:b])
        self.statusBar().showMessage("Noise learned. Now choose Effects > Clean up > Remove background noise and tick “Use the noise I picked”.", 9000)

    def split_voice_music(self):
        d = self.doc
        if not d:
            return
        if d.x.shape[1] < 2:
            QMessageBox.information(self, "Voice and music", "This needs a stereo sound (two channels).")
            return
        a, b = self._range(d)
        fields = [("s", "Strength", "i", 70, 0, 100, " %")]
        dlg = EffectDialog(self, "Split into voice and music… (Preview plays the voice)", fields,
                           lambda x, sr, p, c: dsp.keep_center(x, sr, p["s"] / 100.0), d.x[a:b], d.sr, self.ctx, self.player)
        if not dlg.exec():
            return
        s = dlg.params()["s"] / 100.0
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            voice, music = dsp.separate_center(d.x[a:b], d.sr, s)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Voice and music", str(e))
            return
        QApplication.restoreOverrideCursor()
        self.add_doc(Doc(voice, d.sr, d.name + " - voice"))
        self.add_doc(Doc(music, d.sr, d.name + " - music"))
        self.statusBar().showMessage("Two new sounds were made (tabs). Voice and music are told apart by their place between the speakers.", 9000)

    # ------------------------------------------------------------ edit actions
    def undo(self):
        d = self.doc
        if d and d.hi > 0:
            d.jump(d.hi - 1)
            self.refresh_audio()

    def redo(self):
        d = self.doc
        if d and d.hi < len(d.hist) - 1:
            d.jump(d.hi + 1)
            self.refresh_audio()

    def copy(self, cut=False):
        d = self.doc
        v = self.view()
        if not d or not v or not v.sel:
            return
        a, b = v.sel
        self.clip = (d.x[a:b].copy(), d.sr)
        if cut:
            self.replace("Cut", a, b, np.zeros((0, d.x.shape[1]), np.float32), select=False)

    def paste(self):
        d = self.doc
        v = self.view()
        if not d or self.clip is None:
            return
        y, sr = self.clip
        if sr != d.sr:
            y = dsp.change_samplerate(y, sr, d.sr)
        a, b = (v.sel if v.sel else (v.cursor, v.cursor))
        self.replace("Paste", a, b, y, select=True)

    def paste_new(self):
        if self.clip is not None:
            self.untitled += 1
            self.add_doc(Doc(self.clip[0].copy(), self.clip[1], f"Untitled {self.untitled}"))

    def delete_sel(self):
        d = self.doc
        v = self.view()
        if d and v and v.sel:
            a, b = v.sel
            self.replace("Delete", a, b, np.zeros((0, d.x.shape[1]), np.float32), select=False)

    def crop(self):
        d = self.doc
        v = self.view()
        if d and v and v.sel:
            a, b = v.sel
            self.replace("Keep selection", 0, len(d.x), d.x[a:b].copy(), select=False)
            v.zoom_all()

    def silence_sel(self):
        d = self.doc
        v = self.view()
        if d and v and v.sel:
            a, b = v.sel
            self.replace("Silence", a, b, np.zeros((b - a, d.x.shape[1]), np.float32))

    def insert_silence(self):
        d = self.doc
        if not d:
            return
        dlg = NewDialog(self, "Insert silence")
        if dlg.exec():
            y = dsp.silence(d.sr, d.x.shape[1], dlg.secs.value())
            c = self.view().cursor
            self.replace("Insert silence", c, c, y)

    def generate(self, tone):
        d = self.doc
        dlg = NewDialog(self if d else None, "Tone or noise" if tone else "Silence", ("new",) if not d else ("x",), tone=tone)
        if not dlg.exec():
            return
        if not d:
            x, sr = dlg.make(44100, 2)
            self.untitled += 1
            self.add_doc(Doc(x, sr, f"Untitled {self.untitled}"))
            return
        y, _ = dlg.make(d.sr, d.x.shape[1])
        v = self.view()
        a, b = (v.sel if v.sel else (v.cursor, v.cursor))
        self.replace("Generate", a, b, y)

    def cut_by_time(self):
        d = self.doc
        v = self.view()
        if not d:
            return
        a, b = (v.sel if v.sel else (v.cursor, min(len(d.x), v.cursor + d.sr)))
        dlg = CutDialog(self, len(d.x) / d.sr, a / d.sr, b / d.sr)
        if not dlg.exec():
            return
        sa, sb = dlg.range()
        a, b = int(round(sa * d.sr)), int(round(sb * d.sr))
        if b <= a:
            return
        k = dlg.action
        if k == "select":
            v.set_selection(a, b)
            v.ensure_visible(a)
        elif k == "copy":
            self.clip = (d.x[a:b].copy(), d.sr)
            self.statusBar().showMessage("Copied. Use Edit > Paste.", 4000)
        elif k == "delete":
            self.replace("Cut out " + fmt_time(sa) + "–" + fmt_time(sb), a, b, np.zeros((0, d.x.shape[1]), np.float32), select=False)
        elif k == "keep":
            self.replace("Keep " + fmt_time(sa) + "–" + fmt_time(sb), 0, len(d.x), d.x[a:b].copy(), select=False)
            v.zoom_all()
        elif k == "silence":
            self.replace("Silence " + fmt_time(sa) + "–" + fmt_time(sb), a, b, np.zeros((b - a, d.x.shape[1]), np.float32))
        elif k == "new":
            self.untitled += 1
            self.add_doc(Doc(d.x[a:b].copy(), d.sr, f"{d.name} {fmt_time(sa)}-{fmt_time(sb)}"))

    def _silent_blocks(self, x, sr, thr_db, blk_ms=10):
        blk = max(1, int(sr * blk_ms / 1000))
        n = len(x) // blk
        if n == 0:
            return blk, np.zeros(0, bool)
        m = np.abs(x[:n * blk]).max(axis=1).reshape(n, blk).max(axis=1)
        return blk, m < dsp.lin(thr_db)

    def trim_silence(self):
        d = self.doc
        if not d:
            return
        blk, sil = self._silent_blocks(d.x, d.sr, -50)
        loud = np.where(~sil)[0]
        if not len(loud):
            self.statusBar().showMessage("The whole sound is quiet; nothing to trim.", 4000)
            return
        a = max(0, (loud[0] - 3) * blk)
        b = min(len(d.x), (loud[-1] + 4) * blk)
        if a == 0 and b == len(d.x):
            self.statusBar().showMessage("There is no silence at the start or end.", 4000)
            return
        self.replace("Remove silence at ends", 0, len(d.x), d.x[a:b].copy(), select=False)

    def shorten_pauses(self):
        d = self.doc
        if not d:
            return
        mx, ok = QInputDialog.getDouble(self, "Shorten long pauses", "Make every pause at most this long (seconds):", 0.5, 0.05, 30, 2)
        if not ok:
            return
        blk, sil = self._silent_blocks(d.x, d.sr, -45)
        keep_blocks = max(1, int(mx * d.sr / blk))
        out, i = [], 0
        n = len(sil)
        while i < n:
            j = i
            while j < n and sil[j] == sil[i]:
                j += 1
            ln = j - i
            use = min(ln, keep_blocks) if sil[i] else ln
            out.append(d.x[i * blk:(i + use) * blk])
            i = j
        out.append(d.x[n * blk:])
        self.replace("Shorten pauses", 0, len(d.x), np.concatenate(out), select=False)

    def select_all(self):
        v = self.view()
        if v:
            v.set_selection(0, len(v.x))

    def add_marker(self):
        d = self.doc
        v = self.view()
        if not d:
            return
        c = (v.sel[0] if v.sel else v.cursor) / d.sr
        d.markers.append((c, f"Marker {len(d.markers) + 1}"))
        d.hist[d.hi] = (d.hist[d.hi][0], d.x, list(d.markers))
        self.refresh_markers()

    # ------------------------------------------------------------ view
    def set_mode(self):
        for d in self.docs():
            d.editor.view.mode = self.mode.currentData()
            d.editor.view.update()

    def set_db(self, on):
        for d in self.docs():
            d.editor.view.show_db = on
            d.editor.view.update()

    def zoom(self, f):
        v = self.view()
        if v:
            v.zoom_by(f)

    def zoom_all(self):
        v = self.view()
        if v:
            v.zoom_all()

    def zoom_sel(self):
        v = self.view()
        if v:
            v.zoom_sel()

    # ------------------------------------------------------------ playing
    def toggle_play(self):
        if self.player.playing:
            self.stop()
        else:
            self.play()

    def play(self):
        d = self.doc
        v = self.view()
        if not d or not v:
            return
        if v.sel:
            a, b = v.sel
        else:
            a, b = v.cursor, len(d.x)
            if b - a < 100:
                a = 0
        self._play_from = v.cursor if not v.sel else v.sel[0]
        if self.player.play(d.x, d.sr, a, b, self.loop):
            self.b_play.setText("⏸")

    def stop(self):
        self.player.stop()
        self.on_play_stop()

    def on_play_stop(self):
        self.b_play.setText("▶")
        v = self.view()
        if v:
            v.play_pos = None
            v.update()

    def set_loop(self):
        self.loop = self.b_loop.isChecked()
        self.player.loop = self.loop

    def go_start(self):
        v = self.view()
        if v:
            v.set_cursor(0)
            v.clear_selection()
            v.scroll_to(0)

    def go_end(self):
        v = self.view()
        if v:
            v.set_cursor(len(v.x))
            v.ensure_visible(len(v.x))

    def on_play_pos(self, f):
        v = self.view()
        d = self.doc
        if not v or not d:
            return
        v.play_pos = f
        v.ensure_visible(f) if not v.sel else None
        v.update()
        self.t_now.setText(fmt_time(f / d.sr))
        w = int(d.sr * 0.03)
        seg = d.x[max(0, f - w):f + w]
        if len(seg):
            self.meter.push([float(np.abs(seg[:, c]).max()) for c in range(seg.shape[1])])

    # ------------------------------------------------------------ record / analyse
    def record(self):
        self.player.stop()
        dlg = RecordDialog(self, self.recorder, devices_in())
        if dlg.exec() and dlg.audio is not None and len(dlg.audio):
            audio, sr = dlg.audio, self.recorder.sr
            d = self.doc
            if d and QMessageBox.question(self, "Recording", "Add the recording into the open sound at the cursor?\n(No = make a new sound)",
                                          QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                if sr != d.sr:
                    audio = dsp.change_samplerate(audio, sr, d.sr)
                v = self.view()
                a, b = (v.sel if v.sel else (v.cursor, v.cursor))
                self.replace("Record", a, b, audio)
            else:
                self.untitled += 1
                self.add_doc(Doc(audio, sr, f"Recording {self.untitled}"))

    def stats(self):
        d = self.doc
        if d:
            a, b = self._range(d)
            StatsDialog(self, dsp.stats(d.x[a:b], d.sr), d.sr, d.name + (" (selection)" if self.view().sel else "")).exec()

    def find_clip(self):
        d = self.doc
        v = self.view()
        if not d:
            return
        idx = np.where(np.abs(d.x).max(axis=1) >= 0.999)[0]
        if not len(idx):
            self.statusBar().showMessage("No clipped (distorted) samples found.", 5000)
            return
        v.set_selection(int(idx[0]), int(idx[0]) + int(d.sr * 0.05))
        v.ensure_visible(int(idx[0]))
        self.statusBar().showMessage(f"{len(idx):,} clipped samples. The first one is selected.", 6000)

    # ------------------------------------------------------------ help
    def tips(self):
        QMessageBox.information(self, "Quick tips",
                                "• Drag on the wave to select. Double-click selects everything. Shift+click extends.\n"
                                "• Mouse wheel zooms, Shift+wheel scrolls, drag with the right mouse button to move.\n"
                                "• Left / Right arrow moves the cursor 10 ms, with Alt 1 ms, with Ctrl 1 s. Shift extends the selection.\n"
                                "• Edit > Cut by time: type exact times like 0:12.345 and cut that piece.\n"
                                "• Effects work on the selection, or on everything when nothing is selected. Every effect has Preview.\n"
                                "• Quiet sound: Effects > Volume > Make quiet audio louder.\n"
                                "• Background noise: select only noise, Effects > Learn noise, then Remove background noise.\n"
                                "• Voice and music: Effects > Split into voice and music (stereo files work best).\n"
                                "• Ctrl+S saves a .daud project (nothing lost). Ctrl+E exports WAV, MP3, FLAC, OGG, M4A and more.")

    def about(self):
        QMessageBox.about(self, "About " + APP_NAME,
                          f"<h3>{APP_NAME} {VERSION}</h3><p>Record, edit and clean up sound. Equalizer, echo and reverb, noise remover, "
                          "volume boost, voice/music split, millisecond cuts.</p>"
                          "<p>Opens almost any audio file. Saves .daud projects and exports WAV, AIFF, MP3, OGG, Opus, FLAC, M4A, WMA, AC3.</p>")


def main():
    if "--version" in sys.argv:
        print(VERSION)
        return
    ui.mac_app_name(APP_NAME)
    app = QApplication(sys.argv)
    if ui.forward_to_running(SLUG, sys.argv[1:]):
        os._exit(0)
    ui.setup_app(app, APP_NAME, SLUG, ui.app_icon(HERE, SLUG))
    w = Main([a for a in sys.argv[1:] if os.path.isfile(a)])
    w.show()
    ui.serve_single(SLUG, w, w.open_path)
    rc = app.exec()
    os._exit(rc)


if __name__ == "__main__":
    main()
