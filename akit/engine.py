"""Digonto Audio - playing and recording (Qt Multimedia)."""
import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject, QTimer, Signal
from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSink, QAudioSource, QMediaDevices


def devices_out():
    return [(d.description(), d) for d in QMediaDevices.audioOutputs()]


def devices_in():
    return [(d.description(), d) for d in QMediaDevices.audioInputs()]


class Player(QObject):
    """plays a piece of the sound; position() is the frame being heard"""
    position = Signal(int)
    stopped = Signal()
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sink = None
        self.buf = None
        self.start_frame = 0
        self.length = 0
        self.sr = 44100
        self.loop = False
        self._args = None
        self.volume = 1.0
        self.device = None
        self.playing = False
        self.timer = QTimer(self)
        self.timer.setInterval(25)
        self.timer.timeout.connect(self._tick)

    def play(self, x, sr, start, end, loop=False):
        self.stop(silent=True)
        end = min(end, len(x))
        if end - start < 16:
            return False
        seg = x[start:end]
        ch = seg.shape[1]
        fmt = QAudioFormat()
        fmt.setSampleRate(int(sr))
        fmt.setChannelCount(int(ch))
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        dev = self.device or QMediaDevices.defaultAudioOutput()
        if dev is None or dev.isNull():
            self.error.emit("No sound output found on this computer.")
            return False
        data = (np.clip(seg, -1, 1) * 32767).astype("<i2").tobytes()
        self.buf = QBuffer()
        self.buf.setData(QByteArray(data))
        self.buf.open(QIODevice.OpenModeFlag.ReadOnly)
        self.sink = QAudioSink(dev, fmt)
        self.sink.setVolume(self.volume)
        self.sink.stateChanged.connect(self._state)
        self.start_frame, self.length, self.sr, self.loop = start, end - start, int(sr), loop
        self._args = (x, sr, start, end)
        self.sink.start(self.buf)
        self.playing = True
        self.timer.start()
        return True

    def set_volume(self, v):
        self.volume = v
        if self.sink:
            self.sink.setVolume(v)

    def _frame(self):
        if not self.sink:
            return self.start_frame
        f = int(self.sink.processedUSecs() * self.sr / 1_000_000)
        return self.start_frame + min(f, self.length)

    def _tick(self):
        if self.playing:
            self.position.emit(self._frame())

    def _state(self, st):
        if st == QAudio.State.IdleState and self.playing:
            if self.loop and self._args:
                a = self._args
                self.play(a[0], a[1], a[2], a[3], True)
            else:
                end = self.start_frame + self.length
                self.stop(silent=True)
                self.position.emit(end)
                self.stopped.emit()

    def stop(self, silent=False):
        was = self.playing
        self.playing = False
        self.timer.stop()
        if self.sink:
            try:
                self.sink.stateChanged.disconnect(self._state)
            except Exception:
                pass
            self.sink.stop()
            self.sink.deleteLater()
            self.sink = None
        if self.buf:
            self.buf.close()
            self.buf = None
        if was and not silent:
            self.stopped.emit()

    def current(self):
        return self._frame() if self.playing else None


class Recorder(QObject):
    """records from the microphone; chunks arrive as float32 arrays"""
    level = Signal(float)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.src = None
        self.io = None
        self.chunks = []
        self.sr = 44100
        self.ch = 1
        self.recording = False
        self.device = None

    def start(self, sr=44100, ch=1):
        dev = self.device or QMediaDevices.defaultAudioInput()
        if dev is None or dev.isNull():
            self.failed.emit("No microphone found. Plug one in, or check the sound settings of the computer.")
            return False
        fmt = QAudioFormat()
        fmt.setSampleRate(sr)
        fmt.setChannelCount(ch)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if not dev.isFormatSupported(fmt):
            fmt = dev.preferredFormat()
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        self.sr, self.ch = fmt.sampleRate(), fmt.channelCount()
        self.src = QAudioSource(dev, fmt)
        self.chunks = []
        self.io = self.src.start()
        if self.io is None:
            self.failed.emit("Could not open the microphone. On macOS allow it in System Settings > Privacy & Security > Microphone.")
            return False
        self.io.readyRead.connect(self._read)
        self.recording = True
        return True

    def _read(self):
        if not self.io:
            return
        d = bytes(self.io.readAll())
        if not d:
            return
        n = len(d) // (2 * self.ch) * (2 * self.ch)
        a = np.frombuffer(d[:n], dtype="<i2").astype(np.float32).reshape(-1, self.ch) / 32768.0
        self.chunks.append(a)
        self.level.emit(float(np.max(np.abs(a))) if a.size else 0.0)

    def frames(self):
        return sum(len(c) for c in self.chunks)

    def stop(self):
        self.recording = False
        if self.src:
            self._read()
            self.src.stop()
            self.src.deleteLater()
            self.src = None
        self.io = None
        if not self.chunks:
            return np.zeros((0, self.ch), np.float32)
        return np.concatenate(self.chunks)
