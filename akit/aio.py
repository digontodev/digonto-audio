"""Digonto Audio - reading and writing sound files, and the own project format (.daud; older .paud files still open)."""
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import wave
import zipfile

import numpy as np

PROJECT_EXT = ".daud"
LEGACY_EXT = ".paud"          # older projects still open
NOWIN = {"creationflags": 0x08000000} if os.name == "nt" else {}

# name -> (extension, ffmpeg codec args, description, has_quality)
FORMATS = {
    "WAV (PCM 16-bit)": (".wav", None, "Uncompressed. Plays everywhere."),
    "WAV (PCM 24-bit)": (".wav", None, "Uncompressed, studio quality."),
    "WAV (32-bit float)": (".wav", None, "Uncompressed, no rounding at all."),
    "AIFF (16-bit)": (".aiff", None, "Uncompressed, Apple."),
    "AIFF (24-bit)": (".aiff", None, "Uncompressed, Apple, studio quality."),
    "MP3": (".mp3", ["-c:a", "libmp3lame"], "Small files, plays everywhere."),
    "OGG Vorbis": (".ogg", ["-c:a", "libvorbis"], "Small files, free format."),
    "Opus": (".opus", ["-c:a", "libopus"], "Best quality for voice and small size."),
    "FLAC (lossless)": (".flac", ["-c:a", "flac"], "Half the size of WAV, no quality loss."),
    "M4A (AAC)": (".m4a", ["-c:a", "aac"], "Small files, iPhone / Apple / YouTube."),
    "M4A (Apple Lossless)": (".m4a", ["-c:a", "alac"], "Lossless, Apple."),
    "WMA": (".wma", ["-c:a", "wmav2"], "Windows Media."),
    "AC3 (Dolby Digital)": (".ac3", ["-c:a", "ac3"], "Video / DVD sound."),
}
LOSSY_KBPS = ("MP3", "Opus", "M4A (AAC)", "WMA", "AC3 (Dolby Digital)", "OGG Vorbis")
OPEN_EXTS = "wav mp3 flac ogg oga opus m4a aac aif aiff wma amr ac3 mp4 m4b mov mkv webm avi caf au 3gp wv ape mka"


def find_ffmpeg():
    """the private ffmpeg that comes with the program (imageio-ffmpeg), else one installed on the computer"""
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.exists(p):
            return p
    except Exception:
        pass
    for name in ("ffmpeg", "ffmpeg.exe"):
        p = shutil.which(name)
        if p:
            return p
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------- numbers <-> bytes
def to_int(x, bits):
    y = np.clip(x, -1.0, 1.0 - 1e-9)
    if bits == 16:
        return np.round(y * 32767).astype("<i2")
    if bits == 24:
        v = np.round(y * 8388607).astype("<i4")
        return v
    return y


def _pcm_to_float(raw, width, ch):
    if width == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    elif width == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v & 0x800000, v - 0x1000000, v)
        a = v.astype(np.float32) / 8388608.0
    elif width == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported sample size")
    return a.reshape(-1, ch)


# ---------------------------------------------------------------- reading
def _load_wav(path):
    with wave.open(path, "rb") as w:
        ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    return _pcm_to_float(raw, width, ch), sr


def _probe(ff, path):
    r = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True, **NOWIN)
    txt = r.stderr
    m = re.search(r"Audio:.*?(\d+) Hz,\s*([^,]+)", txt)
    if not m:
        raise ValueError("No sound found in this file.")
    sr = int(m.group(1))
    lay = m.group(2).strip().lower()
    ch = 1 if lay.startswith("mono") else 2 if lay.startswith("stereo") else None
    if ch is None:
        mm = re.match(r"(\d+)\s*channels", lay)
        ch = int(mm.group(1)) if mm else 2
    return sr, min(ch, 2)


def _load_ffmpeg(ff, path):
    sr, ch = _probe(ff, path)
    cmd = [ff, "-v", "error", "-i", path, "-vn", "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(sr), "-ac", str(ch), "-"]
    r = subprocess.run(cmd, capture_output=True, **NOWIN)
    if r.returncode != 0 and not r.stdout:
        raise ValueError("Could not read this file:\n" + r.stderr.decode(errors="replace")[-300:])
    a = np.frombuffer(r.stdout, dtype="<f4")
    a = a[:len(a) // ch * ch].reshape(-1, ch).astype(np.float32)
    return a, sr


def _load_qt(path):
    from PySide6.QtCore import QEventLoop, QTimer, QUrl
    from PySide6.QtMultimedia import QAudioDecoder, QAudioFormat
    dec = QAudioDecoder()
    dec.setSource(QUrl.fromLocalFile(path))
    chunks = []
    info = {}
    loop = QEventLoop()
    err = []

    def got():
        b = dec.read()
        if not b.isValid():
            return
        f = b.format()
        info["sr"], info["ch"] = f.sampleRate(), f.channelCount()
        sf = f.sampleFormat()
        data = bytes(b.constData())
        ch = f.channelCount()
        if sf == QAudioFormat.SampleFormat.Float:
            a = np.frombuffer(data, dtype="<f4").reshape(-1, ch)
        elif sf == QAudioFormat.SampleFormat.Int16:
            a = np.frombuffer(data, dtype="<i2").astype(np.float32).reshape(-1, ch) / 32768.0
        elif sf == QAudioFormat.SampleFormat.Int32:
            a = np.frombuffer(data, dtype="<i4").astype(np.float32).reshape(-1, ch) / 2147483648.0
        elif sf == QAudioFormat.SampleFormat.UInt8:
            a = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128).reshape(-1, ch) / 128.0
        else:
            return
        chunks.append(a.copy())
    dec.bufferReady.connect(got)
    dec.finished.connect(loop.quit)
    dec.errorOccurred.connect(lambda e, s: (err.append(s), loop.quit()))
    QTimer.singleShot(600000, loop.quit)
    dec.start()
    loop.exec()
    if not chunks:
        raise ValueError("Could not read this file." + (" " + err[0] if err else ""))
    return np.concatenate(chunks), info["sr"]


def load(path):
    """returns (audio float32 (n, ch), samplerate, extra dict). Mono and stereo are kept; more channels are mixed to stereo."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (PROJECT_EXT, LEGACY_EXT):
        return load_project(path)
    x = sr = None
    if ext == ".wav":
        try:
            x, sr = _load_wav(path)
        except Exception:
            x = None
    if x is None:
        ff = find_ffmpeg()
        if ff:
            x, sr = _load_ffmpeg(ff, path)
        else:
            x, sr = _load_qt(path)
    if x.shape[1] > 2:
        x = x[:, :2]
    return np.ascontiguousarray(x, dtype=np.float32), int(sr), {}


# ---------------------------------------------------------------- writing
def _write_wav(path, x, sr, bits):
    ch = x.shape[1]
    if bits == 32:
        fmt, width, data = 3, 4, np.clip(x, -4, 4).astype("<f4").tobytes()
    elif bits == 24:
        v = to_int(x, 24).astype("<i4")
        b = v.view(np.uint8).reshape(-1, 4)[:, :3]
        fmt, width, data = 1, 3, b.tobytes()
    else:
        fmt, width, data = 1, 2, to_int(x, 16).tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, fmt, ch, sr, sr * ch * width, ch * width, width * 8))
        f.write(b"data" + struct.pack("<I", len(data)))
        f.write(data)


def _ext80(rate):
    m, e = np.frexp(float(rate))
    mant = int(m * (1 << 63) * 2) if False else int(m * 2 ** 64)
    return struct.pack(">HQ", int(e) - 1 + 16383, mant & 0xFFFFFFFFFFFFFFFF)


def _write_aiff(path, x, sr, bits):
    ch = x.shape[1]
    n = len(x)
    if bits == 24:
        v = to_int(x, 24).astype(">i4")
        data = v.view(np.uint8).reshape(-1, 4)[:, 1:].tobytes()
        width = 3
    else:
        data = to_int(x, 16).astype(">i2").tobytes()
        width = 2
    comm = struct.pack(">hIh", ch, n, width * 8) + _ext80(sr)
    ssnd = struct.pack(">II", 0, 0) + data
    body = b"AIFF" + b"COMM" + struct.pack(">I", len(comm)) + comm + b"SSND" + struct.pack(">I", len(ssnd)) + ssnd
    with open(path, "wb") as f:
        f.write(b"FORM" + struct.pack(">I", len(body)) + body)


def export(path, x, sr, fmt="WAV (PCM 16-bit)", kbps=192, tags=None, samplerate=None, mono=False, progress=None):
    """write x to a file. fmt is a key of FORMATS."""
    if fmt not in FORMATS:
        raise ValueError("Unknown format")
    if mono and x.shape[1] > 1:
        x = x.mean(axis=1, keepdims=True).astype(np.float32)
    if samplerate and samplerate != sr:
        from . import dsp
        x = dsp.change_samplerate(x, sr, samplerate)
        sr = samplerate
    ext = FORMATS[fmt][0]
    if fmt.startswith("WAV"):
        return _write_wav(path, x, sr, 32 if "32" in fmt else 24 if "24" in fmt else 16)
    if fmt.startswith("AIFF"):
        return _write_aiff(path, x, sr, 24 if "24" in fmt else 16)
    ff = find_ffmpeg()
    if not ff:
        raise ValueError("This format needs the program's sound converter (ffmpeg). Reinstall Digonto Audio from the installer, or choose WAV / AIFF.")
    args = list(FORMATS[fmt][1])
    if fmt in LOSSY_KBPS:
        if fmt == "OGG Vorbis":
            args += ["-q:a", str(max(0, min(10, round((kbps - 64) / 32))))]
        else:
            args += ["-b:a", f"{int(kbps)}k"]
    if fmt == "Opus" and sr not in (8000, 12000, 16000, 24000, 48000):
        args += ["-ar", "48000"]
    if fmt == "AC3 (Dolby Digital)" and sr not in (32000, 44100, 48000):
        args += ["-ar", "48000"]
    meta = []
    for k, v in (tags or {}).items():
        if v:
            meta += ["-metadata", f"{k}={v}"]
    tmp = path + ".part" + ext
    cmd = [ff, "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", str(x.shape[1]), "-i", "-"] + args + meta + [tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **NOWIN)
    data = np.ascontiguousarray(x, dtype="<f4").tobytes()
    try:
        step = 1 << 22
        for i in range(0, len(data), step):
            p.stdin.write(data[i:i + step])
            if progress:
                progress(i / max(len(data), 1))
        p.stdin.close()
    except BrokenPipeError:
        pass
    err = p.stderr.read().decode(errors="replace")
    rc = p.wait()
    if rc != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise ValueError("Could not write the file:\n" + err[-300:])
    os.replace(tmp, path)


# ---------------------------------------------------------------- own project format
def save_project(path, x, sr, markers=None, extra=None):
    """.daud (.paud before) = a zip with a small json and the sound as 32-bit float, nothing is lost"""
    meta = {"app": "Digonto Audio", "format": 1, "samplerate": int(sr), "channels": int(x.shape[1]), "frames": int(len(x)),
            "markers": [[float(a), str(b)] for a, b in (markers or [])], "extra": extra or {}}
    tmp = path + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        z.writestr("meta.json", json.dumps(meta, indent=1))
        with z.open("audio.f32", "w", force_zip64=True) as f:
            step = 1 << 20
            flat = np.ascontiguousarray(x, dtype="<f4")
            for i in range(0, len(flat), step):
                f.write(flat[i:i + step].tobytes())
    os.replace(tmp, path)


def load_project(path):
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("meta.json"))
        raw = z.read("audio.f32")
    a = np.frombuffer(raw, dtype="<f4").astype(np.float32).reshape(-1, int(meta["channels"]))
    extra = {"markers": [(float(a_), str(b_)) for a_, b_ in meta.get("markers", [])], "project": True}
    extra.update(meta.get("extra", {}))
    return a, int(meta["samplerate"]), extra
