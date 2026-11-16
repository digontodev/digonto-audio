"""Digonto Audio - sound processing. Pure numpy. Audio is a float32 array shaped (samples, channels), range -1..1."""
import numpy as np

EPS = 1e-12


def db(v):
    return 20.0 * np.log10(max(abs(float(v)), EPS))


def lin(d):
    return 10.0 ** (d / 20.0)


def peak(x):
    return float(np.max(np.abs(x))) if x.size else 0.0


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


# ---------------------------------------------------------------- level
def gain(x, d_b):
    return (x * lin(d_b)).astype(np.float32)


def boost(x, d_b, protect=True, ceiling_db=-0.3):
    """make quieter audio louder. With protect on, peaks are rounded off softly instead of clipping."""
    y = x.astype(np.float64) * lin(d_b)
    if protect:
        c = lin(ceiling_db)
        t = 0.6 * c                                   # below this the sound is untouched
        a = np.abs(y)
        over = a > t
        if over.any():
            k = c - t
            y[over] = np.sign(y[over]) * (t + k * np.tanh((a[over] - t) / k))
    return y.astype(np.float32)


def normalize(x, peak_db=-1.0, rms_db=None, each=False):
    if x.size == 0:
        return x
    y = x.astype(np.float64)
    chans = [y[:, c:c + 1] for c in range(y.shape[1])] if each else [y]
    out = []
    for p in chans:
        if rms_db is not None:
            r = rms(p)
            g = lin(rms_db) / r if r > EPS else 1.0
            # never push the peak above the ceiling
            m = peak(p) * g
            lim = lin(peak_db)
            if m > lim:
                g *= lim / m
        else:
            m = peak(p)
            g = lin(peak_db) / m if m > EPS else 1.0
        out.append(p * g)
    return (np.concatenate(out, axis=1) if each else out[0]).astype(np.float32)


def fade(x, kind="in", curve="linear"):
    n = len(x)
    if n < 2:
        return x
    t = np.linspace(0.0, 1.0, n)
    if curve == "cosine":
        t = 0.5 - 0.5 * np.cos(np.pi * t)
    elif curve == "log":
        t = t ** 2.5
    if kind == "out":
        t = t[::-1]
    return (x * t[:, None]).astype(np.float32)


def remove_dc(x):
    return (x - x.mean(axis=0, keepdims=True)).astype(np.float32)


def invert(x):
    return (-x).astype(np.float32)


def reverse(x):
    return x[::-1].copy()


def to_mono(x):
    return x.mean(axis=1, keepdims=True).astype(np.float32)


def to_stereo(x):
    return np.repeat(x[:, :1], 2, axis=1) if x.shape[1] == 1 else x


def swap_channels(x):
    return x[:, ::-1].copy() if x.shape[1] == 2 else x


def pan(x, amount):
    """amount -1 (left) .. +1 (right), constant power"""
    x = to_stereo(x)
    a = (amount + 1) * np.pi / 4
    y = x.copy()
    y[:, 0] *= np.cos(a) * np.sqrt(2)
    y[:, 1] *= np.sin(a) * np.sqrt(2)
    return y.astype(np.float32)


def widen(x, width=1.5):
    """stereo width: 0 = mono, 1 = as is, 2 = very wide"""
    if x.shape[1] != 2:
        return x
    m = (x[:, 0] + x[:, 1]) / 2
    s = (x[:, 0] - x[:, 1]) / 2 * width
    return np.stack([m + s, m - s], axis=1).astype(np.float32)


# ---------------------------------------------------------------- frequency filters (zero-phase, by FFT)
def _apply_mag(x, sr, mag_fn):
    n = len(x)
    if n < 8:
        return x
    size = 1 << int(np.ceil(np.log2(n + 1)))
    f = np.fft.rfftfreq(size, 1.0 / sr)
    m = mag_fn(f).astype(np.float64)
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[1]):
        X = np.fft.rfft(x[:, c].astype(np.float64), size)
        out[:, c] = np.fft.irfft(X * m, size)[:n]
    return out


def pass_filter(x, sr, kind="low", fc=1000.0, order=4):
    fc = max(10.0, float(fc))
    if kind == "low":
        fn = lambda f: 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * order))
    else:
        fn = lambda f: 1.0 / np.sqrt(1.0 + (fc / np.maximum(f, 1e-3)) ** (2 * order))
    return _apply_mag(x, sr, fn)


EQ_BANDS = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def eq_graphic(x, sr, gains_db, freqs=None, smooth=True):
    """gains_db: one gain (dB) per band; interpolated smoothly over a log-frequency axis"""
    freqs = freqs or EQ_BANDS
    lf = np.log10(np.array(freqs, dtype=np.float64))
    g = np.array(gains_db, dtype=np.float64)
    if not np.any(g):
        return x

    def fn(f):
        lg = np.log10(np.maximum(f, 1.0))
        if smooth:
            # smooth curve through the band gains
            idx = np.interp(lg, lf, np.arange(len(lf)))
            i0 = np.floor(idx).astype(int).clip(0, len(g) - 1)
            i1 = (i0 + 1).clip(0, len(g) - 1)
            t = idx - i0
            t = t * t * (3 - 2 * t)
            gg = g[i0] * (1 - t) + g[i1] * t
        else:
            gg = np.interp(lg, lf, g)
        return 10.0 ** (gg / 20.0)
    return _apply_mag(x, sr, fn)


def eq_peak(x, sr, fc, gain_db, q=1.0):
    """one parametric bell"""
    lf = np.log2(max(fc, 10.0))

    def fn(f):
        o = np.log2(np.maximum(f, 1.0)) - lf
        return 10.0 ** (gain_db * np.exp(-0.5 * (o * 2.0 * q) ** 2) / 20.0)
    return _apply_mag(x, sr, fn)


def eq_shelf(x, sr, fc, gain_db, high=False):
    lf = np.log2(max(fc, 10.0))

    def fn(f):
        o = np.log2(np.maximum(f, 1.0)) - lf
        s = 1.0 / (1.0 + np.exp(-o * 4.0))
        return 10.0 ** (gain_db * (s if high else 1 - s) / 20.0)
    return _apply_mag(x, sr, fn)


EQ_PRESETS = {
    "Flat": [0] * 10,
    "Voice clarity": [-6, -4, -2, 0, 1, 3, 4, 3, 1, -2],
    "Bass boost": [7, 6, 4, 2, 0, 0, 0, 0, 0, 0],
    "Treble boost": [0, 0, 0, 0, 0, 1, 3, 5, 6, 7],
    "Warm": [3, 3, 2, 1, 0, -1, -2, -2, -3, -4],
    "Telephone": [-30, -30, -20, -8, 2, 6, 6, 0, -20, -30],
    "Loudness": [6, 4, 1, 0, -1, 0, 0, 2, 4, 5],
    "Reduce rumble + hiss": [-18, -10, -2, 0, 0, 0, 0, 0, -4, -9],
}


# ---------------------------------------------------------------- time based
def echo(x, sr, delay_ms=350.0, feedback=0.4, mix=0.5, repeats=6):
    """echo / delay. The tail after the end is kept."""
    d = max(1, int(sr * delay_ms / 1000.0))
    feedback = min(0.95, max(0.0, feedback))
    n = len(x)
    tail = d * repeats
    out = np.zeros((n + tail, x.shape[1]), dtype=np.float64)
    out[:n] += x * (1.0 - mix * 0.5)
    for k in range(1, repeats + 1):
        a = mix * (feedback ** (k - 1))
        if a < 0.003:
            break
        out[k * d:k * d + n] += x * a
    return out.astype(np.float32)


def _ir_reverb(sr, room=0.5, damping=0.5, pre_ms=15.0, seed=1):
    rng = np.random.RandomState(seed)
    secs = 0.4 + 2.6 * room
    n = int(sr * secs)
    t = np.arange(n) / sr
    decay = np.exp(-6.9 * t / secs)
    ir = rng.randn(n) * decay
    # damping: highs die faster
    if damping > 0:
        k = int(1 + damping * 12)
        ker = np.ones(k) / k
        slow = np.convolve(ir, ker, mode="same")
        mixv = np.clip(t / secs * (0.4 + damping), 0, 1)
        ir = ir * (1 - mixv) + slow * mixv
    pre = int(sr * pre_ms / 1000)
    ir = np.concatenate([np.zeros(pre), ir])
    for dly, a in ((0.011, 0.7), (0.019, 0.55), (0.029, 0.45), (0.041, 0.35)):   # early reflections
        i = pre + int(dly * sr * (0.6 + room))
        if i < len(ir):
            ir[i] += a * 3
    return ir / (np.sqrt(np.sum(ir ** 2)) + EPS)


def _fft_conv(sig, ir):
    n = len(sig) + len(ir) - 1
    size = 1 << int(np.ceil(np.log2(n)))
    return np.fft.irfft(np.fft.rfft(sig, size) * np.fft.rfft(ir, size), size)[:n]


def reverb(x, sr, room=0.5, damping=0.5, wet=0.3):
    ir = _ir_reverb(sr, room, damping)
    n = len(x)
    out = np.zeros((n + len(ir) - 1, x.shape[1]), dtype=np.float64)
    for c in range(x.shape[1]):
        ir_c = _ir_reverb(sr, room, damping, seed=1 + c) if x.shape[1] > 1 else ir
        w = _fft_conv(x[:, c].astype(np.float64), ir_c)
        out[:, c] = w * wet * 2.2
    out[:n] += x * (1.0 - wet * 0.6)
    return out.astype(np.float32)


def tremolo(x, sr, rate=5.0, depth=0.5):
    t = np.arange(len(x)) / sr
    m = 1.0 - depth * (0.5 - 0.5 * np.cos(2 * np.pi * rate * t))
    return (x * m[:, None]).astype(np.float32)


def chorus(x, sr, rate=0.8, depth_ms=6.0, base_ms=18.0, mix=0.5, feedback=0.0):
    n = len(x)
    t = np.arange(n) / sr
    delay = (base_ms + depth_ms * np.sin(2 * np.pi * rate * t)) * sr / 1000.0
    pos = np.arange(n) - delay
    pos = np.clip(pos, 0, n - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    fr = (pos - i0)[:, None]
    wet = x[i0] * (1 - fr) + x[i1] * fr
    return (x * (1 - mix * 0.5) + wet * mix).astype(np.float32)


def distortion(x, drive_db=12.0, mix=1.0):
    g = lin(drive_db)
    y = np.tanh(x * g) / np.tanh(g if g > 1 else 1.0)
    return (x * (1 - mix) + y * mix).astype(np.float32)


# ---------------------------------------------------------------- dynamics
def _block_gain(env_db, thresh, ratio, attack_b, release_b):
    """gain reduction (dB, <=0) per block, smoothed with attack/release"""
    over = np.maximum(env_db - thresh, 0.0)
    target = -over * (1.0 - 1.0 / ratio)
    g = np.empty_like(target)
    cur = 0.0
    ac = np.exp(-1.0 / max(attack_b, 0.01))
    rc = np.exp(-1.0 / max(release_b, 0.01))
    for i, t in enumerate(target):
        if t < cur:
            cur = ac * cur + (1 - ac) * t        # more reduction: attack
        else:
            cur = rc * cur + (1 - rc) * t        # less reduction: release
        g[i] = cur
    return g


def compress(x, sr, thresh_db=-20.0, ratio=3.0, attack_ms=10.0, release_ms=120.0, makeup_db=0.0, auto_makeup=False):
    n = len(x)
    if n < 64:
        return x
    blk = 32
    nb = (n + blk - 1) // blk
    pad = np.zeros((nb * blk, x.shape[1]), dtype=np.float32)
    pad[:n] = x
    env = np.abs(pad).max(axis=1).reshape(nb, blk).max(axis=1)
    env_db = 20 * np.log10(np.maximum(env, EPS))
    g = _block_gain(env_db, thresh_db, max(ratio, 1.0), attack_ms * sr / 1000.0 / blk, release_ms * sr / 1000.0 / blk)
    mk = makeup_db
    if auto_makeup:
        mk += -thresh_db * (1.0 - 1.0 / max(ratio, 1.0)) * 0.5
    gl = lin(g + mk)
    idx = (np.arange(n) / blk)
    gs = np.interp(idx, np.arange(nb) + 0.5, gl)
    return (x * gs[:, None]).astype(np.float32)


def limiter(x, sr, ceiling_db=-1.0, release_ms=80.0, lookahead_ms=3.0):
    n = len(x)
    if n < 8:
        return x
    c = lin(ceiling_db)
    a = np.abs(x).max(axis=1)
    la = max(1, int(sr * lookahead_ms / 1000))
    # required gain per sample, then the lowest over the look-ahead window
    req = np.minimum(1.0, c / np.maximum(a, EPS))
    k = la
    pad = np.concatenate([req, np.ones(k)])
    from numpy.lib.stride_tricks import sliding_window_view
    m = sliding_window_view(pad, k).min(axis=1)[:n] if k > 1 else req
    # release: fast to go down, slow to come back up
    rel = np.exp(-1.0 / max(sr * release_ms / 1000.0, 1.0))
    blk = 16
    nb = (n + blk - 1) // blk
    mb = np.ones(nb * blk)
    mb[:n] = m
    mb = mb.reshape(nb, blk).min(axis=1)
    out = np.empty(nb)
    cur = 1.0
    r2 = rel ** blk
    for i, v in enumerate(mb):
        cur = v if v < cur else r2 * cur + (1 - r2) * v
        out[i] = min(cur, v) if v < cur else cur
    gs = np.interp(np.arange(n) / blk, np.arange(nb) + 0.5, out)
    gs = np.minimum(gs, 1.0)
    y = x * gs[:, None]
    return np.clip(y, -c, c).astype(np.float32)


def gate(x, sr, thresh_db=-50.0, attack_ms=5.0, release_ms=120.0, floor_db=-60.0):
    n = len(x)
    if n < 64:
        return x
    blk = 64
    nb = (n + blk - 1) // blk
    pad = np.zeros((nb * blk, x.shape[1]), dtype=np.float32)
    pad[:n] = x
    env = np.abs(pad).max(axis=1).reshape(nb, blk).max(axis=1)
    open_ = (20 * np.log10(np.maximum(env, EPS)) > thresh_db).astype(np.float64)
    ac = np.exp(-blk / max(sr * attack_ms / 1000.0, 1.0))
    rc = np.exp(-blk / max(sr * release_ms / 1000.0, 1.0))
    g = np.empty(nb)
    cur = 0.0
    for i, o in enumerate(open_):
        c = ac if o > cur else rc
        cur = c * cur + (1 - c) * o
        g[i] = cur
    fl = lin(floor_db)
    g = fl + (1 - fl) * g
    gs = np.interp(np.arange(n) / blk, np.arange(nb) + 0.5, g)
    return (x * gs[:, None]).astype(np.float32)


# ---------------------------------------------------------------- spectral (STFT)
def stft(sig, n_fft=2048, hop=512):
    sig = np.asarray(sig, dtype=np.float64)
    pad = n_fft
    s = np.concatenate([np.zeros(pad), sig, np.zeros(pad + n_fft)])
    frames = 1 + (len(s) - n_fft) // hop
    from numpy.lib.stride_tricks import as_strided
    st = s.strides[0]
    fr = as_strided(s, shape=(frames, n_fft), strides=(hop * st, st))
    win = np.hanning(n_fft + 1)[:-1]
    return np.fft.rfft(fr * win, axis=1)


def istft(S, length, n_fft=2048, hop=512):
    win = np.hanning(n_fft + 1)[:-1]
    fr = np.fft.irfft(S, n_fft, axis=1) * win
    total = (S.shape[0] - 1) * hop + n_fft
    out = np.zeros(total)
    norm = np.zeros(total)
    w2 = win ** 2
    for i in range(S.shape[0]):
        out[i * hop:i * hop + n_fft] += fr[i]
        norm[i * hop:i * hop + n_fft] += w2
    out = out / np.maximum(norm, 1e-6)
    return out[n_fft:n_fft + length]


def noise_profile(noise, n_fft=2048, hop=512):
    """average spectrum of a noise-only piece"""
    S = np.abs(stft(noise[:, 0] if noise.ndim > 1 else noise, n_fft, hop))
    return np.mean(S, axis=0), np.std(S, axis=0)


def auto_noise_profile(x, n_fft=2048, hop=512):
    """no noise sample chosen: use the quietest 10 % of the frames"""
    m = x.mean(axis=1) if x.ndim > 1 else x
    S = np.abs(stft(m, n_fft, hop))
    e = S.sum(axis=1)
    k = max(2, int(len(e) * 0.1))
    q = np.argsort(e)[:k]
    return S[q].mean(axis=0), S[q].std(axis=0)


def noise_reduce(x, sr, profile=None, reduction_db=15.0, sensitivity=1.5, smoothing=2, n_fft=2048, hop=512):
    """spectral noise reduction. profile = (mean, std) magnitude of the noise; None = estimate from the quietest parts."""
    if len(x) < n_fft * 2:
        return x
    if profile is None:
        profile = auto_noise_profile(x, n_fft, hop)
    mean, std = profile
    thr = mean + sensitivity * std * 0.5 + mean * (sensitivity - 1.0) * 0.5
    floor = lin(-abs(reduction_db))
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[1]):
        S = stft(x[:, c], n_fft, hop)
        mag = np.abs(S)
        m = np.clip((mag - thr[None, :]) / np.maximum(mag, EPS), 0.0, 1.0)     # spectral subtraction gain
        m = np.maximum(m, floor)
        if smoothing > 0:      # smooth over time and frequency so it does not "warble"
            k = 2 * int(smoothing) + 1
            ker = np.ones(k) / k
            m = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 0, m)
            m = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, m)
        out[:, c] = istft(S * m, len(x), n_fft, hop)
    return out


def hum_remove(x, sr, base=50.0, harmonics=6, width=3.0):
    """notch out mains hum (50 or 60 Hz) and its harmonics"""
    def fn(f):
        m = np.ones_like(f)
        for h in range(1, harmonics + 1):
            m *= 1.0 - np.exp(-0.5 * ((f - base * h) / width) ** 2) * 0.995
        return m
    return _apply_mag(x, sr, fn)


def de_click(x, sr, strength=5.0):
    """remove short clicks / pops: a sample far from its neighbours is replaced by their average"""
    y = x.copy()
    for c in range(x.shape[1]):
        v = x[:, c].astype(np.float64)
        d = np.abs(v[1:-1] - 0.5 * (v[:-2] + v[2:]))
        lim = strength * (np.median(d) + 1e-5) * 4
        bad = np.where(d > max(lim, 0.02))[0] + 1
        for i in bad:
            y[i, c] = 0.5 * (v[i - 1] + v[i + 1])
    return y


def _phase_vocoder(sig, rate, n_fft=2048, hop=512):
    S = stft(sig, n_fft, hop)
    n_fr = S.shape[0]
    steps = np.arange(0, n_fr - 1, rate)
    omega = 2 * np.pi * hop * np.arange(S.shape[1]) / n_fft
    phase = np.angle(S[0])
    out = np.empty((len(steps), S.shape[1]), dtype=np.complex128)
    last = np.angle(S[0])
    for k, s in enumerate(steps):
        i = int(s)
        fr = s - i
        a, b = S[i], S[i + 1]
        mag = (1 - fr) * np.abs(a) + fr * np.abs(b)
        out[k] = mag * np.exp(1j * phase)
        d = np.angle(b) - np.angle(a) - omega
        d -= 2 * np.pi * np.round(d / (2 * np.pi))
        phase = phase + omega + d
    return out


def time_stretch(x, sr, rate=1.0):
    """change the speed without changing the pitch (rate 2 = twice as fast)"""
    if abs(rate - 1.0) < 1e-3 or len(x) < 4096:
        return x
    outs = []
    for c in range(x.shape[1]):
        S = _phase_vocoder(x[:, c], rate)
        n_out = int(len(x) / rate)
        outs.append(istft(S, n_out, 2048, 512))
    return np.stack(outs, axis=1).astype(np.float32)


def resample(x, ratio):
    """new length = old length * ratio (changes speed and pitch together)"""
    n = len(x)
    m = max(1, int(round(n * ratio)))
    if m == n:
        return x
    src = np.linspace(0, n - 1, m)
    out = np.empty((m, x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(src, np.arange(n), x[:, c])
    return out


def change_samplerate(x, sr_from, sr_to):
    if sr_from == sr_to:
        return x
    ratio = sr_to / sr_from
    if ratio < 1.0:      # remove what would alias
        x = pass_filter(x, sr_from, "low", 0.45 * sr_to, 8)
    return resample(x, ratio)


def pitch_shift(x, sr, semitones=0.0):
    """change the pitch without changing the length"""
    if abs(semitones) < 0.01:
        return x
    f = 2.0 ** (semitones / 12.0)
    y = time_stretch(x, sr, 1.0 / f)         # longer/shorter, same pitch
    y = resample(y, 1.0 / f)                 # back to the old length, pitch moves
    n = len(x)
    if len(y) < n:
        y = np.concatenate([y, np.zeros((n - len(y), y.shape[1]), np.float32)])
    return y[:n].astype(np.float32)


# ---------------------------------------------------------------- voice and music
def separate_center(x, sr, strength=1.0, n_fft=4096, hop=1024):
    """Split a stereo song into the 'centre' (usually the singer or speaker) and the rest (usually the music).
    It works from where each sound sits between the two speakers - it is not artificial intelligence, so a result is
    clean only when the voice is exactly in the middle and the instruments are spread out."""
    if x.shape[1] < 2:
        raise ValueError("Needs a stereo file (two channels): the voice and music are told apart by their place between the speakers.")
    L = stft(x[:, 0], n_fft, hop)
    R = stft(x[:, 1], n_fft, hop)
    num = 2.0 * np.abs(L * np.conj(R))
    den = np.abs(L) ** 2 + np.abs(R) ** 2 + 1e-12
    sim = num / den                                     # 1 = exactly in the centre
    # same level on both sides too
    bal = 1.0 - np.abs(np.abs(L) - np.abs(R)) / (np.abs(L) + np.abs(R) + 1e-12)
    cen = np.clip(sim * bal, 0, 1) ** (1.0 + 2.0 * strength)
    # ignore very low (bass, kick) and very high bins: usually not the voice
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = np.interp(f, [0, 80, 160, 6000, 12000, sr / 2 + 1], [0, 0, 1, 1, 0.5, 0.2])
    cen = cen * band[None, :]
    M = (L + R) / 2
    voice = istft(M * cen, len(x), n_fft, hop)
    voice_st = np.stack([voice, voice], axis=1)
    music_L = istft(L - M * cen, len(x), n_fft, hop)
    music_R = istft(R - M * cen, len(x), n_fft, hop)
    music = np.stack([music_L, music_R], axis=1)
    return voice_st.astype(np.float32), music.astype(np.float32)


def remove_center(x, sr, strength=1.0):
    return separate_center(x, sr, strength)[1]


def keep_center(x, sr, strength=1.0):
    return separate_center(x, sr, strength)[0]


# ---------------------------------------------------------------- generators
def silence(sr, ch, secs):
    return np.zeros((int(sr * secs), ch), dtype=np.float32)


def tone(sr, ch, secs, freq=440.0, kind="sine", level_db=-6.0):
    n = int(sr * secs)
    t = np.arange(n) / sr
    ph = (t * freq) % 1.0
    if kind == "square":
        y = np.where(ph < 0.5, 1.0, -1.0)
    elif kind == "saw":
        y = 2 * ph - 1
    elif kind == "triangle":
        y = 4 * np.abs(ph - 0.5) - 1
    elif kind == "white noise":
        y = np.random.uniform(-1, 1, n)
    elif kind == "pink noise":
        w = np.random.randn(n)
        F = np.fft.rfft(w)
        f = np.maximum(np.arange(len(F)), 1)
        y = np.fft.irfft(F / np.sqrt(f), n)
        y /= max(np.max(np.abs(y)), EPS)
    else:
        y = np.sin(2 * np.pi * ph)
    y = y * lin(level_db)
    return np.repeat(y[:, None], ch, axis=1).astype(np.float32)


# ---------------------------------------------------------------- analysis
def stats(x, sr):
    n = len(x)
    p = peak(x)
    r = rms(x)
    clipped = int(np.sum(np.abs(x) >= 0.999))
    dc = [float(v) for v in x.mean(axis=0)] if n else [0.0]
    return {"samples": n, "seconds": n / sr if sr else 0, "peak_db": db(p), "rms_db": db(r), "clipped": clipped, "dc": dc,
            "channels": x.shape[1] if x.ndim > 1 else 1}


def peaks_for(x, start, end, width):
    """min/max per pixel column for drawing: returns (mins, maxs) each (width, channels)"""
    n = end - start
    ch = x.shape[1]
    if width <= 0 or n <= 0:
        return np.zeros((0, ch)), np.zeros((0, ch))
    if n <= width:
        seg = x[start:end]
        xs = np.linspace(0, width - 1, n)
        return seg, seg
    seg = x[start:end]
    edges = np.linspace(0, n, width + 1).astype(np.int64)[:-1]
    edges = np.minimum(edges, n - 1)
    mins = np.minimum.reduceat(seg, edges, axis=0)
    maxs = np.maximum.reduceat(seg, edges, axis=0)
    return mins, maxs
