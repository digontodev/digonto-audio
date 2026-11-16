"""Digonto Audio - the list of effects. Each one: name, settings (shown in a dialog) and the function that does the work."""
import numpy as np

from . import dsp

F, I, C, B = "f", "i", "c", "b"
# field = (key, label, kind, default, low, high, suffix_or_choices)


def _chain(*fs):
    def run(x, sr, p, ctx):
        for f in fs:
            x = f(x, sr, p, ctx)
        return x
    return run


def _leveler(x, sr, p, ctx):
    s = p["strength"] / 100.0
    y = dsp.compress(x, sr, thresh_db=-34 + 12 * (1 - s), ratio=1.5 + 4.5 * s, attack_ms=15, release_ms=250)
    y = dsp.compress(y, sr, thresh_db=-22, ratio=2.5, attack_ms=5, release_ms=120)
    return dsp.normalize(y, p["peak"])


def _noise(x, sr, p, ctx):
    prof = ctx.get("noise") if p.get("use_learned") else None
    return dsp.noise_reduce(x, sr, prof, p["reduction"], p["sens"], int(p["smooth"]))


def _speed(x, sr, p, ctx):
    return dsp.resample(x, 1.0 / p["factor"])


def _echo(x, sr, p, ctx):
    return dsp.echo(x, sr, p["delay"], p["feedback"] / 100.0, p["mix"] / 100.0, int(p["repeats"]))


# group -> list of (name, fields, function)
EFFECTS = {
    "Volume": [
        ("Amplify (louder / quieter)…", [("db", "Change by", F, 6.0, -60, 60, " dB")], lambda x, sr, p, c: dsp.gain(x, p["db"])),
        ("Make quiet audio louder…", [("db", "Boost by", F, 12.0, 0, 48, " dB"), ("protect", "Round off peaks (no harsh clipping)", B, True, 0, 0, None)],
         lambda x, sr, p, c: dsp.boost(x, p["db"], p["protect"])),
        ("Normalize…", [("peak", "Loudest peak at", F, -1.0, -30, 0, " dB"), ("rms_on", "Also match average loudness", B, False, 0, 0, None),
                        ("rms", "Average loudness", F, -18.0, -50, -3, " dB"), ("each", "Each channel on its own", B, False, 0, 0, None)],
         lambda x, sr, p, c: dsp.normalize(x, p["peak"], p["rms"] if p["rms_on"] else None, p["each"])),
        ("Even out voice (leveler)…", [("strength", "Strength", I, 60, 0, 100, " %"), ("peak", "Loudest peak at", F, -1.0, -12, 0, " dB")],
         _leveler),
        ("Fade in…", [("curve", "Curve", C, "cosine", 0, 0, ["linear", "cosine", "log"])], lambda x, sr, p, c: dsp.fade(x, "in", p["curve"])),
        ("Fade out…", [("curve", "Curve", C, "cosine", 0, 0, ["linear", "cosine", "log"])], lambda x, sr, p, c: dsp.fade(x, "out", p["curve"])),
        ("Compressor…", [("thr", "Threshold", F, -20.0, -60, 0, " dB"), ("ratio", "Ratio", F, 3.0, 1, 20, " : 1"), ("att", "Attack", F, 10.0, 1, 200, " ms"),
                         ("rel", "Release", F, 120.0, 10, 1500, " ms"), ("mk", "Make-up gain", F, 3.0, 0, 24, " dB")],
         lambda x, sr, p, c: dsp.compress(x, sr, p["thr"], p["ratio"], p["att"], p["rel"], p["mk"])),
        ("Limiter…", [("ceil", "Never louder than", F, -1.0, -20, 0, " dB"), ("rel", "Release", F, 80.0, 10, 500, " ms")],
         lambda x, sr, p, c: dsp.limiter(x, sr, p["ceil"], p["rel"])),
    ],
    "Clean up": [
        ("Remove background noise…", [("use_learned", "Use the noise I picked (Learn noise)", B, False, 0, 0, None),
                                      ("reduction", "Reduce by", F, 15.0, 3, 40, " dB"), ("sens", "Sensitivity", F, 1.5, 0.5, 4.0, ""),
                                      ("smooth", "Smoothing (less 'bubbly' sound)", I, 2, 0, 6, "")], _noise),
        ("Remove hum (50 / 60 Hz)…", [("base", "Mains frequency", C, "50", 0, 0, ["50", "60"]), ("harm", "Harmonics", I, 6, 1, 12, "")],
         lambda x, sr, p, c: dsp.hum_remove(x, sr, float(p["base"]), int(p["harm"]))),
        ("Noise gate (silence the gaps)…", [("thr", "Open above", F, -45.0, -80, -10, " dB"), ("att", "Attack", F, 5.0, 1, 100, " ms"),
                                           ("rel", "Release", F, 150.0, 10, 1000, " ms"), ("floor", "Gaps get", F, -70.0, -90, -10, " dB")],
         lambda x, sr, p, c: dsp.gate(x, sr, p["thr"], p["att"], p["rel"], p["floor"])),
        ("Remove clicks and pops…", [("s", "Strength", F, 5.0, 1, 20, "")], lambda x, sr, p, c: dsp.de_click(x, sr, p["s"])),
        ("Remove DC offset", [], lambda x, sr, p, c: dsp.remove_dc(x)),
    ],
    "Filters": [
        ("High-pass (cut low rumble)…", [("fc", "Cut below", F, 80.0, 20, 2000, " Hz"), ("order", "Steepness", I, 4, 1, 12, "")],
         lambda x, sr, p, c: dsp.pass_filter(x, sr, "high", p["fc"], int(p["order"]))),
        ("Low-pass (cut hiss)…", [("fc", "Cut above", F, 8000.0, 200, 20000, " Hz"), ("order", "Steepness", I, 4, 1, 12, "")],
         lambda x, sr, p, c: dsp.pass_filter(x, sr, "low", p["fc"], int(p["order"]))),
        ("Bell (boost or cut one frequency)…", [("fc", "Frequency", F, 1000.0, 30, 18000, " Hz"), ("g", "Gain", F, 6.0, -24, 24, " dB"), ("q", "Width (Q)", F, 1.0, 0.2, 8, "")],
         lambda x, sr, p, c: dsp.eq_peak(x, sr, p["fc"], p["g"], p["q"])),
        ("Bass shelf…", [("fc", "Below", F, 200.0, 40, 1000, " Hz"), ("g", "Gain", F, 6.0, -24, 24, " dB")],
         lambda x, sr, p, c: dsp.eq_shelf(x, sr, p["fc"], p["g"], False)),
        ("Treble shelf…", [("fc", "Above", F, 5000.0, 1000, 16000, " Hz"), ("g", "Gain", F, 4.0, -24, 24, " dB")],
         lambda x, sr, p, c: dsp.eq_shelf(x, sr, p["fc"], p["g"], True)),
    ],
    "Echo and space": [
        ("Echo…", [("delay", "Delay", F, 350.0, 20, 2000, " ms"), ("feedback", "Repeats fade slowly", I, 45, 0, 90, " %"), ("mix", "Amount", I, 50, 0, 100, " %"),
                   ("repeats", "Number of repeats", I, 6, 1, 20, "")], _echo),
        ("Reverb (room sound)…", [("room", "Room size", I, 50, 0, 100, " %"), ("damp", "Soft walls (less bright)", I, 50, 0, 100, " %"), ("wet", "Amount", I, 30, 0, 100, " %")],
         lambda x, sr, p, c: dsp.reverb(x, sr, p["room"] / 100.0, p["damp"] / 100.0, p["wet"] / 100.0)),
        ("Chorus / Flanger…", [("rate", "Speed", F, 0.8, 0.05, 8, " Hz"), ("depth", "Depth", F, 6.0, 0.5, 20, " ms"), ("base", "Delay", F, 18.0, 1, 40, " ms"), ("mix", "Amount", I, 50, 0, 100, " %")],
         lambda x, sr, p, c: dsp.chorus(x, sr, p["rate"], p["depth"], p["base"], p["mix"] / 100.0)),
        ("Tremolo…", [("rate", "Speed", F, 5.0, 0.1, 30, " Hz"), ("depth", "Depth", I, 50, 0, 100, " %")], lambda x, sr, p, c: dsp.tremolo(x, sr, p["rate"], p["depth"] / 100.0)),
        ("Distortion…", [("drive", "Drive", F, 12.0, 0, 40, " dB"), ("mix", "Amount", I, 100, 0, 100, " %")], lambda x, sr, p, c: dsp.distortion(x, p["drive"], p["mix"] / 100.0)),
    ],
    "Speed and pitch": [
        ("Pitch (keep the speed)…", [("st", "Change by", F, 2.0, -24, 24, " semitones")], lambda x, sr, p, c: dsp.pitch_shift(x, sr, p["st"])),
        ("Tempo (keep the pitch)…", [("rate", "Speed", F, 1.25, 0.25, 4.0, " ×")], lambda x, sr, p, c: dsp.time_stretch(x, sr, p["rate"])),
        ("Speed (tape style, pitch follows)…", [("factor", "Speed", F, 1.25, 0.25, 4.0, " ×")], _speed),
    ],
    "Stereo": [
        ("Make mono", [], lambda x, sr, p, c: dsp.to_mono(x)),
        ("Make stereo", [], lambda x, sr, p, c: dsp.to_stereo(x)),
        ("Swap left and right", [], lambda x, sr, p, c: dsp.swap_channels(x)),
        ("Pan…", [("pan", "Left ← → Right", I, 0, -100, 100, " %")], lambda x, sr, p, c: dsp.pan(x, p["pan"] / 100.0)),
        ("Stereo width…", [("w", "Width", I, 150, 0, 300, " %")], lambda x, sr, p, c: dsp.widen(x, p["w"] / 100.0)),
        ("Remove the voice (karaoke)…", [("s", "Strength", I, 70, 0, 100, " %")], lambda x, sr, p, c: dsp.remove_center(x, sr, p["s"] / 100.0)),
        ("Keep only the voice…", [("s", "Strength", I, 70, 0, 100, " %")], lambda x, sr, p, c: dsp.keep_center(x, sr, p["s"] / 100.0)),
    ],
    "Other": [
        ("Reverse", [], lambda x, sr, p, c: dsp.reverse(x)),
        ("Invert (flip up/down)", [], lambda x, sr, p, c: dsp.invert(x)),
    ],
}
