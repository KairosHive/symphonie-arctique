"""
Synthesize each depth's EMD-derived microtonal chord as a short stereo audio
clip that can be fed to StableAudioPipeline as `initial_audio_waveforms`.

The chord is the set of frequencies extracted from depth_chords_microtonal.mid
and stored in outputs/depth_harmonies.json. Synthesis uses additive sines with
a soft envelope; partial amplitudes follow a natural-harmonic-series-style
falloff (lower freqs louder).

The resulting audio prompt seeds the diffusion process with the *spectral
content* of the depth's chord. SAO then generates orchestral material that
"starts from" that spectrum -- a much more direct way of bringing the EMD
microtonal pitches into the final audio than text-prompt cues.

Usage:
    from synthesize_chord import synthesize_chord
    audio = synthesize_chord(freqs, duration_s=47.0, sr=44100)
    # audio shape: (n_samples, 2) float32 in [-1, 1]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import numpy as np


def synthesize_chord(
    freqs: list[float],
    duration_s: float = 47.0,
    sr: int = 44100,
    *,
    target_peak: float = 0.30,
    fade_s: float = 0.5,
    stereo_phase_jitter: float = 0.15,
    weight_exponent: float = 0.6,
    max_freq_hz: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Synthesize a chord from a list of frequencies as a stereo additive-sine tone.

    Parameters
    ----------
    freqs : list[float]
        Frequencies in Hz (any number; freqs above Nyquist are skipped).
    duration_s : float
        Output duration. Default 47 s to match Stable Audio Open's max output.
    sr : int
        Sample rate. 44.1 kHz matches SAO.
    target_peak : float
        Final absolute peak level. Default 0.30 (~ -10 dBFS) — quiet so the
        model has room to add material on top without clipping.
    fade_s : float
        Half-cosine fade in and out length. Default 0.5 s.
    stereo_phase_jitter : float
        Small random L/R phase offset (radians) per partial for stereo width.
        Default 0.15.
    weight_exponent : float
        Amplitude falloff exponent vs partial index (sorted lowest-first).
        weight_i = 1 / (i+1)^exponent. Default 0.6 (mild falloff; lower
        partials a bit louder).
    max_freq_hz : float or None
        Optional cap; partials above this are dropped before synthesis.
    seed : int
        RNG seed for the stereo jitter.

    Returns
    -------
    np.ndarray, shape (n_samples, 2), float32, values roughly in [-target_peak, target_peak].
    """
    rng = np.random.default_rng(seed)
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr

    nyquist = sr * 0.5 - 100  # leave a small margin
    freqs_sorted = sorted(f for f in freqs if 1.0 < f < nyquist)
    if max_freq_hz is not None:
        freqs_sorted = [f for f in freqs_sorted if f <= max_freq_hz]
    if not freqs_sorted:
        raise ValueError("no usable frequencies after filtering")

    out = np.zeros((n, 2), dtype=np.float32)
    for i, f in enumerate(freqs_sorted):
        w = 1.0 / (i + 1) ** weight_exponent
        phase_l = float(rng.uniform(0, 0.05))
        phase_r = float(phase_l + rng.uniform(-stereo_phase_jitter, stereo_phase_jitter))
        ang = (2 * np.pi * f * t).astype(np.float32)
        out[:, 0] += (w * np.sin(ang + phase_l)).astype(np.float32)
        out[:, 1] += (w * np.sin(ang + phase_r)).astype(np.float32)

    # Soft fade in/out (half-cosine)
    fade_n = int(sr * fade_s)
    if fade_n > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_n)).astype(np.float32))
        out[:fade_n] *= ramp[:, None]
        out[-fade_n:] *= ramp[::-1, None]

    peak = float(np.max(np.abs(out)))
    if peak > 1e-7:
        out *= target_peak / peak

    return out.astype(np.float32)


def synthesize_chord_for_depth(
    depth_cm: int,
    *,
    duration_s: float = 47.0,
    sr: int = 44100,
    harmonies_path: str | Path = "outputs/depth_harmonies.json",
    **kwargs,
) -> np.ndarray:
    """Load the depth's chord from harmonies JSON and synthesize."""
    harmonies = json.loads(Path(harmonies_path).read_text(encoding="utf-8"))
    key = str(depth_cm)
    if key not in harmonies:
        raise KeyError(f"depth {depth_cm} not in {harmonies_path}")
    freqs = harmonies[key]["raw_freqs"]
    return synthesize_chord(freqs, duration_s=duration_s, sr=sr, **kwargs)


if __name__ == "__main__":
    # Smoke test: render and save one chord per depth.
    import soundfile as sf

    out_dir = Path("outputs/chord_seeds")
    out_dir.mkdir(parents=True, exist_ok=True)

    harmonies = json.loads(Path("outputs/depth_harmonies.json").read_text(encoding="utf-8"))
    for d_str in harmonies:
        d = int(d_str)
        audio = synthesize_chord_for_depth(d, duration_s=10.0)  # 10 s preview
        p = out_dir / f"chord_seed_{d:>04d}cm.wav"
        sf.write(p, audio, 44100)
        print(f"[save] {p}  ({len(harmonies[d_str]['raw_freqs'])} partials, "
              f"range {min(harmonies[d_str]['raw_freqs']):.1f}..{max(harmonies[d_str]['raw_freqs']):.1f} Hz)")
