"""
Apply soil envelope as audio effects.

Two modulations applied in sequence:
  1. Amplitude envelope (with optional gamma curve for stronger contrast):
        out = audio * env^gamma   (env in [floor, 1.0])
        gamma > 1 → makes cold sections much quieter (more dramatic dynamic).
  2. Dynamic low-pass cutoff:
        cutoff(t) = lpf_low_hz + (lpf_high_hz - lpf_low_hz) * env(t)
        At low envelope (cold): cutoff drops near lpf_low_hz, audio sounds dark/muffled.
        At high envelope (warm): cutoff opens to lpf_high_hz, audio is bright.
        Implemented with frame-wise biquad filtering.

Together these make the soil signal *audibly* shape the music's character, not
just its loudness.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import sosfilt, butter

from soil_data import envelope_for_depth, find_first_full_year, load_csv


def normalize_rms(
    audio: np.ndarray,
    target_rms: float = 0.15,
    *,
    max_gain: float = 6.0,
    peak_limit: float = 0.85,
) -> np.ndarray:
    """
    Loudness-normalize audio to a target RMS so all stems start at equal level.
    target_rms=0.15 -> roughly -16 dBFS, comfortable headroom for layering.

    Two safety guards prevent sparse content (occasional cracks + long silences)
    from blowing up peaks:
      max_gain  : cap the multiplicative scaling factor (default 6x).
      peak_limit: after RMS scaling, peak-limit to this absolute value
                  (default 0.85).
    """
    arr = np.asarray(audio, dtype=np.float64)
    cur = float(np.sqrt(np.mean(arr ** 2)))
    if cur < 1e-7:
        return audio
    gain = min(target_rms / cur, max_gain)
    out = arr * gain
    peak = float(np.max(np.abs(out)))
    if peak > peak_limit:
        out = out * (peak_limit / peak)
    return out.astype(np.float32)


def dynamic_hpf(
    audio: np.ndarray,
    env: np.ndarray,
    sr: int,
    *,
    cutoff_low_hz: float = 20.0,
    cutoff_high_hz: float = 400.0,
    **_unused,
) -> np.ndarray:
    """
    Dynamic high-pass via PARAMETRIC CROSSFADE (no IIR state discontinuities).
    Apply the full HPF (heavy bass cut) once, then crossfade between filtered
    (cold) and dry (warm) using env. Sample-smooth, glitch-free.

      env=0 (cold) -> fully filtered (heavy bass cut)
      env=1 (warm) -> dry (full bass)
    """
    from scipy.signal import butter, sosfiltfilt

    if audio.ndim == 1:
        audio2 = audio[:, None]
    else:
        audio2 = audio

    # Heavy bass cut version
    cutoff = max(min(cutoff_high_hz, sr / 2 - 100), 10)
    sos = butter(2, cutoff / (sr / 2), btype="highpass", output="sos")
    filtered = sosfiltfilt(sos, audio2, axis=0).astype(np.float32)

    # If cutoff_low_hz > 20 there's still some HPF in the "warm" version
    dry = audio2.astype(np.float32)
    if cutoff_low_hz > 20:
        sos_low = butter(2, cutoff_low_hz / (sr / 2), btype="highpass", output="sos")
        dry = sosfiltfilt(sos_low, dry, axis=0).astype(np.float32)

    env_c = np.clip(env.astype(np.float32), 0.0, 1.0)[:, None]
    out = filtered * (1.0 - env_c) + dry * env_c
    if audio.ndim == 1:
        out = out[:, 0]
    return out.astype(np.float32)


def dynamic_stereo_width(
    audio: np.ndarray,
    env: np.ndarray,
    *,
    min_width: float = 0.0,
    max_width: float = 1.0,
) -> np.ndarray:
    """
    Sample-wise stereo width modulation via Mid/Side processing.
    env=0 -> width=min_width (closer to mono)
    env=1 -> width=max_width (full stereo)
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        return audio  # only meaningful on stereo
    L = audio[:, 0].astype(np.float32)
    R = audio[:, 1].astype(np.float32)
    M = (L + R) * 0.5
    S = (L - R) * 0.5
    env = np.clip(env.astype(np.float32), 0.0, 1.0)
    width = min_width + (max_width - min_width) * env
    L_new = M + width * S
    R_new = M - width * S
    return np.stack([L_new, R_new], axis=1)


def dynamic_tremolo(
    audio: np.ndarray,
    env: np.ndarray,
    sr: int,
    *,
    rate_hz: float = 5.0,
    max_depth: float = 0.5,
) -> np.ndarray:
    """
    Envelope-driven tremolo depth. The tremolo LFO is constant rate; its DEPTH
    follows env (cold=no tremolo, warm=strong tremolo).
    Result: warm sections oscillate audibly, cold sections are steady.
    """
    n = audio.shape[0]
    t = np.arange(n, dtype=np.float32) / sr
    lfo = (1.0 + np.cos(2 * np.pi * rate_hz * t)) * 0.5  # in [0, 1]
    env = np.clip(env.astype(np.float32), 0.0, 1.0)
    depth = max_depth * env
    # modulator: 1 when lfo at peak (cos=+1 => lfo=1 => no attenuation)
    # at lfo trough we attenuate by `depth`
    modulator = 1.0 - depth + depth * lfo
    if audio.ndim == 1:
        return (audio * modulator).astype(np.float32)
    return (audio * modulator[:, None]).astype(np.float32)


def apply_modulation_envelope(
    audio: np.ndarray,
    env_norm: np.ndarray,
    sr: int,
    *,
    mod_depth: float = 0.5,
    gamma: float = 1.0,
    apply_lpf: bool = True,
    lpf_low_hz: float = 2000.0,
    lpf_high_hz: float = 16000.0,
) -> np.ndarray:
    """
    Gentle envelope: modulate audio AROUND its baseline level instead of
    crushing it toward silence.

      env_norm:   1D array in [0, 1] (the soil envelope, pre-normalized)
      mod_depth:  modulation strength in [0, 1].
                    0 = no effect; 1 = cold sections silent.
                    0.5 → loudness oscillates between 50% and 100% of baseline.
      gamma:      shape the env curve (>1 → more time spent at low values).
      apply_lpf:  dynamic LPF whose cutoff also follows env. Default 2000-16000 Hz
                  (much gentler than the v3/v4 350-Hz floor — cold = darker but not
                  muffled to silence).

    Returns audio multiplied by an envelope-driven gain in `[1 - mod_depth, 1]`.
    """
    env_norm = np.clip(np.asarray(env_norm, dtype=np.float32), 0.0, 1.0)
    env_curve = env_norm ** gamma
    gain = (1.0 - mod_depth) + mod_depth * env_curve  # in [1 - mod_depth, 1]

    if audio.ndim == 1:
        out = audio * gain
    else:
        out = audio * gain[:, None]

    if apply_lpf:
        out = dynamic_lpf(out, env_curve, sr,
                          cutoff_low_hz=lpf_low_hz,
                          cutoff_high_hz=lpf_high_hz)
    return out.astype(np.float32)


def gamma_envelope(env: np.ndarray, floor: float, gamma: float) -> np.ndarray:
    """
    Re-shape envelope with a gamma curve.
    Input env is assumed already in [floor, 1.0].  Output stays in [floor, 1.0].
    gamma > 1 stretches the lower part toward floor (darker contrast).
    """
    # Map env from [floor, 1] to [0, 1], apply gamma, map back to [floor, 1].
    norm = (env - floor) / (1.0 - floor)
    norm = np.clip(norm, 0.0, 1.0) ** gamma
    return floor + norm * (1.0 - floor)


def dynamic_lpf(
    audio: np.ndarray,
    env: np.ndarray,
    sr: int,
    *,
    cutoff_low_hz: float = 350.0,
    cutoff_high_hz: float = 16000.0,
    **_unused,
) -> np.ndarray:
    """
    Dynamic low-pass via PARAMETRIC CROSSFADE (no IIR state discontinuities).
    Build the heavy-LPF version once with sosfiltfilt (zero-phase), then
    sample-wise crossfade between dark (filtered) and bright (dry).

      env=0 (cold) -> fully filtered (dark, muffled, cutoff ~ cutoff_low_hz)
      env=1 (warm) -> dry / full bandwidth (cutoff ~ cutoff_high_hz)

    Glitch-free even with rapidly-changing env, because we never reset filter
    state during processing — we only blend two pre-computed signals.
    """
    from scipy.signal import sosfiltfilt

    if audio.ndim == 1:
        audio2 = audio[:, None]
    else:
        audio2 = audio

    cutoff = max(min(cutoff_low_hz, sr / 2 - 100), 20)
    sos = butter(2, cutoff / (sr / 2), btype="lowpass", output="sos")
    filtered_dark = sosfiltfilt(sos, audio2, axis=0).astype(np.float32)

    dry = audio2.astype(np.float32)
    if cutoff_high_hz < sr / 2 - 100:
        cutoff_hi = max(cutoff_high_hz, 50)
        sos_hi = butter(2, cutoff_hi / (sr / 2), btype="lowpass", output="sos")
        dry = sosfiltfilt(sos_hi, dry, axis=0).astype(np.float32)

    env_c = np.clip(env.astype(np.float32), 0.0, 1.0)[:, None]
    out = filtered_dark * (1.0 - env_c) + dry * env_c
    if audio.ndim == 1:
        out = out[:, 0]
    return out.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--depth", required=True)
    p.add_argument("--window", choices=["first_year", "full_history"], default="first_year")
    p.add_argument("--floor", type=float, default=0.02, help="Min envelope value")
    p.add_argument("--gamma", type=float, default=2.0, help="Envelope contrast curve")
    p.add_argument("--smooth_s", type=float, default=0.5)
    p.add_argument("--lpf_low_hz", type=float, default=350.0)
    p.add_argument("--lpf_high_hz", type=float, default=16000.0)
    p.add_argument("--no_lpf", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"[load] {args.src}")
    audio, sr = sf.read(args.src, dtype="float32")
    if audio.ndim == 1:
        audio = audio[:, None]
    n = audio.shape[0]
    print(f"[load] {n} samples @ {sr}Hz, {audio.shape}")

    print(f"[soil] depth={args.depth}cm window={args.window}")
    df = load_csv()
    if args.window == "first_year":
        start, length = find_first_full_year(df, args.depth)
    else:
        start, length = 0, None
    env_obj = envelope_for_depth(
        df,
        args.depth,
        target_samples=n,
        start_day=start,
        n_days=length,
        floor=args.floor,
        smooth_kernel=int(sr * args.smooth_s),
    )
    env = env_obj.values
    print(
        f"[env ] raw {env_obj.raw_min:.2f}..{env_obj.raw_max:.2f}°C  std={env_obj.raw_std:.2f}  "
        f"floor={args.floor}"
    )

    # Gamma curve
    env_g = gamma_envelope(env, floor=args.floor, gamma=args.gamma)
    print(f"[env ] gamma={args.gamma}  effective range [{env_g.min():.3f}, {env_g.max():.3f}]")

    # Apply amplitude
    out = audio * env_g[:, None]

    # Dynamic LPF
    if not args.no_lpf:
        print(f"[lpf ] {args.lpf_low_hz:.0f}..{args.lpf_high_hz:.0f} Hz")
        out = dynamic_lpf(
            out, env_g, sr,
            cutoff_low_hz=args.lpf_low_hz,
            cutoff_high_hz=args.lpf_high_hz,
        )

    peak = float(np.max(np.abs(out)))
    target = 10 ** (-1.0 / 20)
    if peak > target:
        out = out * (target / peak)
        print(f"[norm] peak {peak:.3f} → {target:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, out, sr)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
