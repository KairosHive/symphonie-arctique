"""
Diagnose stem balance: per-stem RMS, peak, and rough frequency-band energy
distribution. Helps identify whether one stem is dominating (e.g. the high-pad
case) or whether stems are well-balanced.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import numpy as np
import soundfile as sf


def db(x):
    return 20 * np.log10(max(x, 1e-9))


def band_rms(audio: np.ndarray, sr: int, bands_hz: list[tuple[float, float]]) -> dict:
    """
    Crude band energy via FFT — log-power per octave band.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = audio.shape[0]
    # Window the audio
    win = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(audio * win))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    out = {}
    for lo, hi in bands_hz:
        mask = (freqs >= lo) & (freqs < hi)
        e = float(np.sqrt(np.mean(spectrum[mask] ** 2))) if mask.any() else 0.0
        out[f"{int(lo)}-{int(hi)}Hz"] = e
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    args = p.parse_args()

    src = Path(args.src)
    bands = [(20, 200), (200, 800), (800, 3000), (3000, 8000), (8000, 22050)]
    band_labels = ["sub", "low-mid", "mid", "presence", "air"]

    rows = []
    for wav in sorted(src.glob("stem_*cm_envmod.wav")):
        audio, sr = sf.read(str(wav), dtype="float32")
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        bands_e = band_rms(audio, sr, bands)
        # Total band energy (for relative %)
        total = sum(bands_e.values()) or 1.0
        pct = {k: 100.0 * v / total for k, v in bands_e.items()}
        rows.append({
            "name": wav.name,
            "rms": rms,
            "peak": peak,
            "rms_db": db(rms),
            "peak_db": db(peak),
            "bands_pct": pct,
        })

    print(f"\n{'stem':<35} {'rms':>7} {'rms_dB':>8} {'peak':>6} "
          f"{band_labels[0]:>5} {band_labels[1]:>8} {band_labels[2]:>5} "
          f"{band_labels[3]:>10} {band_labels[4]:>5}")
    print("-" * 120)
    for r in rows:
        bp = r["bands_pct"]
        keys = list(bp.keys())
        print(
            f"{r['name']:<35} {r['rms']:>7.4f} {r['rms_db']:>+8.1f} {r['peak']:>6.3f} "
            f"{bp[keys[0]]:>4.0f}% {bp[keys[1]]:>7.0f}% {bp[keys[2]]:>4.0f}% "
            f"{bp[keys[3]]:>9.0f}% {bp[keys[4]]:>4.0f}%"
        )
    print("\nBands:", "  ".join(f"{l}={b[0]}-{b[1]}Hz" for l, b in zip(band_labels, bands)))


if __name__ == "__main__":
    main()
