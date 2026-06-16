"""
Mix per-depth stems into a single layered piece.

Two mix strategies:
  --mode equal    : all stems summed at equal weight, then peak-normalized.
  --mode physical : weight each stem by its temperature variance (std-of-temp).
                    Volatile depths (surface) get more presence;
                    stable deep depths sit underneath as drone foundation.

Usage:
    python mix.py --mode physical --src outputs/stems --out outputs/mix_physical.wav
    python mix.py --mode equal    --src outputs/stems --out outputs/mix_equal.wav
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="outputs/stems",
                   help="Directory containing stem_NNNNcm_envmod.wav files + manifest.json")
    p.add_argument("--mode", choices=["equal", "physical"], default="physical")
    p.add_argument("--variant", choices=["raw", "envmod"], default="envmod",
                   help="Mix the raw stems or the envelope-modulated stems")
    p.add_argument("--gain_floor_db", type=float, default=-18.0,
                   help="Lowest stem gain in dB (in physical mode)")
    p.add_argument("--gain_ceil_db", type=float, default=0.0,
                   help="Highest stem gain in dB (in physical mode)")
    p.add_argument(
        "--equalize_rms", type=float, default=0.0,
        help="If >0, renormalize each stem to this target RMS before mixing. "
             "Sparse stems get boosted; dense stems get attenuated. "
             "Recommended 0.10-0.15. With cap of 4x to prevent explosions.",
    )
    p.add_argument(
        "--per_stem_gain_db", default="",
        help='Manual per-stem dB gain after equalize_rms. Format: '
             '"200=-6,700=+3,900=+3,1100=+6". Applied after rms-norm, before mix mode.',
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    src = Path(args.src)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))

    # Collect (depth, audio, sr, weight)
    print(f"[mix ] mode={args.mode} variant={args.variant} src={src}  "
          f"equalize_rms={args.equalize_rms}")
    stems = []
    sr_check = None
    n_check = None
    for entry in manifest:
        d = entry["depth_cm"]
        path = entry[f"{args.variant}_wav"]
        audio, sr = sf.read(path, dtype="float32")
        # Per-stem RMS equalization (lifts sparse stems up to common loudness)
        if args.equalize_rms > 0:
            cur_rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))
            if cur_rms > 1e-7:
                gain = min(args.equalize_rms / cur_rms, 4.0)  # cap 4x = +12 dB
                pre_peak = float(np.max(np.abs(audio)))
                audio = audio * gain
                # Soft peak limit at 0.95 to avoid runaway clipping
                post_peak = float(np.max(np.abs(audio)))
                if post_peak > 0.95:
                    audio = audio * (0.95 / post_peak)
                print(f"   d={d:>4}cm  rms_pre={cur_rms:.4f}  gain={20*np.log10(gain):+.1f}dB  "
                      f"peak_pre={pre_peak:.3f}  peak_post={float(np.max(np.abs(audio))):.3f}")

        # Manual per-stem dB gain on top of RMS-equalization
        if args.per_stem_gain_db:
            manual_gains = {}
            for tok in args.per_stem_gain_db.split(","):
                if "=" not in tok:
                    continue
                k, v = tok.split("=")
                manual_gains[int(k.strip())] = float(v.strip())
            if d in manual_gains:
                amp = 10 ** (manual_gains[d] / 20)
                audio = audio * amp
                print(f"   d={d:>4}cm  manual gain {manual_gains[d]:+.1f}dB")
        if sr_check is None:
            sr_check = sr
            n_check = audio.shape[0]
        if sr != sr_check or audio.shape[0] != n_check:
            raise ValueError(f"stem {path} has mismatched sr/length: {sr}, {audio.shape[0]}")
        stems.append({
            "depth": d,
            "audio": audio,
            "std_temp": entry["envelope_raw_std"],
        })

    if not stems:
        raise SystemExit("no stems found")

    # Compute per-stem gain
    if args.mode == "equal":
        gains = {s["depth"]: 1.0 for s in stems}
    else:  # physical
        # Map std_temp logarithmically to [gain_floor_db, gain_ceil_db]
        stds = np.array([s["std_temp"] for s in stems], dtype=np.float64)
        # Add tiny epsilon to avoid log(0)
        log_stds = np.log10(stds + 0.01)
        lo, hi = log_stds.min(), log_stds.max()
        rng = hi - lo if hi > lo else 1.0
        norm = (log_stds - lo) / rng  # [0, 1]
        db = args.gain_floor_db + (args.gain_ceil_db - args.gain_floor_db) * norm
        amps = 10 ** (db / 20)
        gains = {s["depth"]: float(a) for s, a in zip(stems, amps)}

    print(f"[mix ] per-stem gains:")
    for s in stems:
        g = gains[s["depth"]]
        print(f"   depth {s['depth']:>4}cm  std_temp={s['std_temp']:5.2f}°C  gain={20*np.log10(g):+6.1f}dB  ({g:.3f})")

    # Sum
    out = np.zeros_like(stems[0]["audio"])
    for s in stems:
        out = out + s["audio"] * gains[s["depth"]]

    # Peak-normalize to -1 dB headroom
    peak = float(np.max(np.abs(out)))
    target = 10 ** (-1.0 / 20)  # -1 dB
    if peak > 0:
        out = out * (target / peak)
        print(f"[norm] peak before {peak:.3f}, scaled to {target:.3f} (-1 dB)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, out, sr_check)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
