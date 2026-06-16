"""
Combine stems from multiple seed-runs into one mix, picking which seed-set to
draw each depth from.

Example — depth 200 from main, depth 300 from alt, ...:

    python mix_combine.py \\
        --src_main outputs/stems_v3 \\
        --src_alt  outputs/stems_v3_alt \\
        --picks "200=alt,300=main,400=main,500=alt,700=main,900=alt,1100=main" \\
        --variant envmod --mode physical \\
        --out outputs/mix_v3_curated.wav
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_main", required=True, help="Primary seed-run stems dir")
    p.add_argument("--src_alt", required=True, help="Alternate seed-run stems dir")
    p.add_argument("--picks", required=True,
                   help='Comma-separated DEPTH=SET, e.g. "200=main,300=alt,..."')
    p.add_argument("--variant", choices=["raw", "envmod"], default="envmod")
    p.add_argument("--mode", choices=["equal", "physical"], default="physical")
    p.add_argument("--gain_floor_db", type=float, default=-18.0)
    p.add_argument("--gain_ceil_db", type=float, default=0.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # Parse picks
    picks = {}
    for token in args.picks.split(","):
        d, s = token.split("=")
        picks[int(d.strip())] = s.strip()
    print(f"[pick] {picks}")

    src_dirs = {"main": Path(args.src_main), "alt": Path(args.src_alt)}
    manifests = {
        name: json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        for name, d in src_dirs.items()
    }
    by_depth = {
        name: {entry["depth_cm"]: entry for entry in m}
        for name, m in manifests.items()
    }

    stems = []
    sr_check = None
    n_check = None
    for d, src in picks.items():
        entry = by_depth[src][d]
        path = entry[f"{args.variant}_wav"]
        audio, sr = sf.read(path, dtype="float32")
        if sr_check is None:
            sr_check, n_check = sr, audio.shape[0]
        if sr != sr_check or audio.shape[0] != n_check:
            raise ValueError(f"{path}: sr/length mismatch")
        stems.append({
            "depth": d,
            "audio": audio,
            "std_temp": entry["envelope_raw_std"],
            "src": src,
            "seed": entry["seed"],
        })

    if args.mode == "equal":
        gains = {s["depth"]: 1.0 for s in stems}
    else:
        stds = np.array([s["std_temp"] for s in stems])
        log_stds = np.log10(stds + 0.01)
        lo, hi = log_stds.min(), log_stds.max()
        rng = hi - lo if hi > lo else 1.0
        norm = (log_stds - lo) / rng
        db = args.gain_floor_db + (args.gain_ceil_db - args.gain_floor_db) * norm
        amps = 10 ** (db / 20)
        gains = {s["depth"]: float(a) for s, a in zip(stems, amps)}

    print(f"[mix ] mode={args.mode} variant={args.variant}")
    for s in stems:
        g = gains[s["depth"]]
        print(f"   depth {s['depth']:>4}cm  src={s['src']:>4}  seed={s['seed']:>4}  "
              f"std={s['std_temp']:5.2f}°C  gain={20*np.log10(g):+6.1f}dB")

    out = np.zeros_like(stems[0]["audio"])
    for s in stems:
        out += s["audio"] * gains[s["depth"]]

    peak = float(np.max(np.abs(out)))
    target = 10 ** (-1.0 / 20)
    if peak > 0:
        out *= target / peak
        print(f"[norm] peak {peak:.3f} -> {target:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, out, sr_check)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
