"""
Rebuild a stems_3min/ manifest from the wav files on disk + soil data.

Use after a partial regen (e.g. python generate_3min.py --depths 400) clobbered
the manifest.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from soil_data import extract_depth_series, load_csv

STEM_RE = re.compile(r"stem_(\d+)cm_envmod\.wav$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="outputs/stems_3min")
    args = p.parse_args()

    src = Path(args.src)
    df = load_csv()

    manifest = []
    for wav in sorted(src.glob("stem_*cm_envmod.wav")):
        m = STEM_RE.search(wav.name)
        if not m:
            continue
        d = int(m.group(1))
        # Get the depth's full-history std as a stand-in for the per-segment-averaged std
        s = extract_depth_series(df, d)
        # Compute mean of per-segment stds (4 segments evenly split) — same as
        # what the original manifest stored
        n = len(s)
        seg_stds = []
        for i in range(4):
            start = i * (n // 4)
            length = n // 4 if i < 3 else n - start
            window = s[start : start + length]
            v = window[~np.isnan(window)]
            if len(v) > 0:
                seg_stds.append(float(np.std(v)))
        avg_std = float(np.mean(seg_stds)) if seg_stds else 0.0

        info = sf.info(str(wav))
        manifest.append({
            "depth_cm": d,
            "envelope_raw_std": avg_std,
            "envmod_wav": str(wav),
            "raw_wav": str(wav),  # not separately available
            "duration_s": info.duration,
        })
        print(f"  d={d}cm  std={avg_std:.2f}°C  dur={info.duration:.1f}s  {wav.name}")

    out = src / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[save] {out}  ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
