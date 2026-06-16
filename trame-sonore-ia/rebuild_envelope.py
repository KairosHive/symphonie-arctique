"""
Re-apply envelope effects from saved raw segments without re-running SAO.

Use this after generate_3min.py has saved its raw (no-envelope) segments to
out_dir/segments_raw/ — typically when you want to iterate envelope params
without spending another 17 min on SAO.

Reads:
    out_dir/segments_raw/depth_NNNNcm_seg{1..4}.wav

Writes:
    out_dir/segments/depth_NNNNcm_seg{1..4}.wav   (with envelope)
    out_dir/stem_NNNNcm_envmod.wav                (3-min stitched stream)
    out_dir/manifest.json                         (rebuilt)

Usage:
    python rebuild_envelope.py --src outputs/stems_3min_v5 \\
        --envelope-style modulation --mod_depth 0.5 --target_rms 0.15 \\
        --gamma 1.5 --lpf_low_hz 2000 --lpf_high_hz 16000

For the buggy first-run v5 stems, this script with the (now-fixed) normalize_rms
will rebuild correct stems without per-stem peak-clobbering.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import numpy as np
import soundfile as sf

from envelope_fx import (
    apply_modulation_envelope,
    dynamic_hpf,
    dynamic_lpf,
    dynamic_stereo_width,
    dynamic_tremolo,
    gamma_envelope,
    normalize_rms,
)
from progression import N_SEGMENTS, segment_day_window
from soil_data import envelope_for_depth, extract_depth_series, load_csv


SEG_RE = re.compile(r"depth_(\d+)cm_seg(\d+)\.wav$")


def stitch_with_crossfade(segments, sr, crossfade_s=3.0):
    if len(segments) == 1:
        return segments[0].copy()
    cf_n = int(sr * crossfade_s)
    fade_out = np.cos(np.linspace(0, np.pi / 2, cf_n)).astype(np.float32)[:, None]
    fade_in = np.sin(np.linspace(0, np.pi / 2, cf_n)).astype(np.float32)[:, None]
    out = segments[0].copy().astype(np.float32)
    for seg in segments[1:]:
        seg = seg.astype(np.float32).copy()
        out[-cf_n:] *= fade_out
        seg[:cf_n] *= fade_in
        out = np.concatenate(
            [out[:-cf_n], out[-cf_n:] + seg[:cf_n], seg[cf_n:]], axis=0
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Stems dir containing segments_raw/")
    p.add_argument(
        "--envelope-style", dest="envelope_style", choices=["dramatic", "modulation"],
        default="modulation",
    )
    p.add_argument("--mod_depth", type=float, default=0.5)
    p.add_argument("--target_rms", type=float, default=0.15)
    p.add_argument("--floor", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--smooth_s", type=float, default=0.5)
    p.add_argument("--lpf_low_hz", type=float, default=2000.0)
    p.add_argument("--lpf_high_hz", type=float, default=16000.0)
    p.add_argument("--no_lpf", action="store_true")
    p.add_argument("--crossfade_s", type=float, default=3.0)
    p.add_argument(
        "--no_per_stem_peak_norm", action="store_true",
        help="Don't peak-normalize each stitched stem to 0.999 — preserves "
             "RMS-equal layering at the cost of possibly hot single-stem playback. "
             "Final mix should peak-normalize.",
    )
    # ---- v10.3 per-depth effect routing ----
    p.add_argument(
        "--effect_map", default="",
        help='Per-depth effect routing. Format: "200=width,300=lpf,400=tremolo,500=width,'
             '700=none,900=hpf,1100=none". Effects: lpf, hpf, tremolo, width, none. '
             'When set, overrides the global --lpf/--hpf/--tremolo/--width flags '
             'and applies only the specified effect per depth (amplitude always on).',
    )
    # ---- v10.2 extra envelope effects (global flags, used if effect_map empty) ----
    p.add_argument(
        "--hpf", action="store_true",
        help="Add envelope-driven HIGH-pass: cold=heavy bass cut, warm=full bass",
    )
    p.add_argument("--hpf_low_hz", type=float, default=20.0,
                   help="HPF cutoff when env=1 (warm); default 20 Hz (no cut)")
    p.add_argument("--hpf_high_hz", type=float, default=400.0,
                   help="HPF cutoff when env=0 (cold); default 400 Hz")
    p.add_argument(
        "--width", action="store_true",
        help="Add envelope-driven stereo WIDTH: cold=narrower/mono, warm=wide",
    )
    p.add_argument("--width_min", type=float, default=0.2,
                   help="Stereo width when env=0 (cold); 0=mono, 1=full stereo")
    p.add_argument("--width_max", type=float, default=1.0,
                   help="Stereo width when env=1 (warm)")
    p.add_argument(
        "--tremolo", action="store_true",
        help="Add envelope-driven TREMOLO DEPTH: cold=no tremolo, warm=wobble",
    )
    p.add_argument("--tremolo_rate", type=float, default=5.0,
                   help="Tremolo LFO rate in Hz")
    p.add_argument("--tremolo_max_depth", type=float, default=0.4,
                   help="Tremolo depth at env=1 (warm)")
    args = p.parse_args()

    src = Path(args.src)
    raw_dir = src / "segments_raw"
    seg_dir = src / "segments"
    seg_dir.mkdir(exist_ok=True)
    if not raw_dir.exists():
        raise SystemExit(f"raw segments dir not found: {raw_dir}")

    # Discover (depth, segment) -> path
    by_depth: dict[int, dict[int, Path]] = {}
    for wav in raw_dir.glob("depth_*cm_seg*.wav"):
        m = SEG_RE.search(wav.name)
        if not m:
            continue
        d = int(m.group(1))
        s = int(m.group(2))
        by_depth.setdefault(d, {})[s] = wav

    if not by_depth:
        raise SystemExit(f"no raw segments found in {raw_dir}")

    df = load_csv()
    n_total = len(df)

    # Parse per-depth effect map
    effect_map: dict[int, str] = {}
    if args.effect_map:
        for tok in args.effect_map.split(","):
            if "=" not in tok:
                continue
            k, v = tok.split("=")
            effect_map[int(k.strip())] = v.strip().lower()
    use_effect_map = bool(effect_map)

    print(f"[cfg ] envelope_style={args.envelope_style}  mod_depth={args.mod_depth}  "
          f"rms={args.target_rms}  gamma={args.gamma}  per_stem_peak_norm="
          f"{'OFF' if args.no_per_stem_peak_norm else 'ON'}")
    print()

    manifest = []
    for d in sorted(by_depth.keys()):
        segs = by_depth[d]
        if len(segs) < N_SEGMENTS:
            print(f"[warn] depth {d}: only {len(segs)} of {N_SEGMENTS} segments found, skipping")
            continue

        depth_segments = []
        depth_meta = []
        for seg_i in range(1, N_SEGMENTS + 1):
            wav_path = segs[seg_i]
            audio, sr = sf.read(str(wav_path), dtype="float32")

            start, length = segment_day_window(n_total, seg_i - 1)
            window = extract_depth_series(df, d)[start : start + length]
            valid = window[~np.isnan(window)]
            std = float(np.std(valid)) if len(valid) > 0 else 0.0

            if args.envelope_style == "modulation":
                audio_norm = normalize_rms(audio, target_rms=args.target_rms)
                env_obj = envelope_for_depth(
                    df, d,
                    target_samples=audio.shape[0],
                    start_day=start, n_days=length,
                    floor=0.0,
                    smooth_kernel=int(sr * args.smooth_s),
                )
                env_vals = env_obj.values
                # Decide which effects to apply for this depth
                if use_effect_map:
                    eff = effect_map.get(d, "none")
                    do_lpf = (eff == "lpf")
                    do_hpf = (eff == "hpf")
                    do_tremolo = (eff == "tremolo")
                    do_width = (eff == "width")
                else:
                    do_lpf = not args.no_lpf
                    do_hpf = args.hpf
                    do_tremolo = args.tremolo
                    do_width = args.width
                    eff = "+".join(x for x, b in [("lpf", do_lpf), ("hpf", do_hpf),
                                                   ("tremolo", do_tremolo), ("width", do_width)] if b)

                # Amplitude modulation is always on; LPF handled by
                # apply_modulation_envelope so we toggle it through `apply_lpf`.
                modulated = apply_modulation_envelope(
                    audio_norm, env_vals, sr,
                    mod_depth=args.mod_depth,
                    gamma=args.gamma,
                    apply_lpf=do_lpf,
                    lpf_low_hz=args.lpf_low_hz,
                    lpf_high_hz=args.lpf_high_hz,
                )
                if do_hpf:
                    modulated = dynamic_hpf(
                        modulated, env_vals, sr,
                        cutoff_low_hz=args.hpf_low_hz,
                        cutoff_high_hz=args.hpf_high_hz,
                    )
                if do_tremolo:
                    modulated = dynamic_tremolo(
                        modulated, env_vals, sr,
                        rate_hz=args.tremolo_rate,
                        max_depth=args.tremolo_max_depth,
                    )
                if do_width:
                    modulated = dynamic_stereo_width(
                        modulated, env_vals,
                        min_width=args.width_min,
                        max_width=args.width_max,
                    )
                if seg_i == 1:  # log once per depth
                    print(f"  d={d:>4}cm  effect={eff or 'amplitude-only'}")
            else:  # dramatic
                env_obj = envelope_for_depth(
                    df, d,
                    target_samples=audio.shape[0],
                    start_day=start, n_days=length,
                    floor=args.floor,
                    smooth_kernel=int(sr * args.smooth_s),
                )
                env_g = gamma_envelope(env_obj.values, floor=args.floor, gamma=args.gamma)
                modulated = audio * env_g[:, None]
                if not args.no_lpf:
                    modulated = dynamic_lpf(
                        modulated, env_g, sr,
                        cutoff_low_hz=args.lpf_low_hz,
                        cutoff_high_hz=args.lpf_high_hz,
                    )

            seg_out = seg_dir / f"depth_{d:>04d}cm_seg{seg_i}.wav"
            sf.write(seg_out, modulated, sr)
            depth_segments.append(modulated)
            depth_meta.append({
                "segment": seg_i,
                "day_start": start,
                "n_days": length,
                "raw_std": std,
                "wav": str(seg_out),
            })
            print(f"  d={d:>4}cm  S{seg_i}  raw_peak={float(np.max(np.abs(audio))):.3f}  "
                  f"out_peak={float(np.max(np.abs(modulated))):.3f}  std={std:.2f}°C")

        # Stitch
        stream = stitch_with_crossfade(depth_segments, sr, crossfade_s=args.crossfade_s)
        peak = float(np.max(np.abs(stream)))
        if not args.no_per_stem_peak_norm and peak > 0.999:
            stream = stream * (0.999 / peak)
            print(f"  [stitch] peak {peak:.3f} -> 0.999")
        else:
            print(f"  [stitch] peak {peak:.3f} (preserved)")
        stream_path = src / f"stem_{d:>04d}cm_envmod.wav"
        sf.write(stream_path, stream, sr)

        avg_std = float(np.mean([m["raw_std"] for m in depth_meta]))
        manifest.append({
            "depth_cm": d,
            "envelope_raw_std": avg_std,
            "envmod_wav": str(stream_path),
            "raw_wav": str(stream_path),
            "duration_s": stream.shape[0] / sr,
            "segments": depth_meta,
        })
        print(f"  [save] {stream_path}\n")

    manifest_path = src / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[save] {manifest_path}  ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
