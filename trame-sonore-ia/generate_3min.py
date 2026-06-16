"""
Generate a 3-minute (~179 s) per-depth stream by stitching 4 × 47 s segments,
each with its own evolving prompt and its own slice of the 27-year temperature
record. See progression.py for narrative + window definitions.

Per depth:
  segment k → prompt_k + envelope from year-window k → audio_k (47 s)
            → equal-power crossfade with neighbours (3 s)
  → final stream stem_NNNNcm_3min.wav  (≈179 s)

After all 7 depths:  outputs/stems_3min/manifest.json + 7 wav streams.
Use mix.py on outputs/stems_3min/ to get the final 3 min mix.

Usage:
    python generate_3min.py
    python generate_3min.py --depths 200,500,1100 --steps 200
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

# Ensure non-ASCII (em-dashes, °C, etc.) print cleanly on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import numpy as np
import soundfile as sf
import torch
from diffusers import StableAudioPipeline

from envelope_fx import (
    apply_modulation_envelope,
    dynamic_lpf,
    gamma_envelope,
    normalize_rms,
)
from progression import (
    DEPTH_PROGRESSION,
    N_SEGMENTS,
    SEGMENT_NARRATIVE,
    build_segment_prompt,
    segment_day_window,
)
# Optional legacy progressions (kept locally for reference; not part of the
# delivered V1/V2 pipelines so they may be absent in a fresh clone).
try:
    from progression_v5 import (
        DEPTH_PROGRESSION_V5,
        build_segment_prompt_v5,
    )
except ImportError:
    DEPTH_PROGRESSION_V5 = None
    build_segment_prompt_v5 = None
try:
    from progression_v7 import (
        DEPTH_PROGRESSION_V7,
        build_segment_prompt_v7,
    )
except ImportError:
    DEPTH_PROGRESSION_V7 = None
    build_segment_prompt_v7 = None
from progression_v8 import (
    DEPTH_PROGRESSION_V8,
    build_segment_prompt_v8,
)
from progression_v15 import (
    DEPTH_PROGRESSION_V15,
    build_segment_prompt_v15,
)
from progression_v15_2 import (
    DEPTH_PROGRESSION_V15_2,
    build_segment_prompt_v15_2,
)
from synthesize_chord import synthesize_chord_for_depth
from audio_prompted_call import audio_prompted_generate
from soil_data import (
    envelope_for_depth,
    extract_depth_series,
    load_csv,
)

ALL_DEPTHS = sorted(DEPTH_PROGRESSION.keys())
HARMONIES_PATH = Path("outputs/depth_harmonies.json")


def stitch_with_crossfade(segments: list[np.ndarray], sr: int, crossfade_s: float = 3.0) -> np.ndarray:
    """Equal-power crossfade between consecutive (samples, channels) segments."""
    if not segments:
        raise ValueError("no segments")
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
            [out[:-cf_n], out[-cf_n:] + seg[:cf_n], seg[cf_n:]],
            axis=0,
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depths", default=",".join(map(str, ALL_DEPTHS)))
    p.add_argument("--seconds", type=float, default=47.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--cfg", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42, help="Base seed; depth+seg added")
    p.add_argument("--smooth_s", type=float, default=0.5)
    p.add_argument("--floor", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--lpf_low_hz", type=float, default=350.0)
    p.add_argument("--lpf_high_hz", type=float, default=16000.0)
    p.add_argument("--no_lpf", action="store_true")
    p.add_argument("--crossfade_s", type=float, default=3.0)
    p.add_argument("--out_dir", default="outputs/stems_3min")
    p.add_argument(
        "--progression", choices=["v4", "v5", "v7", "v8", "v15", "v15_2"], default="v4",
        help="Which depth-progression module to use",
    )
    p.add_argument(
        "--segments", default="",
        help='Subset of segment indices to (re)generate (1-based, comma-separated). '
             'Empty = all 4 segments. Example: "4" to regen only S4 for each depth.',
    )
    p.add_argument(
        "--envelope-style", dest="envelope_style", choices=["dramatic", "modulation"],
        default="dramatic",
        help="dramatic = v4 (gamma + LPF, can go to silence). "
             "modulation = v5 (RMS-equalized stems + ±mod_depth around baseline)",
    )
    p.add_argument(
        "--mod_depth", type=float, default=0.5,
        help="(modulation only) modulation depth in [0,1]. 0.5 = ±50% around baseline.",
    )
    p.add_argument(
        "--target_rms", type=float, default=0.15,
        help="(modulation only) RMS target for stem-level loudness equalization.",
    )
    p.add_argument(
        "--no_pitch_hint", action="store_true",
        help="Omit the 'in F# minor' style pitch-class hint from prompts.",
    )
    p.add_argument(
        "--audio_prompt", action="store_true",
        help="Seed each depth's diffusion process with the synthesized EMD chord "
             "(additive sines). Brings actual microtonal pitch content into the audio.",
    )
    p.add_argument(
        "--audio_prompt_peak", type=float, default=0.30,
        help="Peak level of the synthesized chord seed (default 0.30 ~ -10 dBFS).",
    )
    p.add_argument(
        "--init_strength", type=float, default=0.7,
        help="Init-audio strength via custom img2img-style call. "
             "1.0 = stock diffusers behavior (init ignored). "
             "0.7 = moderate. 0.4 = strong. 0.2 = very strong.",
    )
    p.add_argument(
        "--chord_max_freq", type=float, default=None,
        help="Drop partials above this freq (Hz) from the chord seed. "
             "Useful to remove strident high content. Try 2000-3000.",
    )
    p.add_argument(
        "--chord_weight_exp", type=float, default=0.6,
        help="Amplitude falloff exponent for the chord seed partials. "
             "Higher = lowest partials dominate more (default 0.6, try 1.2-1.8 to tame).",
    )
    args = p.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
    out_dir = Path(args.out_dir)
    seg_dir = out_dir / "segments"
    raw_seg_dir = out_dir / "segments_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    raw_seg_dir.mkdir(parents=True, exist_ok=True)

    # Pick progression
    if args.progression == "v15_2":
        active_progression = DEPTH_PROGRESSION_V15_2
        prompt_builder = build_segment_prompt_v15_2
    elif args.progression == "v15":
        active_progression = DEPTH_PROGRESSION_V15
        prompt_builder = build_segment_prompt_v15
    elif args.progression == "v8":
        active_progression = DEPTH_PROGRESSION_V8
        prompt_builder = build_segment_prompt_v8
    elif args.progression == "v7":
        active_progression = DEPTH_PROGRESSION_V7
        prompt_builder = build_segment_prompt_v7
    elif args.progression == "v5":
        active_progression = DEPTH_PROGRESSION_V5
        prompt_builder = build_segment_prompt_v5
    else:
        active_progression = DEPTH_PROGRESSION
        prompt_builder = build_segment_prompt
    print(f"[cfg ] progression={args.progression}  envelope_style={args.envelope_style}")

    harmonies = json.loads(HARMONIES_PATH.read_text(encoding="utf-8"))
    df = load_csv()
    n_total = len(df)

    print(f"[load] Stable Audio Open 1.0 (fp16, cuda)")
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16
    ).to("cuda")
    sr = pipe.vae.sampling_rate
    n_per_seg = int(sr * args.seconds)
    target_total = N_SEGMENTS * args.seconds - (N_SEGMENTS - 1) * args.crossfade_s
    print(f"[load] sr={sr}Hz, n_per_seg={n_per_seg}, target_total~{target_total:.1f}s")
    print()

    # Parse segments-subset CLI arg (1-based -> 0-based indices)
    if args.segments:
        segments_to_gen = sorted({int(x) - 1 for x in args.segments.split(",") if x.strip()})
        for s in segments_to_gen:
            if s < 0 or s >= N_SEGMENTS:
                raise SystemExit(f"--segments contains invalid index {s+1} (must be 1..{N_SEGMENTS})")
        print(f"[cfg ] regenerating only segments: {[s+1 for s in segments_to_gen]}")
    else:
        segments_to_gen = list(range(N_SEGMENTS))

    # Print narrative windows
    for i, narr in enumerate(SEGMENT_NARRATIVE):
        s, n = segment_day_window(n_total, i)
        y_start = df["Year"].iloc[s]
        y_end = df["Year"].iloc[min(s + n - 1, n_total - 1)]
        print(f"  segment {i+1}: days [{s}..{s+n-1}]  ~{y_start}–{y_end}  ({narr})")
    print()

    manifest = []
    for d in depths:
        if d not in active_progression:
            print(f"[skip] depth {d} not in active progression ({args.progression})")
            continue
        if str(d) not in harmonies:
            print(f"[skip] depth {d} not in harmonies json")
            continue

        top_pc = harmonies[str(d)]["top_pitch_classes"][0][0]
        raw = extract_depth_series(df, d)
        depth_segments = []
        depth_meta = []

        # If audio-prompted: synthesize the depth's chord once per depth
        # (same harmonic seed for all 4 segments). Reshape to (1, 2, n) on GPU.
        audio_prompt_tensor = None
        if args.audio_prompt:
            seed_audio_np = synthesize_chord_for_depth(
                d, duration_s=args.seconds, sr=sr,
                target_peak=args.audio_prompt_peak,
                max_freq_hz=args.chord_max_freq,
                weight_exponent=args.chord_weight_exp,
            )
            # synthesize returns (n_samples, 2); pipeline wants (batch, channels, samples)
            audio_prompt_tensor = (
                torch.from_numpy(seed_audio_np.T)  # (2, n)
                .unsqueeze(0)                       # (1, 2, n)
                .to("cuda", dtype=torch.float16)
            )
            print(f"  [seed] chord init-audio: {seed_audio_np.shape[0]} samples @ {sr}Hz, "
                  f"peak={args.audio_prompt_peak}")

        for seg_i in range(N_SEGMENTS):
            if seg_i not in segments_to_gen:
                # Read the existing raw segment from disk so stitching still works.
                existing_raw_path = raw_seg_dir / f"depth_{d:>04d}cm_seg{seg_i+1}.wav"
                if not existing_raw_path.exists():
                    raise SystemExit(
                        f"--segments excluded S{seg_i+1} for depth {d} cm but "
                        f"no existing raw file at {existing_raw_path}. "
                        f"Pre-populate by copying from a complete run."
                    )
                existing, _sr = sf.read(str(existing_raw_path), dtype="float32")
                print(f"=== Depth {d} cm  S{seg_i+1}  [reusing existing raw segment]")
                depth_segments_placeholder = existing  # noqa: F841
                # We still need to apply envelope and append to the stitch list.
                start, length = segment_day_window(n_total, seg_i)
                window = raw[start : start + length]
                valid = window[~np.isnan(window)]
                std = float(np.std(valid)) if len(valid) > 0 else 0.0
                # Build modulated version exactly as we would for a fresh segment.
                if args.envelope_style == "modulation":
                    audio_norm = normalize_rms(existing, target_rms=args.target_rms)
                    env_obj = envelope_for_depth(
                        df, d, target_samples=existing.shape[0],
                        start_day=start, n_days=length, floor=0.0,
                        smooth_kernel=int(sr * args.smooth_s),
                    )
                    modulated = apply_modulation_envelope(
                        audio_norm, env_obj.values, sr,
                        mod_depth=args.mod_depth, gamma=args.gamma,
                        apply_lpf=not args.no_lpf,
                        lpf_low_hz=args.lpf_low_hz if args.lpf_low_hz >= 1000 else 2000.0,
                        lpf_high_hz=args.lpf_high_hz,
                    )
                else:
                    env_obj = envelope_for_depth(
                        df, d, target_samples=existing.shape[0],
                        start_day=start, n_days=length, floor=args.floor,
                        smooth_kernel=int(sr * args.smooth_s),
                    )
                    env_g = gamma_envelope(env_obj.values, floor=args.floor, gamma=args.gamma)
                    modulated = existing * env_g[:, None]
                    if not args.no_lpf:
                        modulated = dynamic_lpf(
                            modulated, env_g, sr,
                            cutoff_low_hz=args.lpf_low_hz,
                            cutoff_high_hz=args.lpf_high_hz,
                        )
                seg_path = seg_dir / f"depth_{d:>04d}cm_seg{seg_i+1}.wav"
                sf.write(seg_path, modulated, sr)
                depth_segments.append(modulated)
                depth_meta.append({
                    "segment": seg_i + 1,
                    "narrative": SEGMENT_NARRATIVE[seg_i],
                    "day_start": start, "n_days": length,
                    "raw_min": env_obj.raw_min, "raw_max": env_obj.raw_max,
                    "raw_std": std, "seed": None,
                    "prompt": "(reused from existing raw segment)",
                    "wav": str(seg_path),
                })
                continue

            start, length = segment_day_window(n_total, seg_i)
            window = raw[start : start + length]
            valid = window[~np.isnan(window)]
            std = float(np.std(valid)) if len(valid) > 0 else 0.0

            dp = prompt_builder(d, seg_i, top_pc, std, include_pitch_hint=not args.no_pitch_hint)
            seed = args.seed + d + seg_i * 10
            print(f"=== Depth {d} cm  S{seg_i+1}  seed={seed}")
            print(f"  prompt: {dp.prompt}")

            t0 = time.time()
            generator = torch.Generator("cuda").manual_seed(seed)
            if audio_prompt_tensor is not None:
                # Custom img2img-style call with strength control (the stock pipeline
                # ignores init audio because of huge noise sigma — see audio_prompted_call.py).
                audio = audio_prompted_generate(
                    pipe, dp.prompt,
                    init_audio=audio_prompt_tensor,
                    init_sample_rate=sr,
                    strength=args.init_strength,
                    num_inference_steps=args.steps,
                    audio_end_in_s=args.seconds,
                    guidance_scale=args.cfg,
                    negative_prompt=dp.negative,
                    generator=generator,
                )
                # audio shape: (1, channels, samples)
                # Mimic the diffusers pipeline output shape used below
                class _Result: pass
                result = _Result()
                result.audios = audio
            else:
                result = pipe(
                    dp.prompt,
                    negative_prompt=dp.negative,
                    num_inference_steps=args.steps,
                    audio_end_in_s=args.seconds,
                    num_waveforms_per_prompt=1,
                    guidance_scale=args.cfg,
                    generator=generator,
                )
            audio = result.audios[0].T.float().cpu().numpy()  # (n, channels)
            print(f"  [gen ] {time.time()-t0:.1f}s")

            # Save the raw (no envelope) segment so future envelope-tweaks don't
            # require re-running SAO.
            raw_seg_path = raw_seg_dir / f"depth_{d:>04d}cm_seg{seg_i+1}.wav"
            sf.write(raw_seg_path, audio, sr)

            if args.envelope_style == "modulation":
                # Gentle envelope around RMS-equalized baseline
                audio_norm = normalize_rms(audio, target_rms=args.target_rms)
                # Build a [0,1] envelope from soil window
                env_obj = envelope_for_depth(
                    df, d,
                    target_samples=audio.shape[0],
                    start_day=start,
                    n_days=length,
                    floor=0.0,  # apply_modulation_envelope wants [0,1]
                    smooth_kernel=int(sr * args.smooth_s),
                )
                modulated = apply_modulation_envelope(
                    audio_norm, env_obj.values, sr,
                    mod_depth=args.mod_depth,
                    gamma=args.gamma,
                    apply_lpf=not args.no_lpf,
                    lpf_low_hz=args.lpf_low_hz if args.lpf_low_hz >= 1000 else 2000.0,
                    lpf_high_hz=args.lpf_high_hz,
                )
                print(f"  [env ] modulation  rms={args.target_rms}  mod_depth={args.mod_depth}  "
                      f"gamma={args.gamma}  lpf={'on' if not args.no_lpf else 'off'}")
            else:
                # Original v4 dramatic envelope (gamma + can-go-to-floor + LPF)
                env_obj = envelope_for_depth(
                    df, d,
                    target_samples=audio.shape[0],
                    start_day=start,
                    n_days=length,
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
                    print(f"  [lpf ] {args.lpf_low_hz:.0f}..{args.lpf_high_hz:.0f} Hz")

            # Save individual segment for inspection
            seg_path = seg_dir / f"depth_{d:>04d}cm_seg{seg_i+1}.wav"
            sf.write(seg_path, modulated, sr)
            depth_segments.append(modulated)
            depth_meta.append({
                "segment": seg_i + 1,
                "narrative": SEGMENT_NARRATIVE[seg_i],
                "day_start": start,
                "n_days": length,
                "raw_min": env_obj.raw_min,
                "raw_max": env_obj.raw_max,
                "raw_std": std,
                "seed": seed,
                "prompt": dp.prompt,
                "wav": str(seg_path),
            })
            print()

        # Stitch all 4 segments
        stream = stitch_with_crossfade(depth_segments, sr, crossfade_s=args.crossfade_s)
        peak = float(np.max(np.abs(stream)))
        if peak > 0.999:
            stream = stream * (0.999 / peak)
        stream_path = out_dir / f"stem_{d:>04d}cm_envmod.wav"
        sf.write(stream_path, stream, sr)
        dur = stream.shape[0] / sr
        print(f"  [stitch+save] {stream_path}  ({dur:.1f}s, peak {peak:.3f})\n")

        manifest.append({
            "depth_cm": d,
            "envelope_raw_std": float(np.mean([m["raw_std"] for m in depth_meta])),
            "envmod_wav": str(stream_path),
            "raw_wav": str(stream_path),  # we don't separately save a raw 3-min stream
            "duration_s": dur,
            "segments": depth_meta,
        })

    # Merge with any existing manifest so that single-depth regens don't clobber
    # the rest.
    manifest_path = out_dir / "manifest.json"
    by_depth: dict[int, dict] = {}
    if manifest_path.exists():
        try:
            for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
                by_depth[entry["depth_cm"]] = entry
        except Exception:
            pass
    for entry in manifest:
        by_depth[entry["depth_cm"]] = entry
    merged = sorted(by_depth.values(), key=lambda e: e["depth_cm"])
    manifest_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[save] {manifest_path}  ({len(merged)} entries, {len(manifest)} this run)")
    print(f"[done] {len(manifest)} streams generated")


if __name__ == "__main__":
    main()
