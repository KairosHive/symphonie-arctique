# Reproduce the delivered audio versions

The methods report (`outputs/Soil_Audio_Methods_Report_v4.pdf`) describes two
delivered versions:

| Report | Audio | Final mix file |
|---|---|---|
| **Version 1** — Envelope-modulated sonification | **v10.3** | `outputs/mix_3min_v10_3_balanced.wav` |
| **Version 2** — Microtonal-anchored sonification | **v15** | `outputs/mix_3min_v15_balanced.wav` |

Each version is fully reproducible from this repository with the exact
commands below. All randomness is seeded; running the same command twice
gives bit-identical output.

---

## Pre-requisites

A conda env with diffusers, scipy, matplotlib, soundfile, pandas, mido.
On this machine:

```
conda env: comfyui
python: 3.12 / fp16 / CUDA 13 / RTX 4090
HF login: required (Stable Audio Open is gated)
```

```bash
hf auth login   # paste a HuggingFace token with access to stable-audio-open-1.0
```

---

## Step 0 — One-time data preparation (shared by both versions)

```bash
# Parse the EMD microtonal MIDI into per-depth harmonics JSON.
python parse_midi.py --notes-per-depth "11,11,11,8,8,9,9"
# -> outputs/depth_harmonies.json
```

---

## Version 1 — `v10.3` (envelope-modulated, no audio prompt)

### Stage A. Generate the 28 raw SAO segments

(only needed once; raw segments are cached afterwards)

```bash
python generate_3min.py \
    --progression v8 \
    --envelope-style modulation \
    --mod_depth 0.7 --target_rms 0.15 --gamma 2.0 \
    --lpf_low_hz 800 --lpf_high_hz 16000 \
    --no_pitch_hint \
    --out_dir outputs/stems_3min_v10
```

This was followed by a targeted regen of depth 700 cm with the
present-percussion progression already baked into `progression_v8.py` (the
final v10 7-stem set lives in `outputs/stems_3min_v10/segments_raw/`).

### Stage B. Apply per-depth envelope effects (Report Version 1's signature)

```bash
mkdir -p outputs/stems_3min_v10_3/segments_raw
cp outputs/stems_3min_v10/segments_raw/*.wav \
   outputs/stems_3min_v10_3/segments_raw/

python rebuild_envelope.py \
    --src outputs/stems_3min_v10_3 \
    --envelope-style modulation \
    --mod_depth 0.7 --target_rms 0.15 --gamma 2.0 \
    --lpf_low_hz 800 --lpf_high_hz 16000 \
    --hpf_low_hz 25 --hpf_high_hz 300 \
    --width_min 0.3 --width_max 1.0 \
    --tremolo_rate 4.5 --tremolo_max_depth 0.25 \
    --effect_map "200=width,300=lpf,400=tremolo,500=width,700=none,900=hpf,1100=none"
```

### Stage C. Final mix

```bash
python mix.py \
    --src outputs/stems_3min_v10_3 \
    --mode equal --variant envmod \
    --equalize_rms 0.12 \
    --per_stem_gain_db "200=-3,300=-2,700=+4,900=+5,1100=+6" \
    --out outputs/mix_3min_v10_3_balanced.wav
```

**→ `outputs/mix_3min_v10_3_balanced.wav` is the delivered Report Version 1.**

---

## Version 2 — `v15` (envelope + microtonal chord seed + melodic prompts)

### Stage A. Generate the 28 raw SAO segments WITH chord-seed audio prompt

```bash
python generate_3min.py \
    --progression v15 \
    --envelope-style modulation \
    --mod_depth 0.7 --target_rms 0.15 --gamma 2.0 \
    --lpf_low_hz 800 --lpf_high_hz 16000 \
    --no_pitch_hint \
    --audio_prompt \
    --audio_prompt_peak 0.15 \
    --init_strength 0.85 \
    --chord_max_freq 2000 \
    --chord_weight_exp 1.5 \
    --seed 42 \
    --out_dir outputs/stems_3min_v15
```

Key differences vs Version 1's Stage A:

- `--progression v15` — the melodic prompts in `progression_v15.py`
- `--audio_prompt` — turns on the microtonal chord seed
- `--audio_prompt_peak 0.15` — quiet chord seed (~-16 dBFS)
- `--init_strength 0.85` — gentle: chord influence over last 30 of 200 steps
- `--chord_max_freq 2000` — drop strident high partials from the seed
- `--chord_weight_exp 1.5` — heavy weighting toward low partials

### Stage B. Apply the same per-depth envelope effects as Version 1

```bash
python rebuild_envelope.py \
    --src outputs/stems_3min_v15 \
    --envelope-style modulation \
    --mod_depth 0.7 --target_rms 0.15 --gamma 2.0 \
    --lpf_low_hz 800 --lpf_high_hz 16000 \
    --hpf_low_hz 25 --hpf_high_hz 300 \
    --width_min 0.3 --width_max 1.0 \
    --tremolo_rate 4.5 --tremolo_max_depth 0.25 \
    --effect_map "200=width,300=lpf,400=tremolo,500=width,700=none,900=hpf,1100=none"
```

### Stage C. Final mix

```bash
python mix.py \
    --src outputs/stems_3min_v15 \
    --mode equal --variant envmod \
    --equalize_rms 0.12 \
    --per_stem_gain_db "200=-3,300=-2,700=+4,900=+5,1100=+6" \
    --out outputs/mix_3min_v15_balanced.wav
```

**→ `outputs/mix_3min_v15_balanced.wav` is the delivered Report Version 2.**

---

## File-by-file reference

| File | Used by | Purpose |
|---|---|---|
| `data/temperature_curves_by_depth_Tasiujaq.csv` | both | Source temperature data (1.1 MB, tracked in repo) |
| `data/depth_chords_microtonal.mid` | V2 only | EMD-derived microtonal chord per depth (897 B, tracked in repo) |
| `soil_data.py` | both | CSV loader + envelope extraction |
| `parse_midi.py` | V2 only | Parses MIDI into `outputs/depth_harmonies.json` |
| `synthesize_chord.py` | V2 only | Additive-sine chord synthesis from harmonies JSON |
| `audio_prompted_call.py` | V2 only | img2img-style call with `strength` control |
| `envelope_fx.py` | both | Modulation envelope, dynamic LPF/HPF, tremolo, stereo width |
| `progression.py` (v4 prompts) | (legacy) | Earlier prompt set |
| `progression_v5.py` | (legacy) | Earlier prompt set |
| `progression_v7.py` | (legacy) | Earlier prompt set |
| `progression_v8.py` | V1 | Prompts + voice layers used for v10.3 generation |
| `progression_v15.py` | V2 | Melodic-prompt version used for v15 generation |
| `generate_3min.py` | both | Orchestrator: prompts × segments × audio prompt → 28 raw SAO clips, stitched |
| `rebuild_envelope.py` | both | Post-process: apply per-depth envelope effects to raw segments |
| `mix.py` | both | Final mix: load 7 streams, equalize RMS, per-stem manual gains, sum |

## Determinism guarantees

- `--seed 42` is used everywhere; per-depth seed = `seed + depth + 10*segment_idx`.
- `torch.Generator("cuda").manual_seed(seed)` controls the diffusion noise.
- Same seed + same prompt + same scheduler + same model weights ⇒ identical
  audio output. The raw segments in `outputs/stems_3min_*/segments_raw/`
  are bit-stable across re-runs.
- Envelope effects (Stage B) and mixing (Stage C) are deterministic
  numpy/scipy operations.

## What v10.3 and v15 share

- Exact same per-depth voice assignments (200 cm = soprano family, …, 1100 cm = permafrost)
- Exact same envelope shaping (`mod_depth=0.7`, `gamma=2.0`, LPF cutoff 800-16000 Hz)
- Exact same per-depth effect routing (width / LPF / tremolo / width / none / HPF / none)
- Exact same mix balance (`equalize_rms 0.12`, the same `per_stem_gain_db` string)

## What v10.3 and v15 differ in

| | v10.3 (Report V1) | v15 (Report V2) |
|---|---|---|
| Prompt set | `progression_v8.py` (ambient/sustained) | `progression_v15.py` (melodic/active) |
| Audio prompt to SAO | none | chord seed (additive sines of EMD freqs) |
| Diffusion path | stock denoising from full noise | img2img-style, denoising from step 170/200 with chord-anchored latent |

That is the entirety of the methodological difference. Everything else is
held constant so the two versions are directly comparable.
