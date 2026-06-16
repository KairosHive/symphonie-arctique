# Soil-Audio: Sonification of Tasiujaq Permafrost Temperature Curves

**Project:** Generate atmospheric orchestral music whose dynamics and harmony are
shaped by the temperature time-series of soil at multiple depths in Tasiujaq.

**Date:** 2026-05-05

---

## TL;DR

We produce a layered orchestral piece in which **each soil depth becomes its own
musical stem**. Per-depth dynamics follow the actual temperature curve of that
depth (volatile surface ↔ near-static deep), and per-depth harmony is anchored
in pitches derived from an EMD (Empirical Mode Decomposition) of the same data.
Stems are summed into a final mix.

The current pipeline uses **post-hoc envelope modulation** of model output —
not gradient-based conditioning. A more rigorous "envelope-conditioned"
generation (DITTO-style inference-time optimization) is documented as **Path B**
for a future iteration.

---

## Inputs

| File | Source | Role |
|---|---|---|
| `temperature_curves_by_depth_Tasiujaq.csv` | Field measurements, 1993–2023, daily | Drives **dynamics** (envelope per depth) |
| `depth_chords_microtonal.mid` | EMD decomposition of the same series, exported as MPE microtonal MIDI | Drives **harmony** (pitch class per depth) |

### CSV structure

- 9 748 rows (≈ 26.7 years daily)
- 14 depth columns: `0, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 700, 900, 1100` (cm)
- Many NaN at shallow depths (5, 10, 20 cm have ~54 % coverage). 11 depths have ≥ 80 % coverage.

### MIDI structure

- 1 track, 67 note events, all velocity 64, evenly spaced (480 ticks each ≈ 0.5 s)
- 16 channels used round-robin (MPE convention) — channel does **not** encode depth
- Depth assignment is **sequential**, with variable block size per depth
  - In MIDI order: `200, 300, 400, 500, 700, 900, 1100` cm
  - Block sizes (notes per depth): `[11, 11, 11, 8, 8, 9, 9]` (sum = 67)
  - Each block lists EMD intrinsic-mode-functions (IMFs) **high-frequency → low-frequency**
  - The lowest IMF of every depth is a sub-bass fundamental ~28–32 Hz (= the multi-year temperature trend)

---

## Pipeline Overview

```
                                   ┌───────────────────────────────────┐
                                   │  temperature_curves_by_depth.csv  │
                                   └─────────────┬─────────────────────┘
                                                 │
                          ┌──────────────────────┼─────────────────────┐
                          ▼                      ▼                     ▼
                  per-depth interp           one-year window      per-depth std°C
                          │                      │                     │
                          └──────────┬───────────┘                     │
                                     ▼                                 │
                              normalized envelope                      │
                              (44.1 kHz, [0.1, 1.0])                   │
                                     │                                 │
                                     │             ┌───────────────────┘
                                     │             │
   ┌──────────────────────────┐      │             ▼
   │  depth_chords.mid (EMD)  │      │   "volatile / static" tag
   └─────────┬────────────────┘      │             │
             ▼                       │             │
     parse + group per depth         │             │
             │                       │             │
             ▼                       │             │
   per-depth tonal anchor            │             │
   (top pitch class)                 │             │
             │                       │             │
             └────────┬──────────────┴─────────────┘
                      ▼
           ┌─────────────────────────┐
           │      prompts.py         │
           │  per-depth SAO prompt   │
           └────────┬────────────────┘
                    ▼
        ┌────────────────────────────┐
        │ Stable Audio Open 1.0      │
        │ (text → 47 s stereo wav)   │
        └────────┬───────────────────┘
                 │
                 ▼ apply envelope (post-hoc amplitude modulation)
                 │
         per-depth stem (.wav)
                 │
                 ▼ layer 7 stems with chosen mix mode
                 │
          final mix (.wav)
```

---

## Per-depth Processing Details

### 1. Envelope (from CSV)

For each depth:
1. Linear interpolation of NaN values.
2. Slice the **first contiguous full-year window** (365 valid days) — anchors all stems
   to the same calendar segment for layered coherence.
3. Resample 365 days → 47 s × 44.1 kHz = 2 072 700 audio samples (linear interp).
4. Min-max normalize to **[0.1, 1.0]** — the floor of 0.1 prevents the audio from
   ever going completely silent.
5. Smooth with a 0.5 s box-car kernel to remove daily noise that would sound like clicks.

### 2. Tonal anchor (from MIDI)

For each depth's chord block (post EMD-grouping):
1. Compute every note's actual frequency including pitch-bend (`midi + bend·2 semitones`).
2. Octave-collapse all frequencies to a single pitch-class histogram, with weight
   `max(0, 1 − |cents_offset|/50)` so notes near a 12-TET pitch count more.
3. The **top pitch class** becomes the depth's tonal anchor.

| Depth | Notes | Lowest IMF | Top pitch class |
|---|---|---|---|
| 200 cm | 11 | A♯0 (-8 ¢) | **F** |
| 300 cm | 11 | A0 (+30 ¢) | **F♯** |
| 400 cm | 11 | A0 (+30 ¢) | **F♯** |
| 500 cm | 8 | A♯0 (-6 ¢) | **A♯** |
| 700 cm | 8 | A♯0 (-22 ¢) | **B** |
| 900 cm | 9 | B0 (+38 ¢) | **D** |
| 1100 cm | 9 | A0 (+4 ¢) | **D** |

### 3. Prompt assembly

A SAO prompt for depth *d* is the concatenation of:

| Ingredient | Source | Purpose |
|---|---|---|
| Shared style skeleton | constant in `prompts.py` | Keeps all stems in one sonic world |
| Tonality phrase | top pitch class (MIDI) | Pushes SAO toward the right key |
| Role descriptor | depth bucket (`surface`/`shallow`/`mid`/`deep`/`very_deep`) | Picks instrumentation/register |
| Dynamics descriptor | `std` of temperature in the year window | "highly dynamic", "subtle", "static drone" |
| Negative prompt | constant | Excludes drums, vocals, electronic, lo-fi |

Example for depth 1100 cm:

> *atmospheric orchestral score, cinematic, warm hall reverb, contemplative, natural acoustic instrumentation, lush dynamics, **in D minor, warm contemplative**, **deep low drone, pedal point in the lowest register, almost static**, **almost completely static, sustained drone**, depth 1100 cm soil layer sonification*

### 4. Generation

- Model: `stabilityai/stable-audio-open-1.0`
- Pipeline: `diffusers.StableAudioPipeline`, fp16, CUDA
- 200 diffusion steps, CFG = 7.0
- Seed: `42 + depth_cm` so each depth has a stable but distinct seed
- Output: stereo 44.1 kHz 47 s wav per depth (~30 s of compute on the RTX 4090 Laptop)

### 5. Envelope modulation (post-hoc, v2)

Two effects applied in sequence, both controlled by the depth's envelope:

1. **Amplitude with gamma curve.**
   `out = audio * (env^γ)`, where `env ∈ [floor=0.02, 1.0]` and `γ = 2.0` by default.
   The gamma > 1 makes cold sections *much* quieter, expanding the dynamic range
   from a flat ratio to ~34 dB peak-to-trough.

2. **Dynamic low-pass filter.**
   Per-frame Butterworth LPF whose cutoff follows the envelope:
   `cutoff(t) = lpf_low + (lpf_high − lpf_low) · env(t)`
   with defaults `350 Hz → 16 kHz`. Cold sections are filtered down to a dark/muffled
   character (cutoff near 350 Hz), warm sections open up to full bandwidth.
   Implemented as overlapping Hann-windowed frames (25 ms / 12.5 ms hop) re-filtered
   per frame.

Together these make the soil signal *audibly* shape the music's tone *and*
loudness, not just its volume — a much stronger perceived effect than v1's plain
amplitude multiplication. It still does not change *content* (notes, rhythm), but
the timbre/dynamic envelope is now distinctly soil-driven.

### 6. Mixing

`mix.py` offers two modes:

- `equal` — every stem at unity gain, then peak-normalized to −1 dBFS
- `physical` — each stem gain is set in dB by the **log of the depth's
  temperature std** (volatile surface depths louder, static deep depths quieter).
  Range: −18 dB ↔ 0 dB.

`physical` is the recommended starting point: it produces a layered piece where the
deep depths function as a quiet drone foundation and the surface depths sit on top
as the most expressive voices, mirroring the actual physical heat-diffusion reality.

---

## Output Files

```
soil-audio/
├── outputs/
│   ├── soil_full_history.png              ← all 11 depths, 1993–2023 overlay
│   ├── soil_envelopes_per_depth_year1.png ← per-depth normalized envelopes (1 year)
│   ├── soil_envelopes_per_depth_full.png  ← per-depth envelopes (full 27 yr compressed)
│   ├── depth_harmonies.json               ← parsed MIDI chords + descriptors
│   ├── test_orchestral_01.wav             ← initial smoke-test SAO clip
│   ├── test_orchestral_01_mod_*cm_year1.wav ← single-clip envelope baselines
│   ├── stems/
│   │   ├── manifest.json                  ← per-stem prompt, seed, env stats
│   │   ├── stem_NNNNcm_raw.wav            ← raw SAO output, no envelope
│   │   └── stem_NNNNcm_envmod.wav         ← envelope-modulated stem
│   ├── mix_raw_equal.wav                  ← 7 raw stems summed equally
│   ├── mix_raw_physical.wav               ← 7 raw stems, std-weighted
│   ├── mix_envmod_equal.wav               ← 7 modulated stems, equal sum
│   └── mix_envmod_physical.wav            ← 7 modulated stems, std-weighted (recommended)
```

---

## How to Run (Reproducibility)

Prereqs: Conda env `comfyui` (Python 3.12, torch 2.9.1+cu130). Already installed
on this machine. To recreate elsewhere, install:

```bash
pip install diffusers>=0.30 transformers accelerate soundfile pandas matplotlib mido
hf auth login          # paste a HuggingFace token with access to stable-audio-open-1.0
```

Then from the project root:

```bash
# 1. Inspect & visualize the soil data (optional, sanity-check)
python inspect_csv.py
python plot_envelopes.py

# 2. Parse the MIDI chord file into per-depth harmonies
python parse_midi.py --notes-per-depth "11,11,11,8,8,9,9"

# 3. Generate one stem per depth (~3–4 min total on RTX 4090)
python generate_per_depth.py

# 4. Mix stems
python mix.py --mode physical --variant envmod --out outputs/final_mix.wav
```

To regenerate a single depth with new prompt/seed:

```bash
python generate_per_depth.py --depths 1100 --seed 100
```

To experiment with the envelope baseline on an arbitrary clip:

```bash
python apply_envelope.py --in some.wav --depth 200 --window first_year --out modulated.wav
```

---

## Key Decisions & Tradeoffs

| Decision | Why | Cost |
|---|---|---|
| **Stable Audio Open 1.0** (not 2.5) | Open weights → required for any future gradient-based conditioning (Path B). | Lower quality than 2.5; 47 s max duration. |
| **47 s clips** (not 3 min) | Architectural max of SAO 1.0. Stitching to 3 min is doable later. | Single-shot ≤ 47 s. |
| **One full year per depth** | 1 annual seasonal cycle compressed into 47 s — the "natural" musical unit of soil dynamics. | Doesn't show multi-year trend. The 27-year-compressed version is also generated for comparison. |
| **First-full-year window** (rather than picking by year) | Anchors all 7 depths to the same calendar segment for inter-stem coherence. | Some depths' first-full-year may be later than others; we take per-depth start that may differ by ±a few years (printed in manifest). |
| **EMD split [11,11,11,8,8,9,9]** | Heuristic from frequency-content patterns + user confirmation that EMD produces variable IMF count per signal. | If the original generating script used a different split, the prompts would be slightly off — easy to override via `--notes-per-depth`. |
| **Top-pitch-class as tonal anchor** (vs. full chord transcription) | SAO doesn't reliably parse complex chord descriptors; a single-key cue is more effective. | Loses the harmonic richness of the EMD spectrum. Could be improved with audio-prompt-based pitch control. |
| **Post-hoc amplitude modulation** (vs. true envelope conditioning) | Trivially correct, no model training, validates the data pipeline end-to-end. | Modulates only **loudness**, not musical content. The stem internally still has the dynamics SAO chose. |
| **`physical` mix mode default** | Mirrors the physics: deep = quiet drone, surface = expressive top voice. | Subjective — `equal` may be preferable for some uses. Both are produced. |
| **Diffusers (not stable-audio-tools)** | Simpler dependency footprint for a forward-only inference workflow. | Doesn't expose sampler internals — would need to switch for Path B. |

---

## Known Limitations

1. **Approximate pitch control.** SAO is text-conditioned, not pitch-conditioned. The
   prompt's *"in F♯ minor"* nudges the model but doesn't enforce it. Listening will
   reveal that the actual key may drift.

2. **No content-level conditioning by the soil signal.** The envelope modulates loudness
   only. The model's note choices, rhythm, instrumentation are independent of the soil
   curve (they come from the prompt + seed, not the data).

3. **Per-depth coherence is not enforced across stems.** Each stem is generated
   independently. Phase, key, tempo of one stem are unrelated to another's. Layering
   relies on the prompts being similar enough (shared atmospheric skeleton, related keys)
   to sit together. Some seeds will combine more harmoniously than others — iterate the
   `--seed` value to taste.

4. **47 s ceiling.** For a longer piece, multiple 47 s segments must be stitched with
   crossfades, with prompts varied across the timeline.

---

## Next Steps (in priority order)

### Immediate

- **Iterate seeds / prompts.** Listen to `outputs/mix_envmod_physical.wav` and the
  individual stems. Adjust `prompts.py` to taste; re-run individual depths with
  `python generate_per_depth.py --depths NNNN --seed K`.
- **Try alternative time windows.** The full 27-year history compressed to 47 s gives
  a very different musical character (multi-year trend visible) — quick to test by
  modifying `generate_per_depth.py` to call `envelope_for_depth(...)` with
  `start_day=0, n_days=None`.

### Path B — true envelope-conditioned generation (DITTO-style)

Replace the post-hoc amplitude modulation with **inference-time noise-latent
optimization**, so the model itself plans dynamics matching the soil envelope rather
than just having its volume turned up and down afterwards.

Approach:
1. Use the same SAO base model.
2. Define a loss `L = ||RMS(generated_audio, win=N) − target_envelope||²`.
3. Backprop through the diffusion sampler (using `torch.utils.checkpoint` to fit in 16 GB VRAM).
4. Optimize the **initial latent noise** with a few gradient steps until the loss is small enough.
5. Decode the optimized latent → final waveform.

Cost: ~10 × baseline generation time (≈ 5 min per stem). Total for 7 stems ≈ 35 min.

This is methodologically the most rigorous path. References: DITTO (Novack et al., 2024),
DITTO-2 (Novack et al., 2024).

### Path C — train a real envelope ControlNet on SAO

Train an envelope-conditioned ControlNet adapter on Stable Audio Open using
`EmilianPostolache/stable-audio-controlnet`'s `controlnet_types=["envelope"]` code path
on a music dataset paired with computed RMS envelopes. ~1–3 days of compute on the 4090.
Cleanest paper story; biggest engineering cost.

### Stitching to 3 minutes

Generate 4 segments × 45 s with a designed prompt arc (intro → build → peak → resolve)
per depth, crossfaded with 3 s equal-power overlaps. Re-run mixing on the longer stems.
Optional: expose `--start_day` / `--n_days` per segment to map a multi-year arc across
the 3 min instead of 4× one-year repeats.

---

## Provenance

- Stable Audio Open 1.0 weights from HuggingFace, license: CC-BY-NC.
- Tasiujaq soil temperature data: client-provided, please cite the original field study
  in any publication.
- This project's source files: `inspect_csv.py`, `soil_data.py`, `plot_envelopes.py`,
  `inspect_midi.py`, `inspect_midi_full.py`, `parse_midi.py`, `prompts.py`, `generate.py`,
  `apply_envelope.py`, `generate_per_depth.py`, `mix.py`, `README.md`.
