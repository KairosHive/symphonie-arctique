"""
3-minute progression: per-depth × per-segment prompt evolution.

Structure: 4 segments × 7 depths = 28 generations, stitched into 7 streams of
~179 s, then mixed.

Per depth, the *instrument family* stays fixed across segments (200 cm always
ice cracking, 400 cm always solo violin, etc.) while *intensity, fragmentation,
and decay descriptors* evolve. The 4 segments tell a "memory melting" arc:

    Seg 1 (1993-2000) — pristine / intact / stable
    Seg 2 (2000-2007) — early warming / subtle fissures
    Seg 3 (2007-2014) — active thaw / dramatic change
    Seg 4 (2014-2023) — late warming / fragmentation / collapse

The data window for each segment spans ~6.75 years of the 27-year record, so
the multi-year temperature trend is embedded in the audio's evolution.
"""
from __future__ import annotations
from dataclasses import dataclass

from prompts import (
    NEGATIVE,
    SHARED_QUALITY,
    PITCH_CLASS_HINT,
    envelope_descriptor,
    DepthPrompt,
)

N_SEGMENTS = 4

# Per-depth × per-segment descriptors. Each entry is the segment-specific
# language replacing the static "world" + "tempo" of v3.
DEPTH_PROGRESSION: dict[int, list[dict[str, str]]] = {
    200: [  # ice cracking surface
        {"world": "intact frozen lake surface, sparse brittle frost shimmer, occasional crystalline pops, mostly stable ice, gentle glassy harmonics, high register",
         "tempo": "rare fragments, mostly silent"},
        {"world": "ice surface beginning to fissure, more frequent crystalline cracks, brittle frost shimmer, glassy harmonics with occasional sharper breaks, water seeping into ice",
         "tempo": "increasing fragments"},
        {"world": "extensive thaw, ice plates fracturing, glassy meltwater pouring, dramatic crystalline breaks, prepared piano string scrapes, glissando ice slides",
         "tempo": "frequent violent fragments"},
        {"world": "broken ice cascading shards, glassy meltwater flooding, surface disintegration, dramatic crystalline shattering, chaotic frost noise",
         "tempo": "dense chaotic fragments"},
    ],
    300: [  # music box memory
        {"world": "intact antique music box melody, complete fragile lullaby, celesta and glockenspiel chimes ringing clearly, delicate full melody",
         "tempo": "slow lullaby, complete phrases"},
        {"world": "music box slowing imperceptibly, occasional missing notes, fragile chimes thinning, faintly imperfect memory",
         "tempo": "slow lullaby, slightly faltering"},
        {"world": "fragmented music box, sparse incomplete melody, faltering chimes, missing notes more frequent, antique reverb",
         "tempo": "slow, broken phrases"},
        {"world": "broken music box, isolated single notes, fragments of forgotten lullaby, ghostly chimes echoing into silence, very sparse",
         "tempo": "very slow, sparse echoes"},
    ],
    400: [  # muted viola lamento (was solo violin in earlier draft — viola has more
            # stable intonation in SAO, and removing "vibrato" cues reduces pitch wobble)
        {"world": "muted viola sul tasto, soft sustained long bowed phrases, complete plaintive melancholic theme, mid register, slow legato",
         "tempo": "slow, full phrases"},
        {"world": "muted viola sul tasto, slower sustained tones with longer breaths between phrases, gentle bowed legato, mid register",
         "tempo": "slow, longer breaths"},
        {"world": "muted viola fragmenting, sparse held sustained notes, longer silences between phrases, mid register, ghostly",
         "tempo": "very slow, sparse"},
        {"world": "fading muted viola, last fragile sustained tones, sparse lonely held notes dissolving into silence, mid register, dying voice",
         "tempo": "near silent, ghostly"},
    ],
    500: [  # ancestral wordless choir
        {"world": "full ancestral wordless choir, dense humming harmonies, complete vocal pad, distant throat singing, rich collective memory",
         "tempo": "slow processional, full chorus"},
        {"world": "ancestral choir slowly thinning, fewer voices, lighter humming, vocal pad with gaps appearing",
         "tempo": "slow, lighter texture"},
        {"world": "scattered voices, half-empty choir, distant singers fading, sparse vocal pad, isolated humming voices",
         "tempo": "slow, sparse"},
        {"world": "few remaining voices, near silence, last ethereal hummings, single distant voice, ghostly choir disappearing",
         "tempo": "very slow, dissolving"},
    ],
    700: [  # slow deep percussion
        {"world": "slow deep timpani heartbeat, regular geological pulse, ritual taiko drums, low rumble, primordial earth rhythm",
         "tempo": "slow heartbeat, regular"},
        {"world": "deep heartbeat slowing, more spaced impacts, fading taiko drums, low rumble persisting",
         "tempo": "slower heartbeat, irregular"},
        {"world": "rare distant drums, sparse muffled timpani hits, fading rituals, long decays into silence",
         "tempo": "very slow, sparse"},
        {"world": "near-silent earth, occasional muted percussion impacts, geological stillness, the heartbeat almost stopped",
         "tempo": "near static, rare hits"},
    ],
    900: [  # tectonic sustained basses
        {"world": "massive sustained double bass section, deep bowed drone, full glacial weight, contrabass density",
         "tempo": "near-static, full"},
        {"world": "sustained basses thinning, lighter section, slightly reduced glacial density",
         "tempo": "near-static, lighter"},
        {"world": "fewer bass voices, lighter drone, distant sustained tones, gentle low register",
         "tempo": "near-static, sparse"},
        {"world": "single bass voice, sparse sustained tone, fading deep drone, glacial weight dissolving",
         "tempo": "near-static, dissolving"},
    ],
    1100: [  # permafrost eternity
        {"world": "permafrost drone, eternal frozen silence, ancient Tibetan singing bowls, motionless lowest register, glacier interior reverb",
         "tempo": "completely static"},
        {"world": "permafrost drone, eternal silence with the faintest hint of stirring deep below, motionless",
         "tempo": "completely static, barely sensed"},
        {"world": "permafrost drone with very subtle warming undertone, motionless ancient ice barely changing",
         "tempo": "completely static, faint shift"},
        {"world": "permafrost drone, ancient frozen silence with subtle melting hint, distant deep groans, motionless eternity yielding",
         "tempo": "completely static, dissolving"},
    ],
}


SEGMENT_NARRATIVE = [
    "1993-2000 — pristine epoch",
    "2000-2007 — early warming",
    "2007-2014 — active thaw",
    "2014-2023 — late warming / collapse",
]


def build_segment_prompt(
    depth_cm: int,
    segment_idx: int,
    top_pitch_class: str,
    raw_std: float,
    include_pitch_hint: bool = True,
) -> DepthPrompt:
    if depth_cm not in DEPTH_PROGRESSION:
        raise ValueError(f"depth {depth_cm} not in DEPTH_PROGRESSION")
    if segment_idx < 0 or segment_idx >= N_SEGMENTS:
        raise ValueError(f"segment_idx must be 0..{N_SEGMENTS-1}")

    seg = DEPTH_PROGRESSION[depth_cm][segment_idx]
    env_lang = envelope_descriptor(raw_std)
    key_lang = PITCH_CLASS_HINT.get(top_pitch_class, "") if include_pitch_hint else ""

    parts = [seg["world"], seg["tempo"]]
    if key_lang:
        parts.append(key_lang)
    parts.extend([env_lang, SHARED_QUALITY])
    prompt = ", ".join(parts)
    return DepthPrompt(
        depth_cm=depth_cm,
        prompt=prompt,
        negative=NEGATIVE,
        role=f"depth-{depth_cm}-seg{segment_idx}",
    )


def segment_day_window(total_days: int, segment_idx: int, n_segments: int = N_SEGMENTS) -> tuple[int, int]:
    """
    Return (start_day, length) for one segment of an evenly-divided history.
    """
    seg_len = total_days // n_segments
    start = segment_idx * seg_len
    if segment_idx == n_segments - 1:
        # Last segment grabs any remainder
        return start, total_days - start
    return start, seg_len


if __name__ == "__main__":
    # Demo: print all 28 prompts.
    import json
    from pathlib import Path
    import numpy as np

    from soil_data import load_csv, find_first_full_year, extract_depth_series

    harmonies = json.loads(Path("outputs/depth_harmonies.json").read_text(encoding="utf-8"))
    df = load_csv()
    n_total = len(df)
    print(f"Total days in CSV: {n_total} ({n_total/365.25:.1f} years)")
    print()
    for i, narr in enumerate(SEGMENT_NARRATIVE):
        s, n = segment_day_window(n_total, i)
        y_start = df["Year"].iloc[s]
        y_end = df["Year"].iloc[min(s + n - 1, n_total - 1)]
        print(f"Segment {i+1}: days [{s}, {s+n-1}]  ~{y_start}–{y_end}  ({narr})")
    print()

    for d_str, info in harmonies.items():
        d = int(d_str)
        top_pc = info["top_pitch_classes"][0][0]
        raw = extract_depth_series(df, d)
        print(f"=== Depth {d} cm  (top_pc={top_pc})")
        for i in range(N_SEGMENTS):
            s, n = segment_day_window(n_total, i)
            window = raw[s : s + n]
            std = float(np.std(window[~np.isnan(window)]))
            p = build_segment_prompt(d, i, top_pc, std)
            print(f"  S{i+1} std={std:.2f}°C")
            print(f"    {p.prompt}")
        print()
