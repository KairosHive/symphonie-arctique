"""
v15.2 progression — same S1/S2/S3 as v15, but S4 (the final ~45 s segment)
is rewritten with more active / movement-rich language.

v15's S4 used "fading / last / dissolving" cues for most depths, which made
the final stretch feel less alive. v15.2 replaces those with cues for
energetic motion, rapid arpeggios, walking bass, choral counterpoint, and
propulsive rhythm — so the final segment becomes the most-warming / most-
melodic climax of the piece.

S1, S2, S3 are imported verbatim from v15.
"""
from __future__ import annotations
import copy

from progression import (
    NEGATIVE,
    SHARED_QUALITY,
    PITCH_CLASS_HINT,
    envelope_descriptor,
    DepthPrompt,
    N_SEGMENTS,
    segment_day_window,
    SEGMENT_NARRATIVE,
)
from progression_v15 import (
    DEPTH_PROGRESSION_V15,
    SEGMENT_CHORDS,
    SEGMENT_TRANSITION,
)


# Build v15.2 by deep-copying v15 then overriding S4 (index 3) per depth.
DEPTH_PROGRESSION_V15_2: dict[int, list[dict[str, str]]] = copy.deepcopy(DEPTH_PROGRESSION_V15)

# More-active S4 prompts — take 2 (less dissonant, more piano, less high-pitch violin)
S4_OVERRIDES = {
    200: {  # soprano + celesta — gentler, mid register, consonant
        "world": "soft soprano singing gentle flowing melodic phrases with soft celesta accompaniment, warm consonant melody in mid register, peaceful melodic motion, no shrill high notes",
        "tempo": "gentle flowing melody",
    },
    300: {  # piano + Rhodes — MORE PIANO, fuller writing
        "world": "rich solo grand piano playing a flowing melodic right hand with arpeggiated left hand bass, warm Rhodes electric piano underneath, expressive piano nocturne, mid register, consonant warm harmony, active piano with multiple notes",
        "tempo": "active flowing piano with bass arpeggios",
    },
    400: {  # viola — LOWER register, warm, no high notes
        "world": "low warm viola playing a melodic line in the lower register, sustained tones with gentle bowed motion, dark and lyrical voice, mid-low register, soft bow, no high notes",
        "tempo": "warm melodic line, mid-low register",
    },
    500: {  # choir — consonant, simpler shifts
        "world": "wordless choir singing gentle moving consonant harmonies with simple vowel shifts Ah-Oh, slow chord changes within a warm key, mid register, harmonious and peaceful, no dissonance",
        "tempo": "gentle consonant choral motion",
    },
    700: {  # taiko — propulsive but warm, less harsh
        "world": "warm rhythmic taiko patterns with steady regular pulse, low male humming melody on simple notes, controlled rhythmic build, low register, propulsive but warm, no sharp dissonance",
        "tempo": "warm regular rhythm",
    },
    900: {  # cellos + bass voice — consonant
        "world": "cello section playing a warm walking bass line with consonant melodic motion, ascending and descending phrases on diatonic notes, soft male bass voice on moving phrases, mid-low register, harmonious bass motion",
        "tempo": "consonant walking bass",
    },
    1100: {  # throat singing — gentler motion
        "world": "deep throat singing with subtle slow pitch shifts on warm fundamentals, ancient Tibetan bowls on changing harmonious tones, gentle deep motion, lowest register, no dissonance",
        "tempo": "slow gentle shifting fundamentals",
    },
}
for d, override in S4_OVERRIDES.items():
    DEPTH_PROGRESSION_V15_2[d][3] = override  # index 3 = S4


def build_segment_prompt_v15_2(
    depth_cm: int,
    segment_idx: int,
    top_pitch_class: str,
    raw_std: float,
    include_pitch_hint: bool = False,
) -> DepthPrompt:
    if depth_cm not in DEPTH_PROGRESSION_V15_2:
        raise ValueError(f"depth {depth_cm} not in DEPTH_PROGRESSION_V15_2")
    if segment_idx < 0 or segment_idx >= N_SEGMENTS:
        raise ValueError(f"segment_idx must be 0..{N_SEGMENTS-1}")

    seg = DEPTH_PROGRESSION_V15_2[depth_cm][segment_idx]
    env_lang = envelope_descriptor(raw_std)
    chord_lang = SEGMENT_CHORDS[segment_idx] if include_pitch_hint else ""
    transition_lang = SEGMENT_TRANSITION[segment_idx]

    parts = [seg["world"], seg["tempo"]]
    if chord_lang:
        parts.append(chord_lang)
    parts.extend([transition_lang, env_lang, SHARED_QUALITY])
    prompt = ", ".join(parts)
    return DepthPrompt(
        depth_cm=depth_cm,
        prompt=prompt,
        negative=NEGATIVE,
        role=f"v15_2-d{depth_cm}-s{segment_idx}",
    )


if __name__ == "__main__":
    # Show only S4 prompts (the new ones)
    import json
    from pathlib import Path
    import numpy as np
    from soil_data import load_csv, extract_depth_series

    harmonies = json.loads(Path("outputs/depth_harmonies.json").read_text(encoding="utf-8"))
    df = load_csv()
    n_total = len(df)
    print("=== v15.2 S4 prompts (the only changes vs v15) ===\n")
    for d_str, info in harmonies.items():
        d = int(d_str)
        top_pc = info["top_pitch_classes"][0][0]
        raw = extract_depth_series(df, d)
        s, n = segment_day_window(n_total, 3)
        window = raw[s : s + n]
        std = float(np.std(window[~np.isnan(window)]))
        p = build_segment_prompt_v15_2(d, 3, top_pc, std)
        print(f"Depth {d} cm, S4:\n  {p.prompt}\n")
