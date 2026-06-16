"""
v8 progression — voices, chord changes, more felt soil fluctuations.

Three new ideas vs v7:
  1. Each segment shares a chord — F# minor → A major → D minor → F major —
     so all 7 stems modulate together. Listener hears clear chord changes at
     ~0:44, 1:28, 2:12.
  2. Voice layers added: soprano (200), humming with viola (400), choir (500
     kept), low male humming with perc (700), throat singing under sub (1100).
  3. The 200 cm "glass bells" stem (which produced a strident high tinkle) is
     replaced by a soft wordless soprano + light celesta — much less piercing,
     and adds a melodic high-register voice instead.

Use with stronger envelope params (mod_depth=0.7, gamma=2.0, lpf_low_hz=800)
to make the soil fluctuations more audible.
"""
from __future__ import annotations

from progression import (
    NEGATIVE,
    SHARED_QUALITY,
    PITCH_CLASS_HINT,
    envelope_descriptor,
    DepthPrompt,
    SEGMENT_NARRATIVE,
    N_SEGMENTS,
    segment_day_window,
)


# Shared chord progression across all 7 depths and across the 4 segments.
# Each tuple is the segment's harmonic anchor — injected into every depth's prompt.
SEGMENT_CHORDS = [
    "in F-sharp minor key",
    "modulating to A major key",
    "in D minor key",
    "ending in F major key",
]

# Each segment also gets a short transition-feel descriptor (helps SAO commit
# to the chord change rather than ignore it).
SEGMENT_TRANSITION = [
    "settled tonal center",
    "lifting to brighter mood",
    "darkening into deeper register",
    "expanding to final resolution",
]


DEPTH_PROGRESSION_V8: dict[int, list[dict[str, str]]] = {
    200: [  # SOFT SOPRANO + LIGHT CELESTA (replaces glass bells)
        {"world": "soft wordless soprano voice singing gentle Ah vowels, fragile high register, light celesta sparkles, ethereal female voice, intimate and warm not piercing",
         "tempo": "slow gentle vocal phrases"},
        {"world": "soft soprano voice slowly opening, melodic Ah phrases with subtle vibrato, distant celesta accents, warm high register, ethereal",
         "tempo": "slow melodic phrases"},
        {"world": "soft soprano voice with deeper vocal expression, longer Ah phrases, gentle celesta beneath, contemplative high register",
         "tempo": "slow expressive phrases"},
        {"world": "soprano voice fading, sparse last vowel phrases, distant celesta echoes, dissolving high register",
         "tempo": "very slow, dissolving"},
    ],
    300: [  # PIANO + WARM RHODES BLEND
        {"world": "solo grand piano with warm Rhodes electric piano underneath, slow contemplative chord, mid register, intimate dual-instrument blend",
         "tempo": "slow contemplative"},
        {"world": "piano and warm Rhodes blend with slightly more elaborate phrases, mid register, soft sustained pedal, contemplative",
         "tempo": "slow flowing"},
        {"world": "piano and Rhodes with hesitant fragmented phrases, mid register, longer pauses between chords, melancholic",
         "tempo": "slow fragmented"},
        {"world": "piano alone with sparse single chord, fading Rhodes, last melodic fragments, mid register, ghostly",
         "tempo": "very slow, sparse"},
    ],
    400: [  # MUTED VIOLA + HUMMING VOICE (adds vocal layer to the lead)
        {"world": "muted viola sul tasto playing slow legato phrases, with quiet wordless humming voice in unison, mid register, intimate dual-voice texture",
         "tempo": "slow, full phrases"},
        {"world": "muted viola with humming voice slowly diverging into harmony, sustained phrases, mid register",
         "tempo": "slow, longer breaths"},
        {"world": "muted viola with sparse humming voice, fragmented sustained notes, longer silences, mid register",
         "tempo": "very slow, sparse"},
        {"world": "fading muted viola with last humming voice fragments, sparse sustained notes dissolving, mid register",
         "tempo": "near silent, ghostly"},
    ],
    500: [  # WORDLESS CHOIR (kept — it works as harmonic backdrop)
        {"world": "wordless ancestral choir with multiple voices, slow harmony chord, soft humming, mid register, harmonic backdrop",
         "tempo": "completely sustained"},
        {"world": "wordless choir slightly thinning with brighter chord harmony, soft humming, mid register",
         "tempo": "completely sustained, lighter"},
        {"world": "scattered choir voices, half-empty vocal pad with darker chord, mid register",
         "tempo": "sustained, sparse"},
        {"world": "few remaining choir voices on warmer chord, near silent vocal pad, last harmonies, mid register",
         "tempo": "sustained, dissolving"},
    ],
    700: [  # PROGRESSIVE PERCUSSION + LOW MALE HUMMING (v10: more present in 2nd half)
        {"world": "distant timpani heartbeat with low male wordless humming, sparse occasional impacts, low register, intro of a long ritual",
         "tempo": "rare distant hits"},
        {"world": "regular timpani pulse with low male humming, present rhythmic pulse establishing, distant taiko entering, low register",
         "tempo": "regular slow pulse"},
        {"world": "full driving taiko and timpani rhythm with bold low male humming, regular powerful pulse, multiple drums, low register, ritual rhythm in full force",
         "tempo": "driving regular rhythm"},
        {"world": "dense rhythmic climax with taiko drums, hand drums, AND fast pizzicato cello ostinato pattern, low male humming voice, urgent rhythmic build, regular insistent rhythm, low register",
         "tempo": "dense fast rhythmic ostinato"},
    ],
    900: [  # SUSTAINED CELLOS + SOFT MALE BASS VOICE
        {"world": "sustained cello section with soft male bass voice singing low wordless tone, slow bowed legato, mid-low register, glacial weight",
         "tempo": "completely sustained"},
        {"world": "sustained cellos with bass voice on brighter chord, lighter density, slow bowed legato, mid-low register",
         "tempo": "completely sustained, lighter"},
        {"world": "thinning cellos with sparse bass voice on darker chord, slow bowed legato, mid-low register",
         "tempo": "sustained, sparse"},
        {"world": "single sustained cello with last bass voice tone, fading low strings, mid-low register, dissolving",
         "tempo": "sustained, dissolving"},
    ],
    1100: [  # PERMAFROST + LOW THROAT SINGING VOICE
        {"world": "permafrost sub-bass drone with deep low throat singing fundamental tone, ancient Tibetan bowls, glacier reverb, eternal sustain",
         "tempo": "completely static"},
        {"world": "permafrost sub-bass with throat singing on brighter fundamental, motionless lowest register, eternal frozen silence",
         "tempo": "completely static"},
        {"world": "permafrost sub-bass with deeper throat singing tone, ancient ice barely shifting, motionless lowest register",
         "tempo": "completely static, faint shift"},
        {"world": "permafrost sub-bass with throat singing on warmest fundamental, distant deep groans, motionless lowest register",
         "tempo": "completely static, dissolving"},
    ],
}


def build_segment_prompt_v8(
    depth_cm: int,
    segment_idx: int,
    top_pitch_class: str,
    raw_std: float,
    include_pitch_hint: bool = True,
) -> DepthPrompt:
    """
    v8 builds a chord-progression prompt instead of using the depth's top pitch class.
    `include_pitch_hint` controls whether the SEGMENT CHORD (not the depth-PC) is injected.
    Default True since the segment chord is the whole point of v8.
    """
    if depth_cm not in DEPTH_PROGRESSION_V8:
        raise ValueError(f"depth {depth_cm} not in DEPTH_PROGRESSION_V8")
    if segment_idx < 0 or segment_idx >= N_SEGMENTS:
        raise ValueError(f"segment_idx must be 0..{N_SEGMENTS-1}")

    seg = DEPTH_PROGRESSION_V8[depth_cm][segment_idx]
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
        role=f"v8-d{depth_cm}-s{segment_idx}",
    )


if __name__ == "__main__":
    import json
    from pathlib import Path
    import numpy as np

    from soil_data import load_csv, extract_depth_series

    harmonies = json.loads(Path("outputs/depth_harmonies.json").read_text(encoding="utf-8"))
    df = load_csv()
    n_total = len(df)

    print("Per-segment chord progression:")
    for i, c in enumerate(SEGMENT_CHORDS):
        print(f"  S{i+1}: {c}  ({SEGMENT_TRANSITION[i]})")
    print()

    for d_str, info in harmonies.items():
        d = int(d_str)
        top_pc = info["top_pitch_classes"][0][0]
        raw = extract_depth_series(df, d)
        print(f"=== Depth {d} cm  v8")
        for i in range(N_SEGMENTS):
            s, n = segment_day_window(n_total, i)
            window = raw[s : s + n]
            std = float(np.std(window[~np.isnan(window)]))
            p = build_segment_prompt_v8(d, i, top_pc, std)
            print(f"  S{i+1}: {p.prompt}")
        print()
