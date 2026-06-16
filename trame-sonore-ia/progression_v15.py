"""
v15 progression — same ensemble identity as v8/v14, but actively MELODIC.

Goal: replace the sustained-pad / drone descriptors with phrased melodic
descriptors. Each voice keeps its instrument family and per-segment evolution
arc (pristine -> warming -> thawing -> final), but the language now asks
for notes, runs, phrases, walking lines, and shifting harmonies instead of
"sustained / motionless / drone".

Used for the report's Version 2 (microtonal-anchored sonification) when the
delivered v15 audio supersedes v14.
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

# Same chord progression structure as v8 (for optional injection); used only if
# include_pitch_hint=True. By default we keep --no_pitch_hint at the CLI for
# more natural variability.
SEGMENT_CHORDS = [
    "in F-sharp minor key",
    "modulating to A major key",
    "in D minor key",
    "ending in F major key",
]
SEGMENT_TRANSITION = [
    "settled tonal center",
    "lifting to brighter mood",
    "darkening into deeper register",
    "expanding to final resolution",
]

DEPTH_PROGRESSION_V15: dict[int, list[dict[str, str]]] = {
    200: [  # SOPRANO MELODIC + CELESTA ARPEGGIOS
        {"world": "soprano voice singing wordless flowing melodic phrases with Ah and Eh vowels, light celesta arpeggio runs underneath, expressive melodic line, mid-high register, warm not piercing",
         "tempo": "flowing melodic phrases"},
        {"world": "soprano with more elaborate melodic phrases, longer expressive lines with subtle ornaments, active celesta arpeggios, warm high register",
         "tempo": "active melodic lines"},
        {"world": "soprano with deeper expressive melody, longer melodic phrases with vocal ornaments, gentle celesta motion underneath, mid-high register",
         "tempo": "expressive flowing melody"},
        {"world": "soprano with last melodic gestures, sparse expressive phrases, final celesta motifs, mid-high register, dissolving with grace",
         "tempo": "slow expressive last phrases"},
    ],
    300: [  # PIANO FLOWING ARPEGGIOS + RHODES MELODIC
        {"world": "solo piano playing flowing arpeggio runs with a singing melodic top line, warm Rhodes electric piano accompaniment, Chopin-like nocturne, mid register, melodic and active",
         "tempo": "flowing arpeggio motion"},
        {"world": "piano with more elaborate melodic runs, Debussy-like cascading arpeggios, Rhodes harmonic underlay, mid register, active piano writing",
         "tempo": "active flowing arpeggios"},
        {"world": "piano with melancholic flowing melody, slower arpeggios with longer phrases, Rhodes warm chord progressions, mid register",
         "tempo": "slow flowing melody"},
        {"world": "piano with sparse fading melodic phrases, last arpeggio gestures, fading Rhodes, mid register, ghostly melodic fragments",
         "tempo": "slow last melodic gestures"},
    ],
    400: [  # VIOLA PHRASED MELODIC LINE
        {"world": "muted viola sul tasto playing a phrased melodic line with longer flowing notes, occasional double stops, plaintive song-like melody, mid register, actively melodic not just sustained",
         "tempo": "flowing melodic phrases"},
        {"world": "muted viola with elaborate melodic phrases, expressive slides between notes, longer bowed lines with motion, mid register",
         "tempo": "active phrased melody"},
        {"world": "muted viola with melancholic melodic line, slower phrases but still note-by-note motion, occasional vibrato, mid register",
         "tempo": "slow phrased melody"},
        {"world": "muted viola with last melodic gestures, sparse phrases of notes, fading melodic line, mid register, dying voice with grace",
         "tempo": "slow last melodic phrases"},
    ],
    500: [  # CHOIR SHIFTING VOWELS + HARMONIC MOTION
        {"world": "wordless choir singing slowly moving harmony with shifting vowels Ah-Oh-Ee, gentle harmonic motion between chords, melodic mid register, moving voices not a static pad",
         "tempo": "slow shifting harmony"},
        {"world": "choir with brighter shifting harmonies, more elaborate vowel changes, harmonic motion through warmer chords, mid register",
         "tempo": "slow moving harmony"},
        {"world": "choir with darker harmonic motion, slowly shifting vowels through minor colorations, melodic vocal motion, mid register",
         "tempo": "slow harmonic motion"},
        {"world": "choir with last shifting harmonies, fewer voices but still moving, slowly dissolving vowels and chords, mid register",
         "tempo": "slow dissolving harmony"},
    ],
    700: [  # TAIKO RHYTHMIC PATTERNS + MALE HUMMING MELODY
        {"world": "patterned taiko rhythm with regular pulse, low male humming melody on long notes, syncopated drum accents, low register, rhythmic and melodic",
         "tempo": "regular rhythmic pulse"},
        {"world": "fuller taiko patterns with hand drums, more elaborate rhythm, low male humming melody with more motion, low register",
         "tempo": "active rhythmic patterns"},
        {"world": "driving taiko patterns with complex rhythmic interplay, low male humming melody with phrased lines, low register, ritual rhythm and song",
         "tempo": "complex driving rhythm"},
        {"world": "dense rhythmic climax with taiko hand drums and pizzicato cello ostinato, bold low male humming melody, low register, urgent rhythmic build",
         "tempo": "dense urgent rhythm"},
    ],
    900: [  # WALKING CELLO BASS LINE
        {"world": "cello section playing a slow walking bass melodic line, slowly moving from note to note, soft male bass voice doubling on long held notes, mid-low register, melodic bass motion",
         "tempo": "slow walking bass"},
        {"world": "cello walking bass with more melodic motion, slightly faster note-to-note movement, bass voice on melodic notes, mid-low register",
         "tempo": "moving bass line"},
        {"world": "cello bass line with melodic phrases, slow ascending and descending motion, bass voice on harmonic notes, mid-low register",
         "tempo": "phrased bass melody"},
        {"world": "cello bass with last melodic phrases, sparse but moving notes, fading bass voice, mid-low register, dissolving bass line",
         "tempo": "slow dissolving bass"},
    ],
    1100: [  # THROAT SINGING WITH SHIFTING NOTES
        {"world": "deep low throat singing with subtle pitch shifts between long held notes, ancient Tibetan singing bowls with shifting fundamentals, occasional harmonic motion, lowest register, slow but moving",
         "tempo": "slow shifting notes"},
        {"world": "throat singing with slight melodic motion, shifting fundamentals more frequently, Tibetan bowls on new tones, lowest register",
         "tempo": "slow shifting melody"},
        {"world": "throat singing with subtle melodic phrases on the fundamental, deep harmonic shifts, ancient ice with breath-like motion, lowest register",
         "tempo": "very slow shifting"},
        {"world": "throat singing on warmer fundamental, last subtle pitch shifts, distant deep groans with melodic hints, lowest register, dissolving",
         "tempo": "very slow dissolving notes"},
    ],
}


def build_segment_prompt_v15(
    depth_cm: int,
    segment_idx: int,
    top_pitch_class: str,
    raw_std: float,
    include_pitch_hint: bool = False,
) -> DepthPrompt:
    """
    Default include_pitch_hint=False (matches the v14/Report v2 setup).
    Pass True to add the segment chord cue back in.
    """
    if depth_cm not in DEPTH_PROGRESSION_V15:
        raise ValueError(f"depth {depth_cm} not in DEPTH_PROGRESSION_V15")
    if segment_idx < 0 or segment_idx >= N_SEGMENTS:
        raise ValueError(f"segment_idx must be 0..{N_SEGMENTS-1}")

    seg = DEPTH_PROGRESSION_V15[depth_cm][segment_idx]
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
        role=f"v15-d{depth_cm}-s{segment_idx}",
    )


if __name__ == "__main__":
    import json
    from pathlib import Path
    import numpy as np

    from soil_data import load_csv, extract_depth_series

    harmonies = json.loads(Path("outputs/depth_harmonies.json").read_text(encoding="utf-8"))
    df = load_csv()
    n_total = len(df)

    for d_str, info in harmonies.items():
        d = int(d_str)
        top_pc = info["top_pitch_classes"][0][0]
        raw = extract_depth_series(df, d)
        print(f"=== Depth {d} cm  v15  (top_pc={top_pc})")
        for i in range(N_SEGMENTS):
            s, n = segment_day_window(n_total, i)
            window = raw[s : s + n]
            std = float(np.std(window[~np.isnan(window)]))
            p = build_segment_prompt_v15(d, i, top_pc, std)
            print(f"  S{i+1}: {p.prompt}")
        print()
