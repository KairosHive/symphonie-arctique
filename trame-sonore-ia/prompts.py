"""
Per-depth prompt engineering for Stable Audio Open.

v3 — "Memory and melting of the ice".

Seven thematically distinct sound worlds, one per soil depth, telling a story
from the active permafrost surface (ice cracking, seasonal thaw) down through
fragmented memory (music box, solo lament, ancestral voices), into the
geological heartbeat of the earth (deep percussion, sustained basses), and
finally to the permafrost eternity (frozen drone).

Surface depths have the biggest seasonal swing — the soil envelope makes them
"melt audibly" in warm season. Deep depths barely move temperature-wise, so
they stay near-static all year (eternity).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DepthPrompt:
    depth_cm: int
    prompt: str
    negative: str
    role: str


SHARED_QUALITY = "high quality acoustic recording, natural reverb"

NEGATIVE = (
    "low quality, distorted, muddy, harsh noise, hiss, vocals lyrics, drum kit, "
    "drum loop, electronic synth, beat, electric guitar, lo-fi, mp3 artifacts, "
    "compressed, talking, speech, dance music"
)


# Seven thematic worlds — surface→deep mirrors the narrative arc.
DEPTH_WORLDS: dict[int, dict[str, str]] = {
    200: {
        "label": "ice cracking surface",
        "world": (
            "crystalline ice cracking textures, glass harmonica, prepared piano string scrapes, "
            "shattering frozen lake, glassy harmonics, brittle frost shimmer, "
            "glissando ice glissandi, high glassy register, fragile and percussive"
        ),
        "tempo": "irregular fragments",
    },
    300: {
        "label": "music box memory",
        "world": (
            "distant antique music box melody, fragmented memory, celesta and glockenspiel chimes, "
            "fragile crystalline tones, gramophone-era reverb, lullaby in slow motion, "
            "mid-high register, nostalgic and tender"
        ),
        "tempo": "slow lullaby",
    },
    400: {
        "label": "solo violin lament",
        "world": (
            "solo violin lament, expressive vibrato and long bowed phrases, "
            "single melancholic voice in vast empty space, slow legato, "
            "mid-high register, the memory of one person speaking through time"
        ),
        "tempo": "slow rubato",
    },
    500: {
        "label": "ancestral wordless choir",
        "world": (
            "wordless ancestral choir, distant throat humming, slowly shifting vowel harmonies, "
            "ethereal vocal pads, transcendent collective memory, "
            "mid register, gentle organ underneath, sacred and ancient"
        ),
        "tempo": "very slow processional",
    },
    700: {
        "label": "slow deep percussion",
        "world": (
            "slow deep percussion, low timpani rolls, distant taiko drums, "
            "earthen wooden hits with long decay, sparse impacts and silences, "
            "geological heartbeat, low register, primordial and ritual"
        ),
        "tempo": "very slow heartbeat",
    },
    900: {
        "label": "tectonic sustained basses",
        "world": (
            "massive sustained double bass section, contrabass bowed drone, "
            "dark warm low strings, tectonic shifts beneath the earth, "
            "very low register, glacial weight, almost no movement"
        ),
        "tempo": "near-static",
    },
    1100: {
        "label": "permafrost eternity",
        "world": (
            "permafrost drone, subterranean rumbling air, ancient Tibetan singing bowls, "
            "glacier interior reverb, infinite low frequencies, motionless, "
            "centuries-old frozen silence, lowest register, eternal sustain"
        ),
        "tempo": "completely static",
    },
}


def envelope_descriptor(std_temp: float) -> str:
    if std_temp >= 8.0:
        return "extremely dynamic, dramatic crescendos and falls, the active thaw"
    if std_temp >= 3.0:
        return "moderately dynamic, gentle swells with the seasons"
    if std_temp >= 1.0:
        return "subtle dynamics, slowly breathing"
    return "completely sustained, no dynamic change, motionless"


PITCH_CLASS_HINT = {
    "C": "in C major",
    "C#": "in C# minor",
    "D": "in D minor",
    "D#": "in Eb minor",
    "E": "in E minor",
    "F": "in F major",
    "F#": "in F# minor",
    "G": "in G major",
    "G#": "in G# minor",
    "A": "in A minor",
    "A#": "in Bb major",
    "B": "in B minor",
}


def build_prompt(depth_cm: int, top_pitch_class: str, raw_std: float) -> DepthPrompt:
    if depth_cm not in DEPTH_WORLDS:
        raise ValueError(
            f"depth {depth_cm} cm not in DEPTH_WORLDS; supported: {sorted(DEPTH_WORLDS)}"
        )
    dw = DEPTH_WORLDS[depth_cm]
    env_lang = envelope_descriptor(raw_std)
    key_lang = PITCH_CLASS_HINT.get(top_pitch_class, "")

    prompt = (
        f"{dw['world']}, {dw['tempo']}, {key_lang}, {env_lang}, {SHARED_QUALITY}"
    )
    return DepthPrompt(depth_cm=depth_cm, prompt=prompt, negative=NEGATIVE, role=dw["label"])


def build_all(harmonies: dict, raw_std_per_depth: dict[int, float]) -> dict[int, DepthPrompt]:
    out = {}
    for d_str, info in harmonies.items():
        d = int(d_str) if isinstance(d_str, str) else d_str
        top_pc = info["top_pitch_classes"][0][0]
        std = raw_std_per_depth.get(d, 1.0)
        out[d] = build_prompt(d, top_pc, std)
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path
    import numpy as np

    from soil_data import load_csv, find_first_full_year, extract_depth_series

    harmonies = json.loads(Path("outputs/depth_harmonies.json").read_text(encoding="utf-8"))
    df = load_csv()
    std_per_depth = {}
    for d in [200, 300, 400, 500, 700, 900, 1100]:
        start, length = find_first_full_year(df, d)
        s = extract_depth_series(df, d)[start : start + length]
        std_per_depth[d] = float(np.std(s))

    prompts = build_all(harmonies, std_per_depth)
    for d in sorted(prompts.keys()):
        p = prompts[d]
        print(f"=== Depth {d} cm  [{p.role}]  std={std_per_depth[d]:.2f}°C")
        print(f"  prompt: {p.prompt}")
        print()
