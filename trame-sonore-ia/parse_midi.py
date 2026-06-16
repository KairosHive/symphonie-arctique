"""
Parse depth_chords_microtonal.mid → per-depth harmonic descriptor.

Convention (from user):
    - 7 depths in order: 200, 300, 400, 500, 700, 900, 1100 (cm)
    - ~10 notes per depth, played sequentially
    - Each note on its own channel with a pitchbend to encode microtonality
    - Total notes = 67  (so blocks aren't perfectly even — default split is
      6 × 10 + 1 × 7. Override with --notes-per-depth if needed.)

Output: outputs/depth_harmonies.json with per-depth:
    - frequencies (Hz, microtonally-detuned)
    - fundamental (lowest freq)
    - harmonic ratios (freq / fundamental)
    - dominant pitch class (12-TET nearest, after octave-collapse)
    - suggested key/mode descriptor for prompt building
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import mido

DEFAULT_MIDI = Path(r"C:\Users\skite\Downloads\depth_chords_microtonal.mid")
DEFAULT_DEPTHS = [200, 300, 400, 500, 700, 900, 1100]  # cm, in MIDI order
PITCH_BEND_RANGE_SEMITONES = 2.0
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_freq(midi_note: float) -> float:
    return 440.0 * 2 ** ((midi_note - 69) / 12)


def freq_to_midi(freq: float) -> float:
    import math

    return 12 * math.log2(freq / 440.0) + 69


def freq_to_pitch_class(freq: float) -> tuple[str, int, float]:
    """Return (note_name, midi_int_nearest, cents_offset) for octave-collapsed pitch class."""
    m = freq_to_midi(freq)
    nearest = round(m)
    cents = (m - nearest) * 100
    name = NOTE_NAMES[nearest % 12]
    return name, nearest, cents


def parse_midi_to_events(path: Path) -> list[dict]:
    mid = mido.MidiFile(path)
    tempo = 500000
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break
    sec_per_tick = (tempo / 1_000_000) / mid.ticks_per_beat

    events = []
    for track in mid.tracks:
        abs_t = 0
        bend = {}
        starts = {}
        for msg in track:
            abs_t += msg.time
            if msg.is_meta:
                continue
            if msg.type == "pitchwheel":
                bend[msg.channel] = msg.pitch / 8192.0
            elif msg.type == "note_on" and msg.velocity > 0:
                b = bend.get(msg.channel, 0.0)
                detuned = msg.note + b * PITCH_BEND_RANGE_SEMITONES
                starts[(msg.channel, msg.note)] = (abs_t, b, detuned, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in starts:
                    s_t, b, detuned, vel = starts.pop(key)
                    events.append(
                        {
                            "t_start_ticks": s_t,
                            "t_start_s": s_t * sec_per_tick,
                            "channel": msg.channel,
                            "midi_note": msg.note,
                            "bend_st": b * PITCH_BEND_RANGE_SEMITONES,
                            "midi_detuned": detuned,
                            "freq_hz": midi_to_freq(detuned),
                            "duration_ticks": abs_t - s_t,
                            "velocity": vel,
                        }
                    )
    events.sort(key=lambda e: e["t_start_ticks"])
    return events


def assign_to_depths(events: list[dict], depths: list[int], notes_per_depth: list[int]) -> dict[int, list[dict]]:
    """Slice the event list into per-depth blocks based on `notes_per_depth`."""
    if sum(notes_per_depth) != len(events):
        raise ValueError(
            f"notes_per_depth sums to {sum(notes_per_depth)} but there are {len(events)} events"
        )
    if len(notes_per_depth) != len(depths):
        raise ValueError(
            f"len(notes_per_depth)={len(notes_per_depth)} != len(depths)={len(depths)}"
        )
    out = {}
    cursor = 0
    for d, n in zip(depths, notes_per_depth):
        out[d] = events[cursor : cursor + n]
        cursor += n
    return out


def collapse_to_pitch_classes(freqs: list[float]) -> dict[str, float]:
    """Octave-collapse all freqs to one octave and return weighted pitch-class histogram."""
    pc = {n: 0.0 for n in NOTE_NAMES}
    for f in freqs:
        m = freq_to_midi(f)
        nearest = round(m)
        cents = abs((m - nearest) * 100)
        name = NOTE_NAMES[nearest % 12]
        # Weight: closer to integer pitch = higher weight (1 at 0¢, 0 at 50¢)
        w = max(0.0, 1.0 - cents / 50.0)
        pc[name] += w
    return pc


def describe_chord(freqs: list[float]) -> dict:
    """Build a music-theory descriptor of a chord from its freq list."""
    if not freqs:
        return {}
    fundamental = min(freqs)
    ratios = [f / fundamental for f in freqs]
    pc_hist = collapse_to_pitch_classes(freqs)
    # Top 3 pitch classes
    top_pc = sorted(pc_hist.items(), key=lambda kv: -kv[1])[:5]
    fund_name, fund_midi, fund_cents = freq_to_pitch_class(fundamental)
    # Approximate range in octaves
    import math

    n_oct = math.log2(max(freqs) / fundamental)
    return {
        "fundamental_hz": fundamental,
        "fundamental_pitch": f"{fund_name}{fund_midi // 12 - 1} ({fund_cents:+.0f}¢)",
        "n_partials": len(freqs),
        "freq_min": min(freqs),
        "freq_max": max(freqs),
        "span_octaves": n_oct,
        "harmonic_ratios": [round(r, 3) for r in sorted(ratios)],
        "top_pitch_classes": [(n, round(w, 2)) for n, w in top_pc],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--midi", default=str(DEFAULT_MIDI))
    p.add_argument(
        "--depths",
        default=",".join(map(str, DEFAULT_DEPTHS)),
        help="Comma-separated depths in MIDI order",
    )
    p.add_argument(
        "--notes-per-depth",
        default="10,10,10,10,10,10,7",
        help="Comma-separated note counts per depth (default 10×6 + 7×1 = 67)",
    )
    p.add_argument("--out", default="outputs/depth_harmonies.json")
    args = p.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
    npd = [int(x) for x in args.notes_per_depth.split(",")]

    events = parse_midi_to_events(Path(args.midi))
    print(f"Parsed {len(events)} note events from {args.midi}")
    print(f"Depth assignment: {dict(zip(depths, npd))}")
    print()

    grouped = assign_to_depths(events, depths, npd)

    summary = {}
    for d, evs in grouped.items():
        freqs = [e["freq_hz"] for e in evs]
        descr = describe_chord(freqs)
        summary[d] = {
            "depth_cm": d,
            "n_notes": len(evs),
            **descr,
            "raw_freqs": [round(f, 2) for f in freqs],
        }
        print(f"=== Depth {d} cm  ({len(evs)} notes)")
        print(
            f"   fundamental: {descr['fundamental_pitch']} = {descr['fundamental_hz']:.2f} Hz"
        )
        print(f"   freq range: {descr['freq_min']:.2f} – {descr['freq_max']:.2f} Hz  ({descr['span_octaves']:.1f} octaves)")
        print(f"   raw freqs:   {[round(f, 1) for f in freqs]}")
        print(f"   ratios/fund: {descr['harmonic_ratios']}")
        print(f"   top pitch classes: {descr['top_pitch_classes']}")
        print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
