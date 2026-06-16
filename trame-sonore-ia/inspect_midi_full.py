"""
Full MIDI message dump — every message type, every track, with timing.
Look for CC, program_change, sysex, meta events that might delimit depths.
"""
from collections import Counter
from pathlib import Path
import mido

MIDI = Path(__file__).parent / "data" / "depth_chords_microtonal.mid"
mid = mido.MidiFile(MIDI)

print(f"Type {mid.type}, ticks_per_beat {mid.ticks_per_beat}, tracks {len(mid.tracks)}")
print()

for ti, track in enumerate(mid.tracks):
    print(f"=== Track {ti}: {track.name!r}  msgs={len(track)}")
    types = Counter(msg.type for msg in track)
    print(f"  Message-type counts: {dict(types)}")
    print()

    # Print every non-pitchwheel, non-note message in detail (likely structural markers)
    print("  Non-note, non-pitchwheel messages:")
    abs_t = 0
    rare_count = 0
    for msg in track:
        abs_t += msg.time
        if msg.type in {"note_on", "note_off", "pitchwheel"}:
            continue
        rare_count += 1
        print(f"    t={abs_t:>8}  {msg!r}")
    if rare_count == 0:
        print("    (none)")
    print()

    # Also: per-channel program/CC state if any
    print("  Per-channel notes (count):")
    ch_counts = Counter()
    for msg in track:
        if msg.type == "note_on" and msg.velocity > 0:
            ch_counts[msg.channel] += 1
    for ch in sorted(ch_counts):
        print(f"    ch{ch:>2}: {ch_counts[ch]} notes")
