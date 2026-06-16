"""
Parse the depth_chords_microtonal.mid file. Show: tracks, all note events with
their actual (microtonally-detuned) frequencies, and try to detect groupings.
"""
from pathlib import Path
import mido

MIDI_PATH = Path(r"C:\Users\skite\Downloads\depth_chords_microtonal.mid")
PITCH_BEND_RANGE_SEMITONES = 2.0


def midi_to_freq(midi_note: float) -> float:
    return 440.0 * 2 ** ((midi_note - 69) / 12)


def midi_note_to_name(midi_note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi_note % 12]}{midi_note // 12 - 1}"


mid = mido.MidiFile(MIDI_PATH)
print(f"=== File: {MIDI_PATH}")
print(f"Type: {mid.type}  ticks_per_beat: {mid.ticks_per_beat}  tracks: {len(mid.tracks)}")
print(f"Length: {mid.length:.2f}s")

# Determine tempo (microseconds per beat). Default 500000 (= 120 BPM) if not set.
tempo = 500000
for track in mid.tracks:
    for msg in track:
        if msg.type == "set_tempo":
            tempo = msg.tempo
            break
print(f"Tempo: {tempo} us/beat = {60_000_000/tempo:.1f} BPM")
seconds_per_tick = (tempo / 1_000_000) / mid.ticks_per_beat
print(f"Seconds per tick: {seconds_per_tick:.6f}")
print()

# Collect every note-on with its detuned frequency and timing
events = []  # (abs_time_ticks, abs_time_s, channel, midi_note, bend, freq, duration_ticks)
channel_bend = {}
note_starts = {}  # (channel, note) -> abs_time_ticks
abs_time = 0
for track in mid.tracks:
    abs_time = 0
    channel_bend = {}
    for msg in track:
        abs_time += msg.time
        if msg.is_meta:
            continue
        if msg.type == "pitchwheel":
            channel_bend[msg.channel] = msg.pitch / 8192.0
        elif msg.type == "note_on" and msg.velocity > 0:
            bend = channel_bend.get(msg.channel, 0.0)
            detuned = msg.note + bend * PITCH_BEND_RANGE_SEMITONES
            freq = midi_to_freq(detuned)
            note_starts[(msg.channel, msg.note)] = (abs_time, bend, freq, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in note_starts:
                start_t, bend, freq, vel = note_starts.pop(key)
                events.append((start_t, start_t * seconds_per_tick, msg.channel, msg.note, bend, freq, abs_time - start_t, vel))

events.sort()

print(f"=== All {len(events)} notes (in time order)")
print(f"{'idx':>4} {'t_ticks':>8} {'t_s':>7} {'ch':>3} {'midi_name':>10} {'bend(st)':>10} {'freq(Hz)':>10} {'dur_ticks':>10}")
for i, (t, ts, ch, n, bend, f, dur, vel) in enumerate(events):
    print(f"{i:>4} {t:>8} {ts:>7.3f} {ch:>3} {midi_note_to_name(n):>10} {bend*PITCH_BEND_RANGE_SEMITONES:>+10.3f} {f:>10.2f} {dur:>10}")

# Try: assume groups of consecutive notes per depth.
# 11 usable depths × N = 67 ?  no  (no clean integer)
# 14 depth columns × ~4.8 = 67 ? no
# 67 itself is prime.
# Maybe groups of 6 or 7 ?

print()
print("=== Frequency stats")
freqs = sorted(f for *_, f, _, _ in events)
import statistics
print(f"min={freqs[0]:.2f}Hz  max={freqs[-1]:.2f}Hz  median={statistics.median(freqs):.2f}Hz")
print()

# Detect inter-event gaps to spot grouping (e.g., a longer gap might separate depths)
print("=== Inter-event gaps (ticks) — look for outliers that may delimit groups")
prev_t = 0
gaps = []
for i, (t, *_rest) in enumerate(events):
    if i == 0:
        prev_t = t
        continue
    gap = t - prev_t
    gaps.append(gap)
    prev_t = t
import collections
ctr = collections.Counter(gaps)
print(f"Most common gaps: {ctr.most_common(10)}")
unique_gaps = sorted(set(gaps))
print(f"Unique gap sizes: {unique_gaps}")
