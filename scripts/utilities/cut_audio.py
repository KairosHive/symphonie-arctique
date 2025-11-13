from pydub import AudioSegment

# --- PARAMETERS ---
input_path = "../assets/song_simon.wav"    # path to input file
output_path = "../assets/song_simon_cut.wav"  # path to save the cut file
start_time = (3*60)+26  # in seconds
end_time = (7*60)+25    # in seconds
# ------------------

# Load the audio file
audio = AudioSegment.from_wav(input_path)

# Convert times to milliseconds (pydub uses ms)
start_ms = start_time * 1000
end_ms = end_time * 1000

# Cut the segment
cut_audio = audio[start_ms:end_ms]

# Export the result
cut_audio.export(output_path, format="wav")

print(f"Saved cut segment from {start_time}s to {end_time}s as '{output_path}'")
