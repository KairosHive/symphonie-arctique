import glob
import json
import tempfile
import uuid
from pathlib import Path

import requests
from pydub import AudioSegment

API_URL = "http://127.0.0.1:8188/prompt"

SOURCE_CUT_TIMES = (0, 0)  # (start, end) for source audio
TARGET_CUT_TIMES = (270, 445)  # (start, end) for target audio

DENOISE_STRENGTH = 0.3
SONG_RATIO = 0.3  # ratio of song 1 vs song 2
PROMPT = "clean audio, minimal noise, clear harmonics, high fidelity, professional mix, studio quality, beautiful harmonics, interesting composition"


def cut_audio_if_needed(audio_path, start, end):
    if start is not None and start > 0 and end is not None and end > 0:
        audio = AudioSegment.from_wav(str(audio_path))
        cut_audio = audio[start * 1000 : end * 1000]
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        cut_audio.export(temp_file.name, format="wav")
        return Path(temp_file.name)
    else:
        return audio_path


def submit_audio_job(path_a, path_b):
    path_a_orig, path_b_orig = Path(path_a), Path(path_b)

    path_a = cut_audio_if_needed(path_a_orig, SOURCE_CUT_TIMES[0], SOURCE_CUT_TIMES[1])
    path_b = cut_audio_if_needed(path_b_orig, TARGET_CUT_TIMES[0], TARGET_CUT_TIMES[1])

    with open("workflow_api.json", "r") as f:
        prompt = json.load(f)

    prompt["112"]["inputs"]["path"] = str(path_a.absolute())
    prompt["105"]["inputs"]["path"] = str(path_b.absolute())

    # Include start and end in filename
    filename_prefix = f"IceAudioMix_rat{SONG_RATIO}_dns{DENOISE_STRENGTH}_sf{SOURCE_CUT_TIMES[0]}-st{SOURCE_CUT_TIMES[1]}_tf{TARGET_CUT_TIMES[0]}-tt{TARGET_CUT_TIMES[1]}/{path_a_orig.stem}--{path_b_orig.stem}"

    prompt["59"]["inputs"]["filename_prefix"] = filename_prefix
    prompt["76"]["inputs"]["ratio"] = SONG_RATIO
    prompt["52"]["inputs"]["denoise"] = DENOISE_STRENGTH

    r = requests.post(API_URL, json={"prompt": prompt, "client_id": str(uuid.uuid4())})
    r.raise_for_status()
    print("Submitted:", path_a, path_b, "->", r.status_code)


if __name__ == "__main__":
    files_a = glob.glob("source/*.wav")
    files_b = glob.glob("target/*.wav")

    for a in files_a:
        for b in files_b:
            submit_audio_job(a, b)
