# CLAP_extractor.py
# ------------------------------------------------------------
# For each labeled section in an audio file, split into 1 s chunks,
# embed each chunk with CLAP, compare to text descriptor dictionaries,
# and write JSON with top-N descriptors per dictionary per chunk.
#
# Defaults (relative to this script):
#   ../assets/song_joey.mp3
#   ../assets/timeline_joey.json
#   ../assets/descriptors.json
# Output:
#   ./results/<auto-named>.json (folder auto-created) unless --out is provided
#
# Quick start (uses defaults):
#   python CLAP_extractor.py
#
# Override inputs:
#   python CLAP_extractor.py --audio ../assets/other.mp3 --sections ../assets/timeline_joey.json --dict ../assets/descriptors.json
# ------------------------------------------------------------

import argparse
import json
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import torch
except Exception as e:
        raise RuntimeError("PyTorch is required. Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121 (or cpu)") from e

try:
    import librosa
except Exception as e:
    raise RuntimeError("librosa is required. Install with: pip install librosa soundfile") from e

try:
    from transformers import ClapProcessor, ClapModel
except Exception as e:
    raise RuntimeError("transformers (with CLAP) is required. Install with: pip install transformers accelerate safetensors") from e

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


# ---------------- Paths & defaults ----------------

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = (SCRIPT_DIR / ".." / "assets").resolve()

DEFAULT_AUDIO       = ASSETS_DIR / "song_joey.mp3"
DEFAULT_SECTIONS    = ASSETS_DIR / "timeline_joey.json"
DEFAULT_DESCRIPTORS = ASSETS_DIR / "descriptors.json"
DEFAULT_RESULTS_DIR = (SCRIPT_DIR / "../results").resolve()


# ---------------- Tolerant JSON helpers ----------------

_COMMENT_RE = re.compile(
    r"""
    (//[^\n\r]*?$)     |   # // line comments
    (/\*.*?\*/)            # /* block comments */
    """,
    re.MULTILINE | re.DOTALL | re.VERBOSE,
)
# Remove trailing commas before } or ]
_TRAIL_RE = re.compile(r",\s*(?=[}\]])")

def _read_json_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")  # strips BOM if present
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")

def _json_tolerant_load(text: str):
    # Try strict first
    try:
        return json.loads(text)
    except JSONDecodeError:
        pass
    # Strip comments
    text2 = _COMMENT_RE.sub("", text)
    try:
        return json.loads(text2)
    except JSONDecodeError:
        # Heal trailing commas (["a","b",] or {"k":1,})
        text3 = _TRAIL_RE.sub("", text2)
        return json.loads(text3)


# ---------------- I/O loaders ----------------

def load_sections(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Sections JSON not found: {path}")
    raw = _read_json_text(path)
    if not raw.strip():
        raise ValueError(f"Sections JSON is empty: {path}")
    try:
        data = _json_tolerant_load(raw)
    except Exception as e:
        preview = raw[:200].replace("\n", "\\n")
        raise ValueError(f"Failed to parse sections JSON in {path} ({e}). First 200 chars: {preview}")
    if "sections" not in data or not isinstance(data["sections"], list):
        raise ValueError("Sections JSON must contain key 'sections' as a list.")
    out = []
    for s in data["sections"]:
        name = str(s.get("name", "")).strip()
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        if not name:
            raise ValueError("Each section must have a non-empty 'name'.")
        if end <= start:
            # skip invalid / zero-length
            continue
        out.append({"name": name, "start": start, "end": end})
    if not out:
        raise ValueError("No valid sections found in the JSON.")
    return out

def load_descriptor_files(paths: List[Path]) -> Dict[str, List[str]]:
    """
    Supports either:
      • {"name": "DictName", "descriptors": [...]}
      • {"DictA": [...], "DictB": [...]}
    Returns: {dict_name: [descriptor, ...], ...}
    """
    if not paths:
        raise ValueError("No descriptor files provided.")
    merged: Dict[str, List[str]] = {}
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Descriptor file not found: {p}")
        if p.is_dir():
            raise IsADirectoryError(f"Expected a JSON file but got a directory: {p}")
        raw = _read_json_text(p)
        if not raw.strip():
            raise ValueError(f"Descriptor file is empty: {p}")
        try:
            data = _json_tolerant_load(raw)
        except Exception as e:
            preview = raw[:200].replace("\n", "\\n")
            raise ValueError(f"Failed to parse JSON in {p} ({e}). First 200 chars: {preview}")

        if isinstance(data, dict) and "name" in data and "descriptors" in data:
            name = str(data["name"]).strip()
            descs = [str(x).strip() for x in data["descriptors"] if str(x).strip()]
            merged.setdefault(name, []).extend(descs)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    name = str(k).strip()
                    descs = [str(x).strip() for x in v if str(x).strip()]
                    merged.setdefault(name, []).extend(descs)
        else:
            raise ValueError(f"Unrecognized descriptor schema in {p}: root type is {type(data).__name__}")

    # Deduplicate
    for k, vals in list(merged.items()):
        seen, uniq = set(), []
        for d in vals:
            if d not in seen:
                uniq.append(d); seen.add(d)
        merged[k] = uniq

    if not merged:
        raise ValueError(f"No descriptors loaded from: {', '.join(str(p) for p in paths)}")
    return merged


# ---------------- CLAP embedder ----------------

class ClapEmbedder:
    """
    Thin wrapper around CLAP for text + audio embeddings with L2 normalization.
    """
    def __init__(self, model_id: str = "laion/clap-htsat-fused", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id).to(self.device).eval()
        self.target_sr = 48000

    @torch.no_grad()
    def embed_text(self, texts: List[str]) -> np.ndarray:
        tin = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        tin = {k: v.to(self.device) for k, v in tin.items()}
        z = self.model.get_text_features(**tin)  # [B, D]
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        return z.cpu().float().numpy()

    @torch.no_grad()
    def embed_audio_batch(self, waves: List[np.ndarray], orig_sr: int) -> np.ndarray:
        # Resample to 48k if needed
        batch = []
        for w in waves:
            if orig_sr != self.target_sr:
                w = librosa.resample(w, orig_sr=orig_sr, target_sr=self.target_sr, res_type="kaiser_fast")
            w = np.asarray(w, dtype=np.float32)
            w = np.clip(w, -1.0, 1.0)
            batch.append(w)
        ain = self.processor(audios=batch, sampling_rate=self.target_sr, return_tensors="pt", padding=True)
        ain = {k: v.to(self.device) for k, v in ain.items()}
        z = self.model.get_audio_features(**ain)  # [B, D]
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        return z.cpu().float().numpy()


# ---------------- Core logic ----------------

def split_into_chunks(
    y: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    chunk_s: float,
    keep_last_partial: bool = False,
    min_partial_s: float = 0.2
) -> List[Tuple[float, float, np.ndarray]]:
    """
    Returns list of (chunk_start_sec, chunk_end_sec, audio_array) within [start_s, end_s).
    If keep_last_partial is True, pads last short chunk with zeros if >= min_partial_s.
    """
    start_idx = int(round(start_s * sr))
    end_idx = int(round(end_s * sr))
    end_idx = min(end_idx, len(y))
    if end_idx <= start_idx:
        return []

    seg = y[start_idx:end_idx]
    chunk_len = int(round(chunk_s * sr))
    chunks = []
    n_full = len(seg) // chunk_len
    for i in range(n_full):
        a = i * chunk_len
        b = a + chunk_len
        c0 = start_s + (a / sr)
        c1 = start_s + (b / sr)
        chunks.append((c0, c1, seg[a:b].copy()))

    # partial
    rem = len(seg) - n_full * chunk_len
    if rem > 0 and keep_last_partial:
        dur = rem / sr
        if dur >= min_partial_s:
            a = n_full * chunk_len
            short = seg[a:]
            pad = np.zeros(chunk_len, dtype=seg.dtype)
            pad[:len(short)] = short
            c0 = start_s + (a / sr)
            c1 = c0 + chunk_s  # nominal end after padding
            chunks.append((c0, c1, pad))
    return chunks

def cosine_topn(audio_vec: np.ndarray, text_mat: np.ndarray, labels: List[str], topn: int) -> List[Tuple[str, float]]:
    # audio_vec and rows of text_mat are already L2-normalized => cosine = dot
    sims = text_mat @ audio_vec  # [M]
    idx = np.argsort(-sims)[:topn]
    return [(labels[i], float(sims[i])) for i in idx]

def _auto_out_path(results_dir: Path, audio: Path, sections: Path, chunk_s: float, topn: int) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ctag = str(chunk_s).replace(".", "p")
    fname = f"{audio.stem}__{sections.stem}__chunk{ctag}s__top{topn}.json"
    return results_dir / fname


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="CLAP: top-N descriptors per dictionary, per chunk, per section."
    )
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help=f"Audio file path (default: {DEFAULT_AUDIO})")
    ap.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS, help=f"Sections JSON (default: {DEFAULT_SECTIONS})")
    ap.add_argument("--dict", nargs="+", type=Path, dest="dicts", default=[DEFAULT_DESCRIPTORS],
                    help=f"One or more descriptor JSON files (default: {DEFAULT_DESCRIPTORS})")
    ap.add_argument("--out", type=Path, default=None, help="Explicit output JSON path. If omitted, auto-named under --results-dir.")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help=f"Directory for outputs when --out not set (default: {DEFAULT_RESULTS_DIR})")
    ap.add_argument("--chunk", type=float, default=1.0, help="Chunk length in seconds (default: 1.0)")
    ap.add_argument("--topn", type=int, default=5, help="Top-N descriptors per dictionary (default: 5)")
    ap.add_argument("--keep-last-partial", action="store_true", help="Pad & keep last partial chunk if >= min_partial_s.")
    ap.add_argument("--min-partial-s", type=float, default=0.2, help="Minimum seconds to keep partial chunk (default: 0.2)")
    ap.add_argument("--clap", type=str, default="laion/clap-htsat-fused", help="HF model id for CLAP.")
    ap.add_argument("--device", type=str, default=None, help="Force device: cuda | cpu")
    args = ap.parse_args()

    out_path = args.out if args.out is not None else _auto_out_path(args.results_dir, args.audio, args.sections, args.chunk, args.topn)

    # Basic info
    print("[INFO] Audio:", args.audio)
    print("[INFO] Sections:", args.sections)
    print("[INFO] Descriptor files:", ", ".join(str(p) for p in args.dicts))
    print("[INFO] Output:", out_path)
    if args.device is None and not torch.cuda.is_available():
        print("[INFO] Running on CPU (set --device cuda if you have a GPU).")

    # Load inputs
    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    y, sr = librosa.load(str(args.audio), sr=None, mono=True)  # keep native SR; resample inside embedder
    sections = load_sections(args.sections)
    desc_dicts = load_descriptor_files(args.dicts)

    # Init CLAP
    embedder = ClapEmbedder(model_id=args.clap, device=args.device)

    # Pre-embed all descriptors per dictionary
    dict_embeddings: Dict[str, np.ndarray] = {}
    dict_labels: Dict[str, List[str]] = {}
    for dname, descs in desc_dicts.items():
        if len(descs) == 0:
            continue
        Z = embedder.embed_text(descs)
        dict_embeddings[dname] = Z.astype(np.float32)
        dict_labels[dname] = descs

    # Process sections
    results = {
        "audio": str(args.audio),
        "sections_file": str(args.sections),
        "descriptor_files": [str(p) for p in args.dicts],
        "sample_rate": sr,
        "chunk_s": args.chunk,
        "topn": args.topn,
        "sections": []
    }

    outer_iter = sections if not _HAS_TQDM else tqdm(sections, desc="Sections")
    for sec in outer_iter:
        name = sec["name"]
        start_s = float(sec["start"])
        end_s = float(sec["end"])
        chunks = split_into_chunks(
            y, sr, start_s, end_s, args.chunk,
            keep_last_partial=args.keep_last_partial,
            min_partial_s=args.min_partial_s
        )

        waves = [c[2] for c in chunks]
        if len(waves) == 0:
            results["sections"].append({"name": name, "start": start_s, "end": end_s, "chunks": []})
            continue

        # Batch-embed chunks with CLAP
        A = embedder.embed_audio_batch(waves, orig_sr=sr)  # [N, D], L2-normalized

        sec_out = {"name": name, "start": start_s, "end": end_s, "chunks": []}
        for (c0, c1, _), avec in zip(chunks, A):
            chunk_entry = {"chunk_start": c0, "chunk_end": c1, "matches": {}}
            for dname, Z in dict_embeddings.items():
                labels = dict_labels[dname]
                top = cosine_topn(avec, Z, labels, args.topn)
                chunk_entry["matches"][dname] = [{"descriptor": t[0], "score": t[1]} for t in top]
            sec_out["chunks"].append(chunk_entry)
        results["sections"].append(sec_out)

    # Write JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
