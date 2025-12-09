#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan a folder for images (recursively) and, for each image, ask Gemma3 (via Ollama)
to produce a detailed technical caption for training image models.

Usage:
  python img2caption_generator.py \
    --root /path/to/images \
    --out captions.jsonl \
    --model gemma3:8b \
    --trigger iceviscous

Optional:
  --sidecar            # also write caption.txt next to each image
  --extensions jpg,jpeg,png,webp,bmp,tif,tiff,gif
  --temperature 0.2 --top-p 0.9 --num-predict 180 --seed 0
  --skip-existing      # resume: skip images already present in out file
  --quiet
"""

from __future__ import annotations

import argparse, base64, json, os, sys, time
from pathlib import Path
from typing import Iterable, Set, Dict

# ---------- tiny logger ----------
def _ts() -> str:
    return time.strftime("%H:%M:%S")

def vprint(enabled: bool, *a, **k):
    if enabled:
        print(f"[{_ts()}]", *a, **k)

# ---------- ollama wrapper ----------
def _require_ollama():
    try:
        import ollama  # noqa: F401
    except Exception as e:
        print("ERROR: ollama not installed. `pip install ollama` (and run the Ollama daemon).", file=sys.stderr)
        raise

def _init_client(timeout_s: int = 600, host: str = None):
    import ollama
    if host:
        return ollama.Client(host=host, timeout=timeout_s)
    return ollama.Client(timeout=timeout_s)


# ---------- utils ----------
DEFAULT_EXTS = ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif")

def is_image_file(p: Path, exts: Set[str]) -> bool:
    return p.is_file() and p.suffix.lower().lstrip(".") in exts

def iter_images(root: Path, exts: Set[str]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if is_image_file(p, exts):
            yield p

def normalize_out_path(p: Path) -> Path:
    # If user passed a directory (like "."), write to a default filename inside it
    if str(p).strip() in (".", "..") or p.is_dir():
        return p / "captions.jsonl"
    # If no extension, default to .jsonl
    if p.suffix.lower() not in (".jsonl", ".json"):
        return p.with_suffix(".jsonl")
    return p


def load_processed_set(jsonl_path: Path) -> Set[str]:
    seen = set()
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "image" in obj:
                        seen.add(str(obj["image"]))
                except Exception:
                    continue
    return seen

def create_system_prompt(trigger: str) -> str:
    return (
        f"You analyze images and create detailed technical captions for training image generation models. "
        f"Each caption must begin with the trigger word '{trigger}' followed by a comprehensive description. "
        f"Guidelines:\n"
        f"- Start with '{trigger}, ' (trigger word + comma + space)\n"
        f"- Describe the subject, composition, and technical details\n"
        f"- Include: framing (macro/close-up/wide), subject details, materials/textures, focus/depth-of-field, "
        f"aperture (f-stop), lighting conditions (quality, angle, color temperature), visual effects (highlights, shadows, contrast)\n"
        f"- Use precise, technical photography/cinematography language\n"
        f"- Length: 25-50 words total\n"
        f"- Write as a single flowing sentence or two connected sentences"
    )

USER_INSTRUCTION = (
    "Analyze this image and produce a detailed technical caption following the specified format. "
    "Return only structured JSON as specified."
)

# JSON schema for structured output (Ollama structured outputs)
def create_output_schema(trigger: str) -> Dict:
    return {
        "type": "object",
        "properties": {
            "caption": {
                "type": "string",
                "description": f"A detailed technical caption starting with '{trigger}, ' (25-50 words total)"
            }
        },
        "required": ["caption"],
        "additionalProperties": False,
    }

def ask_model_for_caption(
        *,
        client,
        model: str,
        image_path: Path,
        trigger: str,
        temperature: float,
        top_p: float,
        num_predict: int,
        seed: int,
        verbose: bool,
    ) -> str:

    messages = [
        {"role": "system", "content": create_system_prompt(trigger)},
        {"role": "user", "content": USER_INSTRUCTION, "images": [str(image_path)]},
    ]
    options = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "num_predict": int(num_predict),
        "seed": int(seed),
    }

    resp = client.chat(
        model=model,
        messages=messages,
        options=options,
        format=create_output_schema(trigger),   # enforce {"caption": "..."}
        stream=False,
        keep_alive="20m",
    )

    # ---- Extract text content safely from both dict and ChatResponse shapes ----
    content = ""
    try:
        if isinstance(resp, dict):
            msg = resp.get("message", {})
            if isinstance(msg, dict):
                content = (msg.get("content") or "").strip()
            else:
                content = (resp.get("response") or "").strip()
        else:
            # ChatResponse-like object
            msg = getattr(resp, "message", None)
            if isinstance(msg, dict):
                content = (msg.get("content") or "").strip()
            elif hasattr(msg, "content"):
                content = (msg.content or "").strip()
            else:
                # last resort: try common attr
                content = (getattr(resp, "response", "") or "").strip()
    except Exception:
        content = ""

    # ---- Parse JSON (schema response should be a JSON object with "caption") ----
    if content:
        # try direct JSON
        try:
            data = json.loads(content)
            if isinstance(data, dict) and isinstance(data.get("caption"), str):
                return data["caption"].strip()
        except Exception:
            # try to find JSON substring
            try:
                s, e = content.find("{"), content.rfind("}") + 1
                if s != -1 and e > s:
                    data = json.loads(content[s:e])
                    if isinstance(data, dict) and isinstance(data.get("caption"), str):
                        return data["caption"].strip()
            except Exception:
                pass

    # If the model ignored the schema, return raw text (still useful)
    return (content or "").strip().strip('"')


def write_jsonl_line(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_caption_file(img_path: Path, text: str):
    """Write caption.txt next to the image (same name, .txt extension)"""
    caption_file = img_path.with_suffix(".txt")
    caption_file.write_text(text, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Generate detailed captions from images using Gemma3 (Ollama).")
    ap.add_argument("--root", type=Path, required=True, help="Root folder to scan for images (recursive).")
    ap.add_argument("--out", type=Path, default=Path("captions.jsonl"), help="Output JSONL file (one result per line).")
    ap.add_argument("--model", type=str, default="gemma3:8b", help="Ollama model id (e.g., 'gemma3:8b').")
    ap.add_argument("--trigger", type=str, required=True, help="Trigger word to start each caption (e.g., 'iceviscous').")
    ap.add_argument("--extensions", type=str, default=",".join(DEFAULT_EXTS),
                    help=f"Comma-separated image extensions (default: {','.join(DEFAULT_EXTS)})")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--num-predict", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true", help="Resume: skip images already present in --out.")
    ap.add_argument("--sidecar", action="store_true", help="Also write 'caption.txt' next to each image.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    ap.add_argument("--max", type=int, default=0, help="Process at most N images (0 = all).")
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args()
    args.out = normalize_out_path(args.out)
    verbose = not args.quiet

    # Validate Ollama
    _require_ollama()
    client = _init_client()

    if not args.root.exists():
        print(f"ERROR: root path not found: {args.root}", file=sys.stderr)
        sys.exit(1)

    exts = set([e.strip().lower().lstrip(".") for e in args.extensions.split(",") if e.strip()])
    images = list(iter_images(args.root, exts))
    if args.max > 0:
        images = images[:args.max]
    vprint(verbose, f"Found {len(images)} images under {args.root}")

    processed = load_processed_set(args.out) if args.skip_existing else set()
    if processed:
        vprint(verbose, f"Resuming: {len(processed)} already in {args.out}")

    if not args.out.exists():
        vprint(verbose, f"Creating {args.out}")

    count_ok, count_err = 0, 0

    for idx, img in enumerate(images, start=1):
        img_str = str(img)
        image_name = img.name
        try:
            image_rel = os.path.relpath(img, args.root)
        except Exception:
            image_rel = image_name  # fallback

        if args.skip_existing and img_str in processed:
            vprint(verbose, f"[{idx}/{len(images)}] SKIP  {img_str}")
            continue

        vprint(verbose, f"[{idx}/{len(images)}] PROC  {img_str}")
        try:
            caption = ask_model_for_caption(
                client=client,
                model=args.model,
                image_path=img,
                trigger=args.trigger,
                temperature=args.temperature,
                top_p=args.top_p,
                num_predict=args.num_predict,
                seed=args.seed + idx,
                verbose=verbose,
            )

            rec = {
                "image": img_str,           # keep for backward-compat/resume
                "image_name": image_name,   # filename only
                "image_rel": image_rel,     # path relative to --root
                "model": args.model,
                "trigger": args.trigger,
                "caption": caption,
                "timestamp": int(time.time()),
                "params": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "num_predict": args.num_predict,
                    "seed": args.seed + idx,
                },
            }
            write_jsonl_line(args.out, rec)
            if args.sidecar:
                write_caption_file(img, caption)
            count_ok += 1

        except KeyboardInterrupt:
            vprint(True, "Interrupted. Partial results preserved.")
            break
        except Exception as e:
            err = {"image": img_str, "error": str(e), "timestamp": int(time.time())}
            write_jsonl_line(args.out, err)
            vprint(verbose, f"  ERROR: {e}")
            count_err += 1

        if args.sleep > 0:
            time.sleep(args.sleep)

    vprint(True, f"Done. OK: {count_ok} | ERR: {count_err} | Out: {args.out}")

if __name__ == "__main__":
    main()
