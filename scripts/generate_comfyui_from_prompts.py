#!/usr/bin/env python3
"""
Generate images from prompts using a saved graph against a local ComfyUI instance.

Usage (Windows cmd.exe):
  python scripts\generate_comfyui_from_prompts.py

Defaults:
 - prompts: assets\timeline_joey_prompts.json
 - graph: assets\Flux-dev_textures_api.json
 - host: localhost
 - port: 8188
 - outdir: outputs/joey
 - start-seed: 1
 - seed-step: 1
 - width/height: 1280x720
 - serial execution (one request at a time)

Notes:
 - The script will attempt a few common ComfyUI graph endpoints to POST the graph JSON.
 - It will mutate prompt-like keys (clip_l, t5xxl, prompt, text, caption) and numeric seed keys
   (noise_seed, seed) inside the graph JSON before sending each request.
 - If the response contains base64-encoded images in a common structure (e.g. "images" list),
   those images will be saved to section subfolders.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("generate_comfyui")


# Keys that likely contain prompt text inside graph JSON nodes.
PROMPT_KEYS = {"prompt", "clip_l", "t5xxl", "text", "caption", "prompt_text", "prompt_raw"}
# Keys that likely contain seeds / noise seeds
SEED_KEYS = {"seed", "noise_seed", "manual_seed"}
# Keys for image size
WIDTH_KEYS = {"width", "image_width", "w"}
HEIGHT_KEYS = {"height", "image_height", "h"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(data)


def looks_like_b64_image(s: str) -> bool:
    # very small heuristic: common PNG/JPEG signatures in base64
    if not isinstance(s, str) or len(s) < 100:
        return False
    return s.startswith("iVBOR") or s.startswith("/9j/") or s.startswith("R0lGOD") or s.startswith("Qk")  # PNG/JPEG/GIF/BMP-ish


def find_and_replace_prompts_in_obj(obj: Any, prompt: str) -> int:
    """
    Walk the graph JSON object and replace prompt-like string values.
    Returns the number of replacements done.
    """
    count = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in PROMPT_KEYS and isinstance(v, str):
                obj[k] = prompt
                count += 1
            else:
                count += find_and_replace_prompts_in_obj(v, prompt)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            # Sometimes prompts are stored inside single-item lists, e.g. ["some text"]
            if isinstance(v, str) and len(obj) == 1 and any(k in PROMPT_KEYS for k in PROMPT_KEYS):
                # don't do this blanket replacement; prefer keyed replacements above.
                pass
            count += find_and_replace_prompts_in_obj(v, prompt)
    return count


def set_seed_in_obj(obj: Any, seed: int) -> int:
    """
    Walk graph JSON and replace numeric seed values for known seed keys.
    Returns the number of replacements done.
    """
    count = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SEED_KEYS and (isinstance(v, int) or isinstance(v, float) or (isinstance(v, str) and v.isdigit())):
                obj[k] = int(seed)
                count += 1
            else:
                count += set_seed_in_obj(v, seed)
    elif isinstance(obj, list):
        for v in obj:
            count += set_seed_in_obj(v, seed)
    return count


def set_size_in_obj(obj: Any, w: int, h: int) -> int:
    """
    Walk graph JSON and set width/height for matching keys.
    Returns the number of replacements done.
    """
    count = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in WIDTH_KEYS and isinstance(v, (int, float)):
                obj[k] = int(w)
                count += 1
            elif k in HEIGHT_KEYS and isinstance(v, (int, float)):
                obj[k] = int(h)
                count += 1
            else:
                count += set_size_in_obj(v, w, h)
    elif isinstance(obj, list):
        for v in obj:
            count += set_size_in_obj(v, w, h)
    return count


def extract_images_from_response(resp: requests.Response) -> List[bytes]:
    """
    Attempt to extract base64 images from common JSON response structures.
    Returns a list of raw bytes for each found image.
    """
    images: List[bytes] = []

    # Try JSON
    try:
        data = resp.json()
    except Exception:
        data = None

    def walk_and_collect(o: Any):
        if o is None:
            return
        if isinstance(o, dict):
            # common ComfyUI returns "images": ["<b64>", ...]
            if "images" in o and isinstance(o["images"], list):
                for item in o["images"]:
                    if isinstance(item, str) and looks_like_b64_image(item):
                        images.append(base64.b64decode(item))
            # some variants embed base64 in "artifacts" or "files"
            if "artifacts" in o and isinstance(o["artifacts"], list):
                for art in o["artifacts"]:
                    if isinstance(art, dict):
                        for v in art.values():
                            if isinstance(v, str) and looks_like_b64_image(v):
                                images.append(base64.b64decode(v))
            for v in o.values():
                walk_and_collect(v)
        elif isinstance(o, list):
            for v in o:
                if isinstance(v, str) and looks_like_b64_image(v):
                    images.append(base64.b64decode(v))
                else:
                    walk_and_collect(v)

    if data:
        walk_and_collect(data)

    # Fallback: if body looks like a single image (rare)
    if not images:
        content = resp.content or b""
        # crude check: PNG/JPEG signatures
        if content.startswith(b"\x89PNG") or content.startswith(b"\xff\xd8\xff"):
            images.append(content)

    return images


def try_post_graph(host: str, port: int, graph_payload: Dict[str, Any], timeout: int = 300) -> Optional[requests.Response]:
    """
    Try a list of likely endpoints for ComfyUI. Returns the first successful Response or None.
    """
    base = f"http://{host}:{port}"
    candidate_paths = [
        "/api/process_graph",
        "/api/graph/process_graph",
        "/api/graph/run_graph",
        "/api/graph",
        "/api/graph/run",
    ]
    headers = {"Content-Type": "application/json"}
    for path in candidate_paths:
        url = base + path
        try:
            logger.info("Posting graph to %s ...", url)
            resp = requests.post(url, json=graph_payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                logger.info("Got 200 from %s", url)
                return resp
            else:
                logger.warning("Non-200 from %s: %s", url, resp.status_code)
        except requests.RequestException as exc:
            logger.debug("Request to %s failed: %s", url, exc)
    logger.error("All endpoint attempts failed.")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run saved graph against ComfyUI for each prompt.")
    parser.add_argument("--prompts", type=Path, default=Path("assets/timeline_joey_prompts.json"))
    parser.add_argument("--graph", type=Path, default=Path("assets/Flux-dev_textures_api.json"))
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/joey"))
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--wait", type=float, default=0.5, help="Wait seconds between requests (serial mode).")
    args = parser.parse_args()

    if not args.prompts.exists():
        logger.error("Prompts file not found: %s", args.prompts)
        return 2
    if not args.graph.exists():
        logger.error("Graph file not found: %s", args.graph)
        return 2

    prompts_json = load_json(args.prompts)
    graph_template = load_json(args.graph)

    # Where to look for sections
    sections = prompts_json.get("sections") if isinstance(prompts_json, dict) else None
    if not sections:
        logger.error("No 'sections' key found in prompts file or file malformed.")
        return 3

    out_root = args.outdir
    out_root.mkdir(parents=True, exist_ok=True)

    global_prompt_index = 0
    seed = args.start_seed

    for section in sections:
        sec_name = section.get("name") or f"section_{sections.index(section):02d}"
        prompts = section.get("prompts") or []
        sec_folder = out_root / sec_name
        sec_folder.mkdir(parents=True, exist_ok=True)
        logger.info("Processing section '%s' with %d prompts -> %s", sec_name, len(prompts), sec_folder)

        for idx, prompt in enumerate(prompts, start=1):
            logger.info("Prompt %d (global %d): %s", idx, global_prompt_index + 1, prompt[:100])
            # Prepare graph payload per-prompt
            graph_payload = copy.deepcopy(graph_template)

            # Replace prompt-like keys
            replaced_prompts = find_and_replace_prompts_in_obj(graph_payload, prompt)
            if replaced_prompts == 0:
                logger.debug("No prompt-like keys replaced; check graph node structure.")

            # Replace seed
            replaced_seeds = set_seed_in_obj(graph_payload, seed)
            if replaced_seeds == 0:
                logger.debug("No seed-like keys found to replace.")

            # Replace size
            replaced_size = set_size_in_obj(graph_payload, args.width, args.height)
            if replaced_size == 0:
                logger.debug("No width/height keys replaced; check graph node structure.")

            # Attempt to POST to ComfyUI
            resp = try_post_graph(args.host, args.port, graph_payload)
            if resp is None:
                logger.error("Failed to get a response for prompt: %s", prompt[:80])
                # Save graph and prompt for debugging
                dbg_path = sec_folder / f"failed_{global_prompt_index+1:04d}_seed{seed}.json"
                dbg_path.write_text(json.dumps({"prompt": prompt, "graph": graph_payload}, indent=2), encoding="utf-8")
            else:
                images = extract_images_from_response(resp)
                if images:
                    for im_idx, im_bytes in enumerate(images, start=1):
                        out_name = sec_folder / f"seed{seed:08d}_idx{im_idx:02d}.png"
                        save_bytes(out_name, im_bytes)
                        logger.info("Saved image: %s", out_name)
                else:
                    # No images found -> save full JSON for inspection
                    try:
                        j = resp.json()
                        dbg_json_path = sec_folder / f"response_{global_prompt_index+1:04d}_seed{seed}.json"
                        dbg_json_path.write_text(json.dumps(j, indent=2), encoding="utf-8")
                        logger.warning("No images found; saved response JSON to %s", dbg_json_path)
                    except Exception:
                        raw_path = sec_folder / f"response_{global_prompt_index+1:04d}_seed{seed}.bin"
                        save_bytes(raw_path, resp.content)
                        logger.warning("No images found; saved raw response to %s", raw_path)

            # Increment counters
            global_prompt_index += 1
            seed += args.seed_step
            time.sleep(args.wait)

    logger.info("Done. Generated images saved under %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
