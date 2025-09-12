#!/usr/bin/env python3
"""
Generate images from prompts using a saved graph against a local ComfyUI instance.

Usage (Windows cmd.exe):
  python scripts/generate_comfyui_from_prompts.py

This script:
 - Loads prompts from assets/timeline_joey_prompts.json (or --prompts)
 - Loads a saved ComfyUI graph JSON (or --graph)
 - For each section/prompt:
    - replaces prompt text, seed, and width/height in the graph JSON
    - queues the graph via the ComfyUI /prompt endpoint
    - polls /history/<prompt_id> until the run completes
    - downloads images reported by the history via /view and saves them to outputs/<root>/<section>/
 - Falls back to trying to POST the graph to some common ComfyUI endpoints if /prompt or history
   doesn't provide images.

Defaults:
 - prompts: assets/timeline_joey_prompts.json
 - graph: assets/Flux-dev_textures_api.json
 - host: localhost
 - port: 8188
 - outdir: outputs/joey
 - start-seed: 1
 - seed-step: 1
 - width/height: 1280x720
 - serial execution (one request at a time)
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
    return s.startswith("iVBOR") or s.startswith("/9j/") or s.startswith("R0lGOD") or s.startswith("Qk")


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
        for v in obj:
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
    logger.error("All HTTP endpoint attempts failed.")
    return None


# ---- ComfyUI /prompt + /history + /view workflow helpers ----
def queue_prompt_via_http(host: str, port: int, graph_payload: Dict[str, Any], prompt_id: str, client_id: str) -> None:
    """
    POST the graph payload to the ComfyUI /prompt endpoint with client/prompt ids.
    """
    url = f"http://{host}:{port}/prompt"
    payload = {"prompt": graph_payload, "client_id": client_id, "prompt_id": prompt_id}
    logger.debug("Queueing prompt via %s (prompt_id=%s)", url, prompt_id)
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()


def poll_history(host: str, port: int, prompt_id: str, timeout: int = 300, poll_interval: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    Poll /history/<prompt_id> until an entry appears containing outputs or until timeout.
    Returns the history entry (dict) or None on timeout/error.
    """
    url = f"http://{host}:{port}/history/{prompt_id}"
    start = time.time()
    while True:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    data = None
                if data:
                    # history endpoint might return { prompt_id: {...} } or the inner dict
                    if isinstance(data, dict) and prompt_id in data:
                        entry = data[prompt_id]
                    else:
                        entry = data
                    # check for outputs
                    if isinstance(entry, dict) and entry.get("outputs"):
                        return entry
            else:
                logger.debug("History returned status %s for %s", r.status_code, url)
        except requests.RequestException as exc:
            logger.debug("Error polling history: %s", exc)

        if time.time() - start > timeout:
            logger.warning("Timed out waiting for history entry for prompt_id=%s", prompt_id)
            return None
        time.sleep(poll_interval)


def get_image_via_http(host: str, port: int, filename: str, subfolder: str, folder_type: str) -> bytes:
    """
    Download an image via the /view endpoint using query params filename, subfolder, type.
    """
    url = f"http://{host}:{port}/view"
    params = {"filename": filename, "subfolder": subfolder or "", "type": folder_type or "images"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.content


def download_images_from_history(host: str, port: int, history_entry: Dict[str, Any]) -> List[bytes]:
    """
    Walk history outputs, call /view for each image meta, return list of image bytes.
    """
    images: List[bytes] = []
    outputs = history_entry.get("outputs", {}) if isinstance(history_entry, dict) else {}
    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        if "images" in node_output and isinstance(node_output["images"], list):
            for img_meta in node_output["images"]:
                if not isinstance(img_meta, dict):
                    continue
                filename = img_meta.get("filename")
                subfolder = img_meta.get("subfolder", "")
                folder_type = img_meta.get("type", "images")
                if not filename:
                    logger.debug("Image meta missing filename: %s", img_meta)
                    continue
                try:
                    img_bytes = get_image_via_http(host, port, filename, subfolder, folder_type)
                    images.append(img_bytes)
                except Exception as exc:
                    logger.warning("Failed to download image %s from node %s: %s", filename, node_id, exc)
    return images


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
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between history polls.")
    parser.add_argument("--history-timeout", type=int, default=300, help="Seconds to wait for history entry per prompt.")
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
    client_id_global = str(uuid.uuid4())

    for section in sections:
        sec_name = section.get("name") or f"section_{sections.index(section):02d}"
        prompts = section.get("prompts") or []
        sec_folder = out_root / sec_name
        sec_folder.mkdir(parents=True, exist_ok=True)
        logger.info("Processing section '%s' with %d prompts -> %s", sec_name, len(prompts), sec_folder)

        for idx, prompt in enumerate(prompts, start=1):
            logger.info("Prompt %d (global %d): %s", idx, global_prompt_index + 1, prompt[:200])
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

            images: List[bytes] = []

            # First, attempt to queue via /prompt + poll /history
            prompt_id = str(uuid.uuid4())
            try:
                queue_prompt_via_http(args.host, args.port, graph_payload, prompt_id, client_id_global)
                history = poll_history(args.host, args.port, prompt_id, timeout=args.history_timeout, poll_interval=args.poll_interval)
                if history:
                    images = download_images_from_history(args.host, args.port, history)
            except Exception as exc:
                logger.warning("Queue/poll workflow failed for prompt (seed=%s): %s", seed, exc)

            # Fallback to direct HTTP POST graph endpoints if we didn't get images
            if not images:
                logger.info("Falling back to HTTP POST graph endpoints for prompt (seed=%s).", seed)
                try:
                    resp = try_post_graph(args.host, args.port, graph_payload)
                    if resp is not None:
                        images = extract_images_from_response(resp)
                except Exception as exc:
                    logger.error("Fallback POST attempt failed: %s", exc)

            if not images:
                logger.error("No images obtained for prompt; saving debug artifacts.")
                dbg_path = sec_folder / f"failed_{global_prompt_index+1:04d}_seed{seed}.json"
                dbg_path.write_text(json.dumps({"prompt": prompt, "graph": graph_payload}, indent=2), encoding="utf-8")
            else:
                for im_idx, im_bytes in enumerate(images, start=1):
                    out_name = sec_folder / f"seed{seed:08d}_idx{im_idx:02d}.png"
                    save_bytes(out_name, im_bytes)
                    logger.info("Saved image: %s", out_name)

            # Increment counters
            global_prompt_index += 1
            seed += args.seed_step
            time.sleep(args.wait)

    logger.info("Done. Generated images saved under %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
