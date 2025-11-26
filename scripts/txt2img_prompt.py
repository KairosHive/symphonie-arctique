from __future__ import annotations

import argparse, json, random, sys, time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# ---------- printing helpers ----------
def _ts() -> str:
    return time.strftime("%H:%M:%S")

def vprint(enabled: bool, *a, **k):
    if enabled:
        print(f"[{_ts()}]", *a, **k)

def hprint(enabled: bool, title: str):
    if enabled:
        print(f"\n[{_ts()}] " + "=" * 10 + f" {title} " + "=" * 10)

# ---------- I/O ----------
def load_clap_results(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "sections" not in data or not isinstance(data["sections"], list):
        raise ValueError("Input results JSON must contain a top-level 'sections' list.")
    return data

def load_system_prompt(path: Path) -> str:
    """Load system prompt from markdown file."""
    return path.read_text(encoding="utf-8").strip()

def auto_out_path(in_path: Path) -> Path:
    return in_path.with_name(in_path.stem + "_prompts.json")

def write_out_json(path: Path, data: dict):
    """
    Write JSON to disk in a straightforward manner.
    Called after each section to persist progress.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- pooling (collect per-section descriptor hits) ----------
@dataclass
class SectionPool:
    name: str
    start: float
    end: float
    by_category: Dict[str, Counter]
    n_chunks: int

def collect_section_descriptors(section: dict, top_k_per_chunk_per_category: int = 2) -> SectionPool:
    pool: Dict[str, Counter] = defaultdict(Counter)
    chunks = section.get("chunks", [])
    for ch in chunks:
        matches: dict = ch.get("matches", {})
        if not isinstance(matches, dict):
            continue
        for cat, items in matches.items():
            if not isinstance(items, list) or not items:
                continue
            for it in items[:top_k_per_chunk_per_category]:
                desc = str(it.get("descriptor", "")).strip()
                if desc:
                    pool[cat][desc] += 1
    return SectionPool(
        name=section.get("name", "unnamed"),
        start=float(section.get("start", 0.0)),
        end=float(section.get("end", 0.0)),
        by_category=pool,
        n_chunks=len(chunks),
    )

def summarize_pool(pool: SectionPool, max_per_category: int = 24) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for cat, counter in pool.by_category.items():
        ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0].lower()))
        out[cat] = [d for d, _ in ranked[:max_per_category]]
    return out

# --- NEW ---: This function creates a random subset of descriptors for a single prompt.
def sample_descriptors(by_category: Dict[str, List[str]], seed: int) -> Dict[str, List[str]]:
    """From the full pool of descriptors, pick a random subset from each category."""
    rng = random.Random(seed)
    sampled_dict: Dict[str, List[str]] = {}
    
    categories = list(by_category.keys())
    rng.shuffle(categories) # Use a random subset of categories too

    # Decide to use between 2 and 5 categories for this prompt
    num_cats_to_use = rng.randint(2, min(len(categories), 5))
    
    for cat in categories[:num_cats_to_use]:
        descs = by_category[cat]
        if not descs:
            continue
        
        # For each chosen category, pick 1 to 3 descriptors
        max_k = min(len(descs), 3)
        k = rng.randint(1, max_k)
        sampled_dict[cat] = rng.sample(descs, k)
        
    return sampled_dict

# ---------- LLM helpers ----------
def try_import_ollama() -> bool:
    try:
        import ollama  # noqa: F401
        return True
    except Exception:
        return False

@dataclass
class LLMConfig:
    model_id: str
    temperature: float = 0.9
    top_p: float = 0.95
    num_predict: int = 150
    seed: int = 42


def make_user_prompt_for_single(section_name: str, start: float, end: float, by_category: Dict[str, List[str]], existing_prompts: List[str]) -> str:
    cat_blocks = []
    for cat, descs in by_category.items():
        if not descs:
            continue
        cat_blocks.append(f"- {cat}: {', '.join(descs)}")
    blocks_str = "\n".join(cat_blocks)
    time_s = f"{start:.2f}–{end:.2f}s"

    instructions = (
            """Generate TEN diverse txt2img optimized prompts, each using a unique random sample of the provided descriptors.

            Draw from **multiple descriptor categories**, mix them when possible, and use them to create original symbolic compositions.
            Use varied visual styles, photographic techniques, and film processes to ensure each prompt has a distinct aesthetic feel. 
            FEEL FREE TO EXPAND upon the descriptors to create rich, evocative imagery using detailed language and suprising combinations of techniques.
            Structure: Follow **Subject + Action + Style + Context** formula , front-loading the descriptors you are given.
            NOTE: the image model won't understand descriptors if you just repeat them, so you must unpack them into evocative, precise language by drawing from your understanding of visual culture, photographic techniques, and art history.
            Constraints: 65-85 words per prompt. """
        )

    diversity_instruction = ""
    if existing_prompts:
        existing_list = "\n".join(f"- {p}" for p in existing_prompts)
        diversity_instruction = (
            "\nIMPORTANT: Make this new prompt significantly different in subject, location, or composition from these already created:\n"
            f"{existing_list}"
        )

    return (
        f"Section: {section_name} ({time_s})\n"
        f"{instructions}{diversity_instruction}\n\n"
        "Descriptors for this prompt:\n" # --- CHANGED ---: Clarified that this is a focused list
        f"{blocks_str}\n\n"
        "Output format:\n"
        "Return a JSON object with a single key \"prompt\" whose value is the string of the generated prompt.\n"
        "Example:\n"
        "{ \"prompt\": \"<the generated prompt text>\" }\n"
        "Do not include any extra text before or after the JSON."
    )

# --- CHANGED ---: Template function now returns the same structure as the LLM path.
def template_10_prompts(section_name: str, start: float, end: float, by_category: Dict[str, List[str]], seed: int = 42) -> List[Dict]:
    """Generates 10 prompts using a template, each with a unique descriptor sample."""
    generated_prompts = []
    time_s = f"{start:.0f}–{end:.0f}s"
    
    for i in range(10):
        current_seed = seed + i
        # Use the same sampling logic as the LLM path
        sampled_descriptors = sample_descriptors(by_category, seed=current_seed)
        
        # Flatten the sampled descriptors into a single string for the template
        picks = [desc for descs in sampled_descriptors.values() for desc in descs]
        picks_str = ", ".join(dict.fromkeys(picks))
        
        prompt_text = f"{section_name} ({time_s}) — Motifs: {picks_str}."
        
        generated_prompts.append({
            "prompt": prompt_text,
            "descriptors_used": sampled_descriptors
        })
    return generated_prompts

class OllamaWrapper:
    def __init__(self, cfg: LLMConfig, verbose: bool = True, host: str = None):
        if not try_import_ollama():
            raise RuntimeError("ollama not installed. pip install ollama")
        import ollama
        self.ollama = ollama
        self.client = ollama.Client(host=host, timeout=600) if host else ollama.Client(timeout=600)
        self.cfg = cfg
        self.verbose = verbose
        self.model_id = cfg.model_id

    def _extract_text_from_resp(self, resp) -> str:
        """Robustly extract text content from various Ollama response shapes."""
        try:
            if isinstance(resp, str): return resp or ""
            if hasattr(resp, "message"):
                msg = getattr(resp, "message")
                if isinstance(msg, dict): return msg.get("content", "") or ""
                if hasattr(msg, "content"): return getattr(msg, "content") or ""
            if isinstance(resp, dict):
                if "message" in resp and isinstance(resp["message"], dict):
                    return resp["message"].get("content", "") or ""
            return ""
        except Exception:
            return ""
            
    def _extract_single_prompt(self, resp) -> str:
        text = self._extract_text_from_resp(resp).strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("prompt"), str):
                return data["prompt"].strip()
        except json.JSONDecodeError:
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = text[start:end]
                    data = json.loads(json_str)
                    if isinstance(data, dict) and isinstance(data.get("prompt"), str):
                        return data["prompt"].strip()
            except Exception:
                pass
        return text.strip('" ')

    def generate_single_prompt(self, system_prompt: str, user_prompt: str, current_seed: int) -> str:
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]
        try:
            options = {
                "temperature": float(self.cfg.temperature),
                "top_p": float(self.cfg.top_p),
                "num_predict": int(self.cfg.num_predict),
                "seed": int(current_seed),
            }
            schema = {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"]
            }
            resp = self.client.chat(
                model=self.model_id,
                messages=messages,
                options=options,
                format=schema,
                stream=False,
                keep_alive="20m",
            )
            prompt = self._extract_single_prompt(resp)
            if not prompt:
                raw = self._extract_text_from_resp(resp)
                vprint(self.verbose, f"[LLM DEBUG] Empty parse. Raw content: {raw!r}")
            return prompt
        except Exception as e:
            vprint(self.verbose, f"[OLLAMA-ERR] {e}")
            raise

# ---------- main ----------
def main():
    """
    Generate 10 image prompts per section from CLAP descriptor results using an Ollama chat model.

    Usage example:
    python scripts/Prompt_creation.py --in path/to/clap_results.json --out path/to/output_prompts.json --model hf.co/gabriellarson/Hermes-4-14B-GGUF:Q8_0 --temperature 1.2 --max-new 250


    """
    ap = argparse.ArgumentParser(description="Generate 10 image prompts per section from CLAP descriptor results using an Ollama chat model.")
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="CLAP results JSON (from CLAP_extractor)")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "results" / "prompts_txt2img" / "output_prompts.json", help="Output prompts JSON (default: <in>_prompts.json)")
    ap.add_argument("--model", type=str, default="hf.co/gabriellarson/Hermes-4-14B-GGUF:Q8_0", help="Ollama model id (e.g. 'gemma3:4b' or any Ollama model)")
    ap.add_argument("--server", type=str, default=None, help="Ollama server URL (e.g. 'http://remote-host:11434')")
    ap.add_argument("--system-prompt", type=Path, default=Path(__file__).parent.parent / "assets" / "system_prompt_txt2img.md", help="Path to system prompt markdown file")
    ap.add_argument("--no-llm", action="store_true", help="Template mode (no model call)")
    ap.add_argument("--max-per-cat", type=int, default=24, help="Max descriptors kept per category")
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=250, help="Max new tokens PER prompt")
    ap.add_argument("--show-per-cat", type=int, default=8)
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args()

    verbose = not args.quiet
    hprint(verbose, "Inputs")
    vprint(verbose, f"Results JSON : {args.inp}")
    out_path = args.out or auto_out_path(args.inp)
    vprint(verbose, f"Output JSON  : {out_path}")
    vprint(verbose, f"Model request: {args.model}")
    if args.server:
        vprint(verbose, f"Server       : {args.server}")
    vprint(verbose, f"Params       : temp={args.temperature} top_p={args.top_p} max_new={args.max_new} seed={args.seed}")
    vprint(verbose, f"Max per cat  : {args.max_per_cat} (printing top {args.show_per_cat})")

    hprint(verbose, "Loading results")
    data = load_clap_results(args.inp)
    sections = data.get("sections", [])
    vprint(verbose, f"Sections found: {len(sections)}")
    if not sections:
        print("No sections in input JSON.")
        sys.exit(1)

    hprint(verbose, "Loading system prompt")
    try:
        system_prompt = load_system_prompt(args.system_prompt)
        vprint(verbose, f"Loaded from: {args.system_prompt}")
    except Exception as e:
        vprint(verbose, f"[ERROR] Could not load system prompt from {args.system_prompt}: {e}")
        sys.exit(1)

    llm = None
    llm_meta = None
    if not args.no_llm:
        try:
            llm = OllamaWrapper(LLMConfig(model_id=args.model, temperature=args.temperature, top_p=args.top_p, num_predict=args.max_new, seed=args.seed), verbose=verbose, host=args.server)
            llm_meta = {"model_id": llm.model_id, "temperature": args.temperature, "top_p": args.top_p, "num_predict": args.max_new}
            if args.server:
                llm_meta["server"] = args.server
        except Exception as e:
            vprint(verbose, f"[WARN] Could not initialize model ({e}). Falling back to --no-llm mode.")
            args.no_llm = True

    out = {"source_results": str(args.inp), "model": (llm_meta or {"mode": "template"}), "sections": []}

    for idx, sec in enumerate(sections, start=1):
        name = sec.get("name", f"section_{idx}")
        start, end = float(sec.get("start", 0.0)), float(sec.get("end", 0.0))
        hprint(verbose, f"Section {idx}/{len(sections)} — {name} ({start:.2f}s–{end:.2f}s)")

        pool = collect_section_descriptors(sec, top_k_per_chunk_per_category=2)
        vprint(verbose, f"  chunks       : {pool.n_chunks}")
        cats_sorted = sorted(pool.by_category.keys(), key=str.lower)
        vprint(verbose, f"  categories   : {len(pool.by_category)} -> {', '.join(cats_sorted) if cats_sorted else '—'}")
        
        # This is the full pool of available descriptors for the section
        full_descriptor_pool = summarize_pool(pool, max_per_category=args.max_per_cat)

        # --- CHANGED ---: This list now stores dicts: {"prompt": str, "descriptors_used": dict}
        generated_prompts: List[Dict] = []
        
        if args.no_llm:
            mode_note = "offline template mode"
            vprint(verbose, f"  generating ({mode_note})")
            generated_prompts = template_10_prompts(pool.name, pool.start, pool.end, full_descriptor_pool, seed=args.seed)
        else:
            vprint(verbose, f"  generating 10 prompts with LLM [{llm.model_id}]…")
            
            # --- CHANGED ---: Main generation logic now samples descriptors in each iteration.
            for i in range(10):
                try:
                    vprint(verbose, f"    > prompt {i+1}/10...")
                    current_seed = args.seed + idx * 10 + i
                    
                    # 1. Get a new random sample of descriptors for this specific prompt
                    sampled_descriptors = sample_descriptors(full_descriptor_pool, seed=current_seed)
                    
                    # 2. Create the user prompt with this sample
                    prompt_texts_only = [p["prompt"] for p in generated_prompts]
                    user_prompt = make_user_prompt_for_single(pool.name, pool.start, pool.end, sampled_descriptors, existing_prompts=prompt_texts_only)
                    
                    # 3. Generate the prompt
                    new_prompt_text = llm.generate_single_prompt(system_prompt, user_prompt, current_seed)
                    
                    if new_prompt_text:
                        # 4. Store the prompt along with the descriptors used to create it
                        generated_prompts.append({
                            "prompt": new_prompt_text,
                            "descriptors_used": sampled_descriptors
                        })
                    else:
                        vprint(verbose, "    [WARN] LLM returned empty prompt, stopping section.")
                        break
                except KeyboardInterrupt:
                    vprint(verbose, "  [INTERRUPT] Saving partial progress and exiting…")
                    write_out_json(out_path, out)
                    raise
                except Exception as e:
                    vprint(verbose, f"  [LLM ERROR] on prompt {i+1}: {e}; stopping section.")
                    break

            if len(generated_prompts) < 10:
                vprint(verbose, f"  [LLM] generated only {len(generated_prompts)} prompts; padding with template fallbacks.")
                template_prompts = template_10_prompts(pool.name, pool.start, pool.end, full_descriptor_pool, seed=args.seed + idx)
                needed = 10 - len(generated_prompts)
                generated_prompts.extend(template_prompts[:needed])

        vprint(verbose, "  preview:")
        # --- CHANGED ---: Access the prompt text from the dictionary
        for i, p_data in enumerate(generated_prompts[:3], start=1):
            p = p_data["prompt"]
            vprint(verbose, f"    {i}. {p[:140]}{'…' if len(p) > 140 else ''}")
        if len(generated_prompts) > 3:
            vprint(verbose, f"    … and {len(generated_prompts) - 3} more")

        # --- CHANGED ---: The output structure for the section is updated
        out["sections"].append({
            "name": pool.name,
            "start": pool.start,
            "end": pool.end,
            "generated_prompts": generated_prompts, # Key name changed for clarity
            "full_descriptor_pool": full_descriptor_pool, # The original full set is also saved
        })
        write_out_json(out_path, out)
        vprint(verbose, f"  [OK] Progress saved to: {out_path}")

    hprint(verbose, "Writing final output")
    # check if there are files with the same name already; if yes, add a number 
    if out_path.exists():
        base = out_path.stem
        ext = out_path.suffix
        parent = out_path.parent
        counter = 1
        while True:
            new_name = f"{base}_{counter}{ext}"
            new_path = parent / new_name
            if not new_path.exists():
                out_path = new_path
                break
            counter += 1
        
    write_out_json(out_path, out)
    vprint(verbose, f"[OK] Wrote all prompts: {out_path}")


if __name__ == "__main__":
    main()