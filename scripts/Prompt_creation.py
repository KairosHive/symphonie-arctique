# Prompt_creation.py — Gemma-3n friendly, verbose; generates 10 prompts per section
# ---------------------------------------------------------------------------------
# Usage:
#   python Prompt_creation.py --in ../results/<clap_results>.json
#     [--out <out.json>]
#     [--model google/gemma-3n-e2b-it]       # supports 3n; auto-fallback to gemma-3-1b-it on 3n errors
#     [--no-llm]                             # offline template mode
#     [--max-per-cat 24] [--temperature 0.9] [--top-p 0.95] [--max-new 700]
#     [--seed 42] [--show-per-cat 8] [--quiet]

# python Prompt_creation.py \
#   --in ../results/song_joey__timeline_joey__chunk1p0s__top5.json \
#   --device cuda \
#   --model google/gemma-3n-e2b-it
#
# Input JSON must be the CLAP matching output from your earlier script:
# {
#   "sections": [
#     {
#       "name": "...", "start": 0.0, "end": 8.0,
#       "chunks": [
#         { "chunk_start": 0.0, "chunk_end": 1.0,
#           "matches": {
#             "SoundQualities": [{"descriptor": "breathy", "score": 0.77}, ...],
#             "Emotions": [...],
#             ...
#           }
#         },
#         ...
#       ]
#     },
#     ...
#   ]
# }
#
# Output JSON:
# {
#   "source_results": "...",
#   "model": { "model_id": "...", "temperature": ..., ... } | { "mode": "template" },
#   "sections": [
#     {
#       "name": "...", "start": ..., "end": ...,
#       "prompts": ["...", "...", ..., "..."],      # 10 prompts
#       "descriptors_used": { "CategoryA": [...], ... }
#     },
#     ...
#   ]
# }
# ---------------------------------------------------------------------------------

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

def auto_out_path(in_path: Path) -> Path:
    return in_path.with_name(in_path.stem + "_prompts.json")

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
    max_new_tokens: int = 700
    seed: int = 42

ARCHIVE_STYLE = (
    "Style: early 20th-century archival documentation; sepia tonality; paper texture; "
    "typewritten captions; catalog stamps and marginalia; soft film grain; subdued contrast; "
    "measured, objective phrasing; no brand names; no camera jargon."
)

def make_user_prompt_for_section(section_name: str, start: float, end: float, by_category: Dict[str, List[str]]) -> str:
    # Build category blocks safely (no backslashes inside f-string expressions)
    cat_blocks = []
    for cat, descs in by_category.items():
        if not descs:
            continue
        cat_blocks.append(f"- {cat}: {', '.join(descs)}")

    blocks_str = "\n".join(cat_blocks)
    time_s = f"{start:.2f}–{end:.2f}s"
    instructions = (
        "Given the descriptor lists per category, compose TEN diverse image prompts that each feel like a "
        "single visual narrative scene from an old archive. Combine elements from multiple categories in each prompt. "
        "Vary location, subject, composition, and time-of-day across the set. Keep each prompt 25–60 words. "
        "Avoid first-person. Do not number the prompts inside the text (we will number externally)."
    )

    return (
        f"Section: {section_name} ({time_s})\n"
        f"{ARCHIVE_STYLE}\n\n"
        f"{instructions}\n\n"
        "Descriptors:\n"
        f"{blocks_str}\n\n"
        "Output format:\n"
        "1) <prompt one>\n"
        "2) <prompt two>\n"
        "...\n"
        "10) <prompt ten>\n"
        "Only write the prompts."
    )

def parse_numbered_list(text: str) -> List[str]:
    lines = [l.strip() for l in text.splitlines()]
    items, curr = [], []
    for ln in lines:
        # detect "1) ..." or "1. ..."
        if any(ln.startswith(f"{i})") or ln.startswith(f"{i}.") for i in range(1, 12)):
            if curr:
                items.append(" ".join(curr).strip())
                curr = []
            ln2 = ln.split(")", 1)[-1] if ")" in ln else ln.split(".", 1)[-1]
            curr.append(ln2.strip())
        else:
            if ln:
                curr.append(ln)
    if curr:
        items.append(" ".join(curr).strip())
    return [x for x in items if x][:10]

def synth_fallback_prompts(context_hint: str, need: int) -> List[str]:
    base = "sepia, archival record, catalog stamp, paper wear, subdued contrast, typewritten caption"
    return [f"A preserved archival scene: {base}. Visual synthesis guided by: {context_hint[:220]}." for _ in range(need)]

def template_10_prompts(section_name: str, start: float, end: float, by_category: Dict[str, List[str]], seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    cats = list(by_category.keys())
    prompts, time_s = [], f"{start:.0f}–{end:.0f}s"
    for _ in range(10):
        picks = []
        rng.shuffle(cats)
        for cat in cats:
            if not by_category[cat]:
                continue
            k = 2 if len(by_category[cat]) >= 2 else 1
            picks.extend(rng.sample(by_category[cat], k=k))
        picks_str = ", ".join(dict.fromkeys(picks))
        prompts.append(f"{section_name} ({time_s}) — Old archive photograph; {ARCHIVE_STYLE.lower()} Motifs: {picks_str}.")
    return prompts

class GemmaWrapper:
    """
    Ollama chat wrapper. Uses cfg.model_id directly (for example: 'gemma3:4b' or any Ollama model).
    Keeps the same public interface (generate_10_prompts) so the rest of the script remains unchanged.
    """
    def __init__(self, cfg: LLMConfig, device_pref: str = "auto", verbose: bool = True):
        if not try_import_ollama():
            raise RuntimeError("ollama not installed. pip install ollama")
        import ollama
        self.ollama = ollama
        self.cfg = cfg
        self.verbose = verbose
        self.model_id = cfg.model_id

    def _approx_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _extract_text_from_resp(self, resp) -> str:
        """
        Robustly extract text content from various Ollama response shapes:
        - {'message': {'content': "<text>"}}
        - {'choices': [{'message': {'content': "<text>"}} , ...]}
        - {'choices': [{'text': "<text>"}, ...]}
        - a plain string
        Fallback to empty string when nothing found.
        """
        try:
            if isinstance(resp, str):
                return resp or ""
            if isinstance(resp, dict):
                # direct message
                if "message" in resp and isinstance(resp["message"], dict):
                    return resp["message"].get("content", "") or ""
                # choices list
                if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                    first = resp["choices"][0]
                    if isinstance(first, dict):
                        if "message" in first and isinstance(first["message"], dict):
                            return first["message"].get("content", "") or ""
                        if "text" in first:
                            return first.get("text", "") or ""
                # some clients may return content at top-level
                if "content" in resp:
                    return resp.get("content", "") or ""
                # As a last resort, stringify
                return str(resp)
        except Exception:
            return ""
        return ""

    def generate_10_prompts(self, system_prompt: str, user_prompt: str) -> List[str]:
        vprint(self.verbose, f"Sampling params: temp={self.cfg.temperature} top_p={self.cfg.top_p} max_new={self.cfg.max_new_tokens}")
        vprint(self.verbose, f"Prompt approx tokens: ~{self._approx_tokens(user_prompt)}")

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

        try:
            # Use Ollama's chat API. Pass top_p if available and map max_new_tokens -> max_tokens.
            call_kwargs = {
                "model": self.model_id,
                "messages": messages,
                "temperature": float(self.cfg.temperature),
                "max_tokens": int(self.cfg.max_new_tokens),
            }
            # top_p may be supported by Ollama; include it if configured
            try:
                call_kwargs["top_p"] = float(self.cfg.top_p)
            except Exception:
                pass

            resp = self.ollama.chat(**call_kwargs)

            out_text = self._extract_text_from_resp(resp)
            prompts = parse_numbered_list(out_text)

            vprint(self.verbose, f"LLM returned {len(prompts)} parsed prompts.")
            return prompts

        except Exception as e:
            vprint(self.verbose, f"[OLLAMA-ERR] {e}")
            # Let the caller handle fallback to template mode by raising
            raise


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Generate 10 image prompts per section from CLAP descriptor results using an Ollama chat model.")
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="CLAP results JSON (from CLAP_extractor)")
    ap.add_argument("--out", type=Path, default=None, help="Output prompts JSON (default: <in>_prompts.json)")
    ap.add_argument("--model", type=str, default="gemma3:4b", help="Ollama model id (e.g. 'gemma3:4b' or any Ollama model)")
    ap.add_argument("--no-llm", action="store_true", help="Template mode (no model call)")
    ap.add_argument("--max-per-cat", type=int, default=24, help="Max descriptors kept per category")
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--show-per-cat", type=int, default=8)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Compatibility option; may be ignored when using Ollama. Kept for callers that expect a device flag."
    )

    args = ap.parse_args()

    verbose = not args.quiet
    hprint(verbose, "Inputs")
    vprint(verbose, f"Results JSON : {args.inp}")
    out_path = args.out or auto_out_path(args.inp)
    vprint(verbose, f"Output JSON  : {out_path}")
    vprint(verbose, f"Model request: {args.model}")
    vprint(verbose, f"Params       : temp={args.temperature} top_p={args.top_p} max_new={args.max_new} seed={args.seed}")
    vprint(verbose, f"Max per cat  : {args.max_per_cat} (printing top {args.show_per_cat})")

    hprint(verbose, "Loading results")
    data = load_clap_results(args.inp)
    sections = data.get("sections", [])
    vprint(verbose, f"Sections found: {len(sections)}")
    if not sections:
        print("No sections in input JSON.")
        sys.exit(1)

    llm = None
    llm_meta = None
    if not args.no_llm:
        try:
            llm = GemmaWrapper(
                LLMConfig(
                    model_id=args.model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new,
                    seed=args.seed,
                ),
                device_pref=args.device,
                verbose=verbose,
            )

            llm_meta = {
                "model_id": llm.model_id,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new,
            }
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

        for cat in cats_sorted:
            counter = pool.by_category[cat]
            ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0].lower()))
            shown, more = ranked[:args.show_per_cat], max(0, len(ranked) - args.show_per_cat)
            top_str = "; ".join([f"{d}×{c}" for d, c in shown])
            vprint(verbose, f"    - {cat:<22} | unique={len(ranked):<3} | total hits={sum(counter.values()):<3} | top: {top_str}{' ...' if more>0 else ''}")

        per_cat = summarize_pool(pool, max_per_category=args.max_per_cat)
        user_prompt = make_user_prompt_for_section(pool.name, pool.start, pool.end, per_cat)
        system_prompt = "You are a meticulous archivist-poet. Turn descriptor lists into concise, evocative scene prompts that feel like historical records in a nordic world full of mythology and animate forces of nature."

        if args.no_llm:
            vprint(verbose, "  generating (offline template mode)")
            prompts = template_10_prompts(pool.name, pool.start, pool.end, per_cat, seed=args.seed)
        else:
            vprint(verbose, f"  generating with LLM [{llm.model_id}] …")
            prompts = llm.generate_10_prompts(system_prompt, user_prompt)
            if len(prompts) < 10:
                vprint(verbose, f"  [LLM] parsed only {len(prompts)} prompts; padding with template fallbacks.")
                prompts.extend(synth_fallback_prompts(user_prompt, need=10 - len(prompts)))
                prompts = prompts[:10]

        vprint(verbose, "  preview:")
        for i, p in enumerate(prompts[:3], start=1):
            vprint(verbose, f"    {i}. {p[:140]}{'…' if len(p) > 140 else ''}")
        if len(prompts) > 3:
            vprint(verbose, f"    … and {len(prompts) - 3} more")

        out["sections"].append({
            "name": pool.name,
            "start": pool.start,
            "end": pool.end,
            "prompts": prompts,
            "descriptors_used": per_cat,
        })

    hprint(verbose, "Writing output")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    vprint(verbose, f"[OK] Wrote prompts: {out_path}")

if __name__ == "__main__":
    main()
