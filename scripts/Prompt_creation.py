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
def try_import_transformers() -> bool:
    try:
        import transformers  # noqa: F401
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
    Handles Gemma-3n (processor-based) and regular Gemma chat models.
    Falls back from 3n to 3-1b if generation raises a per-layer/shape error.
    """
    def __init__(self, cfg: LLMConfig, device_pref: str = "auto", verbose: bool = True):
        if not try_import_transformers():
            raise RuntimeError("transformers not installed. pip install transformers accelerate safetensors")
        import torch
        self.torch = torch
        self.cfg = cfg
        self.verbose = verbose
        self._is_3n = "gemma-3n" in cfg.model_id.lower()

        # Resolve device preference
        if device_pref == "auto":
            if torch.cuda.is_available():
                self.device_str = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device_str = "mps"
            else:
                self.device_str = "cpu"
        else:
            self.device_str = device_pref

        self.runtime_device = torch.device(
            "cuda:0" if self.device_str == "cuda"
            else ("mps" if self.device_str == "mps" else "cpu")
        )
        vprint(self.verbose, f"Device preference resolved to: {self.runtime_device}")

        ok = self._load_model(cfg.model_id)
        if not ok:
            raise RuntimeError("Failed to load any model.")

    def _load_model(self, mid: str) -> bool:
        try:
            if "gemma-3n" in mid.lower():
                from transformers import AutoProcessor, Gemma3nForConditionalGeneration
                hprint(self.verbose, f"Loading model {mid}")

                # Choose dtype + placement
                if self.device_str == "cuda":
                    dtype = self.torch.bfloat16 if self.torch.cuda.is_bf16_supported() else self.torch.float16
                    device_map = {"": 0}  # force full model on GPU0
                    self.model = Gemma3nForConditionalGeneration.from_pretrained(
                        mid, device_map=device_map, torch_dtype=dtype
                    )
                elif self.device_str == "mps":
                    # Load on CPU then move to MPS (common HF pattern)
                    self.model = Gemma3nForConditionalGeneration.from_pretrained(
                        mid, device_map={"": "cpu"}, torch_dtype=self.torch.float16
                    ).to("mps")
                else:  # cpu
                    self.model = Gemma3nForConditionalGeneration.from_pretrained(
                        mid, device_map={"": "cpu"}, torch_dtype=self.torch.float32
                    )

                self.processor = AutoProcessor.from_pretrained(mid)
                self.tokenizer = getattr(self.processor, "tokenizer", None)
                self.model_id = mid
                self._is_3n = True
                vprint(self.verbose, f"Model ready: {mid} | device={self.runtime_device} | dtype={self.model.dtype} (3n)")

            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                hprint(self.verbose, f"Loading model {mid}")

                if self.device_str == "cuda":
                    dtype = self.torch.bfloat16 if self.torch.cuda.is_bf16_supported() else self.torch.float16
                    device_map = {"": 0}
                    self.model = AutoModelForCausalLM.from_pretrained(
                        mid, device_map=device_map, torch_dtype=dtype
                    )
                elif self.device_str == "mps":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        mid, device_map={"": "cpu"}, torch_dtype=self.torch.float16
                    ).to("mps")
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        mid, device_map={"": "cpu"}, torch_dtype=self.torch.float32
                    )

                self.processor = None
                self.tokenizer = AutoTokenizer.from_pretrained(mid)
                self.model_id = mid
                self._is_3n = False
                vprint(self.verbose, f"Model ready: {mid} | device={self.runtime_device} | dtype={self.model.dtype}")

            try:
                self.torch.manual_seed(self.cfg.seed)
            except Exception:
                pass
            return True

        except Exception as e:
            vprint(self.verbose, f"[LOAD-ERR] {mid}: {e}")
            if "gemma-3n" in mid.lower():
                return self._load_model("google/gemma-3-1b-it")
            return False

    def _approx_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _build_inputs(self, system_prompt: str, user_prompt: str):
        if self._is_3n:
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ]
            chat = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(
                text=[chat],
                return_tensors="pt",
                padding=True
            ).to(self.runtime_device)  # <-- send to requested device
            vprint(self.verbose, f"Built 3n inputs on {self.runtime_device}: "
                                 f"input_ids.shape={tuple(inputs['input_ids'].shape)}")
            return inputs
        else:
            msgs = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ]
            text = None
            if self.tokenizer is not None:
                try:
                    chat = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    text = chat
                except Exception:
                    pass
            if text is None:
                text = (system_prompt + "\n\n" + user_prompt).strip()
            return self.tokenizer([text], return_tensors="pt").to(self.runtime_device)

    def generate_10_prompts(self, system_prompt: str, user_prompt: str) -> List[str]:
        vprint(self.verbose, f"Sampling params: temp={self.cfg.temperature} top_p={self.cfg.top_p} max_new={self.cfg.max_new_tokens}")
        vprint(self.verbose, f"Prompt approx tokens: ~{self._approx_tokens(user_prompt)}")
        vprint(self.verbose, f"Generating on device: {self.runtime_device} | dtype={self.model.dtype}")

        inputs = self._build_inputs(system_prompt, user_prompt)

        pad_id = None
        if getattr(self, "tokenizer", None) is not None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.model.config, "pad_token_id", None) or getattr(self.model.config, "eos_token_id", None) or 2

        try:
            gen = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=pad_id,
                repetition_penalty=1.1,
            )
        except Exception as e:
            if self._is_3n:
                vprint(self.verbose, f"[3n-generate] error: {e}\nFalling back to google/gemma-3-1b-it and retrying …")
                self._load_model("google/gemma-3-1b-it")
                inputs = self._build_inputs(system_prompt, user_prompt)
                gen = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    max_new_tokens=self.cfg.max_new_tokens,
                    pad_token_id=pad_id,
                    repetition_penalty=1.1,
                )
            else:
                raise

        if self._is_3n and getattr(self, "processor", None) is not None and hasattr(self.processor, "tokenizer"):
            tok = self.processor.tokenizer
            out_text = tok.decode(gen[0], skip_special_tokens=True)
        else:
            out_text = self.tokenizer.decode(gen[0], skip_special_tokens=True)

        if "assistant" in out_text.lower():
            parts = out_text.split("\n")
            out_text = "\n".join(parts[-60:])

        prompts = parse_numbered_list(out_text)
        vprint(self.verbose, f"LLM returned {len(prompts)} parsed prompts.")
        return prompts


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Generate 10 Gemma prompts per section from CLAP descriptor results.")
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="CLAP results JSON (from CLAP_extractor)")
    ap.add_argument("--out", type=Path, default=None, help="Output prompts JSON (default: <in>_prompts.json)")
    ap.add_argument("--model", type=str, default="google/gemma-3n-e2b-it", help="HF model id (3n supported)")
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
        help="Where to run the model/generation. 'cuda' forces GPU; 'auto' lets HF decide."
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
