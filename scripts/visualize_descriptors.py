#!/usr/bin/env python3
"""
visualize_descriptors_clean.py

Generates *two* elegant, single-figure-per-category visuals:

A) <dict>__diff_heatmap.png
   - Heatmap of mean scores for the top-K descriptors (global within dict),
     columns = sections, rows = descriptors. Shows how descriptors differ by section.

B) <dict>__top3_per_section.png
   - One chart for the whole dict: each section has 3 bars (ranked 1..3),
     bar labels are the descriptor names (no busy legend).

Usage:
  python visualize_descriptors_clean.py --json results.json --outdir figs --heatmap-topk 12 --normalize zscore

Normalization:
  --normalize none|zscore|minmax
  (Applied per-descriptor across sections for the heatmap only; top-3 plot uses raw means.)
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------- Load & Flatten ---------------------

def load_results(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_to_df(data: dict) -> pd.DataFrame:
    """
    Returns tidy DF:
      section, sec_start, sec_end, dict, descriptor, score, t_mid
    """
    rows = []
    for sec in data.get("sections", []):
        sec_name = sec.get("name", "")
        sec_start = float(sec.get("start", 0.0))
        sec_end = float(sec.get("end", 0.0))
        for ch in sec.get("chunks", []):
            c0 = float(ch.get("chunk_start", 0.0))
            c1 = float(ch.get("chunk_end", 0.0))
            t_mid = 0.5 * (c0 + c1)
            for dict_name, items in ch.get("matches", {}).items():
                for it in items:
                    rows.append({
                        "section": sec_name,
                        "sec_start": sec_start,
                        "sec_end": sec_end,
                        "dict": dict_name,
                        "descriptor": it["descriptor"],
                        "score": float(it["score"]),
                        "t_mid": t_mid,
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Parsed 0 rows from JSON. Check the file.")
    return df


def order_sections(df: pd.DataFrame) -> list:
    return (
        df.groupby(["section", "sec_start"], as_index=False).size()
          .sort_values("sec_start")["section"]
          .tolist()
    )


# --------------------- Aggregations ---------------------

def mean_pivot(df_dict: pd.DataFrame) -> pd.DataFrame:
    """
    Mean score per section x descriptor (values).
    Rows = descriptors, Cols = sections.
    """
    piv = (
        df_dict.groupby(["section", "descriptor"])["score"]
               .mean()
               .unstack("section")
               .fillna(0.0)
    )
    # reorder section columns by time
    sec_order = order_sections(df_dict)
    piv = piv.reindex(columns=sec_order)
    return piv


def select_topk_global(piv: pd.DataFrame, k: int) -> pd.DataFrame:
    """Pick top-K descriptors by global mean across sections."""
    if k is None or k <= 0 or k >= len(piv.index):
        return piv
    order = piv.mean(axis=1).sort_values(ascending=False).head(k).index
    return piv.loc[order]


def normalize_matrix(piv: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, str]:
    """
    Normalize per-descriptor across sections for visual comparability.

    mode:
      - "none": return as-is
      - "zscore": (x - mean) / std per row; if std=0, return zeros
      - "minmax": (x - min) / (max - min) per row; if flat row, return zeros
    """
    if mode == "none":
        return piv, "Mean score"
    piv_norm = piv.copy()
    vals = piv_norm.values.astype(float)

    if mode == "zscore":
        mu = vals.mean(axis=1, keepdims=True)
        sigma = vals.std(axis=1, ddof=0, keepdims=True)
        sigma[sigma == 0] = 1.0
        vals = (vals - mu) / sigma
        label = "Z-scored (per descriptor)"
    elif mode == "minmax":
        vmin = vals.min(axis=1, keepdims=True)
        vmax = vals.max(axis=1, keepdims=True)
        rng = (vmax - vmin)
        rng[rng == 0] = 1.0
        vals = (vals - vmin) / rng
        label = "Min-max (per descriptor)"
    else:
        return piv, "Mean score"

    piv_norm.loc[:, :] = vals
    return piv_norm, label


# --------------------- Plotters ---------------------

def plot_diff_heatmap(piv: pd.DataFrame, dict_name: str, outdir: Path,
                      normalize: str = "none", heatmap_topk: int = 12):
    """
    ONE plot per category showing how descriptors differ between sections.
    Uses top-K descriptors globally for readability.
    """
    piv = select_topk_global(piv, heatmap_topk)
    piv_norm, cbar_label = normalize_matrix(piv, normalize)

    # size scales with content
    h = max(2.5, 0.4 * len(piv_norm.index) + 1.5)
    w = max(4.0, 0.9 * len(piv_norm.columns) + 1.5)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(piv_norm.values, aspect="auto", interpolation="nearest")

    ax.set_yticks(np.arange(len(piv_norm.index)))
    ax.set_yticklabels(piv_norm.index)
    ax.set_xticks(np.arange(len(piv_norm.columns)))
    ax.set_xticklabels(piv_norm.columns, rotation=45, ha="right")

    ax.set_title(f"{dict_name} — descriptor differences across sections")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    out = outdir / f"{dict_name}__diff_heatmap.png"
    plt.savefig(out, dpi=160)
    plt.close()


def plot_top3_per_section(piv: pd.DataFrame, dict_name: str, outdir: Path):
    """
    ONE plot per category: for each section, show its top-3 descriptors (raw means).
    - x-axis: sections
    - 3 bars per section with labels = descriptor names
    - color indicates rank (1=darkest)
    """
    sections = list(piv.columns)
    # compute top-3 per section
    top3 = []
    for sec in sections:
        col = piv[sec].sort_values(ascending=False).head(3)
        for rank, (desc, val) in enumerate(col.items(), start=1):
            top3.append({"section": sec, "rank": rank, "descriptor": desc, "score": float(val)})

    if not top3:
        return

    df3 = pd.DataFrame(top3)

    fig_w = max(5.0, 1.0 * len(sections) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))

    x = np.arange(len(sections))
    width = 0.75 / 3.0  # 3 bars per section
    colors = {1: "#3b4cc0", 2: "#89a1ef", 3: "#c7d3f8"}

    for r in [1, 2, 3]:
        sub = df3[df3["rank"] == r].set_index("section").reindex(sections)
        heights = sub["score"].values
        x_pos = x + (r - 2) * width

        ax.bar(x_pos, heights, width=width, label=f"Top {r}", color=colors[r])

        # labels (use annotate to offset slightly above bar)
        descs = sub["descriptor"].values
        for xi, h, desc in zip(x_pos, heights, descs):
            if not np.isfinite(h) or h <= 0 or desc is None:
                continue
            ax.annotate(
                str(desc),
                xy=(xi, h),
                xytext=(0, 2),  # points above bar
                textcoords="offset points",
                ha="center", va="bottom",
                rotation=90, fontsize=8
            )

    # headroom for labels
    ymax = np.nanmax(df3["score"].values)
    if np.isfinite(ymax) and ymax > 0:
        ax.set_ylim(0, ymax * 1.12)

    ax.set_xticks(x)
    ax.set_xticklabels(sections, rotation=45, ha="right")
    ax.set_ylabel("Mean score")
    ax.set_title(f"{dict_name} — top-3 descriptors per section (single view)")
    ax.legend(title="Rank", ncols=3, loc="upper left", fontsize=9)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    plt.tight_layout()

    out = outdir / f"{dict_name}__top3_per_section.png"
    plt.savefig(out, dpi=160)
    plt.close()


# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True, help="Path to CLAP_extractor output JSON")
    ap.add_argument("--outdir", type=Path, default=Path("figs"), help="Directory for figures")
    ap.add_argument("--heatmap-topk", type=int, default=12, help="Rows (descriptors) to show in heatmap (global top-K)")
    ap.add_argument("--normalize", type=str, default="zscore", choices=["none", "zscore", "minmax"],
                    help="Normalization for heatmap (per-descriptor across sections)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    data = load_results(args.json)
    df = flatten_to_df(data)

    for dict_name in sorted(df["dict"].unique()):
        df_dict = df[df["dict"] == dict_name]
        piv = mean_pivot(df_dict)
        if piv.empty:
            continue
        # A) difference heatmap
        plot_diff_heatmap(piv, dict_name, args.outdir, normalize=args.normalize, heatmap_topk=args.heatmap_topk)
        # B) top-3 per section (single plot)
        plot_top3_per_section(piv, dict_name, args.outdir)

    print(f"Done. Saved figures to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
