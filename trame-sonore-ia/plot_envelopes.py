"""
Plot per-depth temperature curves and the resampled audio-rate envelopes.
Saves PNGs to outputs/ for visual sanity-check.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soil_data import (
    USABLE_DEPTHS,
    envelope_for_depth,
    extract_depth_series,
    find_first_full_year,
    load_csv,
)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

SR = 44100
DURATION_S = 47.0
TARGET_N = int(SR * DURATION_S)


def plot_full_history():
    df = load_csv()
    fig, ax = plt.subplots(figsize=(14, 7))
    for d in USABLE_DEPTHS:
        s = extract_depth_series(df, d)
        # x in years from start
        x_years = np.arange(len(s)) / 365.25 + df["Year"].min()
        ax.plot(x_years, s, label=f"{d} cm", linewidth=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Tasiujaq soil temperatures by depth (full history, NaN-interpolated)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = OUT / "soil_full_history.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[save] {out_path}")


def plot_one_year_per_depth():
    """One full year per depth, normalized to audio-rate envelope at 47s, 44.1kHz."""
    df = load_csv()
    n_depths = len(USABLE_DEPTHS)
    fig, axes = plt.subplots(n_depths, 1, figsize=(14, 1.6 * n_depths), sharex=True)

    for ax, d in zip(axes, USABLE_DEPTHS):
        start, length = find_first_full_year(df, d)
        env = envelope_for_depth(
            df,
            d,
            target_samples=TARGET_N,
            start_day=start,
            n_days=length,
            floor=0.1,
            smooth_kernel=int(SR * 0.5),  # 0.5s smoothing
        )
        t = np.arange(TARGET_N) / SR
        ax.plot(t, env.values, linewidth=0.8)
        ax.fill_between(t, 0, env.values, alpha=0.2)
        year_start = df["Year"].iloc[start] if start < len(df) else "?"
        ax.set_ylabel(f"{d} cm", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.text(
            0.99,
            0.85,
            f"raw min={env.raw_min:.1f}°C  max={env.raw_max:.1f}°C  std={env.raw_std:.2f}  start_year={year_start}",
            transform=ax.transAxes,
            ha="right",
            fontsize=7,
            family="monospace",
            color="gray",
        )
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(f"Time within {DURATION_S}s clip (= 1 year compressed)")
    fig.suptitle(
        f"Per-depth envelopes — first full year, resampled to {DURATION_S}s @ {SR}Hz, "
        f"min-max normalized to [0.1, 1.0], 0.5s smoothing",
        y=1.0,
    )
    fig.tight_layout()
    out_path = OUT / "soil_envelopes_per_depth_year1.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[save] {out_path}")


def plot_full_history_compressed():
    """Whole 27-year history compressed to one 47s clip per depth."""
    df = load_csv()
    n_depths = len(USABLE_DEPTHS)
    fig, axes = plt.subplots(n_depths, 1, figsize=(14, 1.6 * n_depths), sharex=True)

    for ax, d in zip(axes, USABLE_DEPTHS):
        env = envelope_for_depth(
            df,
            d,
            target_samples=TARGET_N,
            start_day=0,
            n_days=None,
            floor=0.1,
            smooth_kernel=int(SR * 0.5),
        )
        t = np.arange(TARGET_N) / SR
        ax.plot(t, env.values, linewidth=0.8)
        ax.fill_between(t, 0, env.values, alpha=0.2)
        ax.set_ylabel(f"{d} cm", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.text(
            0.99,
            0.85,
            f"raw min={env.raw_min:.1f}°C  max={env.raw_max:.1f}°C  std={env.raw_std:.2f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=7,
            family="monospace",
            color="gray",
        )
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(f"Time within {DURATION_S}s clip (= 27 years compressed)")
    fig.suptitle(
        f"Per-depth envelopes — full 1993-2023 history, compressed to {DURATION_S}s @ {SR}Hz",
        y=1.0,
    )
    fig.tight_layout()
    out_path = OUT / "soil_envelopes_per_depth_full.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    plot_full_history()
    plot_one_year_per_depth()
    plot_full_history_compressed()
