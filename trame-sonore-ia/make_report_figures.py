"""
Generate the figures used in the methods report.

Produces in outputs/report_figs/:
  - 01_pipeline.png           (high-level data → audio block diagram)
  - 02_depth_voices.png       (per-depth instrument + role table)
  - 03_segments_timeline.png  (27-year history mapped to 4 audio segments)
  - 04_envelope_concept.png   (single-depth illustration of how temp → envelope)
  - 05_mix_stack.png          (7 stems stacked, weighted into final mix)

Reuses existing figures:
  - outputs/soil_envelopes_per_depth_year1.png
  - outputs/soil_full_history.png
"""
from __future__ import annotations
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from soil_data import extract_depth_series, find_first_full_year, load_csv, to_envelope

OUT = Path("outputs/report_figs")
OUT.mkdir(parents=True, exist_ok=True)


# ---------- Figure 1: Pipeline overview (both versions in one diagram) ----------
def make_pipeline():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.5, 7.5)
    ax.axis("off")

    def box(x, y, w, h, text, color="#dceaf3", text_size=9, bold=False,
            edgecolor="#3d5d75", linestyle="solid"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor=edgecolor, linewidth=1.2,
            linestyle=linestyle,
        )
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=text_size, fontweight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, color="#3d5d75", linestyle="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color=color,
                                     linestyle=linestyle))

    # Input data (left column)
    box(0.2, 5.7, 2.8, 0.9,
        "Temperature CSV\n(1993-2023, daily,\n7 usable depths)",
        color="#e8f3dc", bold=True)
    box(0.2, 4.1, 2.8, 0.9,
        "EMD microtonal chord\nper depth\n(from MIDI)",
        color="#f3ece0", bold=True, edgecolor="#a06030")

    # Per-depth processing (middle-left)
    box(4.0, 5.7, 3.0, 0.9,
        "Per-depth envelope\n(yearly+multi-year curve)")
    box(4.0, 4.1, 3.0, 0.9,
        "Chord synthesis\n(additive sines)",
        edgecolor="#a06030")
    box(4.0, 2.5, 3.0, 0.9,
        "Per-depth + per-segment\nprompt language")

    # Right column - generation
    box(8.0, 4.4, 2.7, 1.7,
        "Stable Audio Open\n47 s stereo wav\nx 7 depths x 4 segments",
        color="#f3e3dc", bold=True)

    box(8.0, 1.9, 2.7, 1.5,
        "Crossfade-stitch\n7 streams (3 min each)\n+ multi-effect envelope\n+ RMS-balanced sum",
        color="#f3dce8", bold=True)

    box(4.0, 0.4, 3.0, 0.9,
        "Final 3-min layered mix",
        color="#dce4f3", bold=True, text_size=11)

    # Arrows
    arrow(3.0, 6.15, 4.0, 6.15)               # CSV -> envelope
    arrow(3.0, 4.55, 4.0, 4.55)               # MIDI -> chord synth
    arrow(7.0, 6.15, 8.0, 5.6)                # envelope -> generation (top entry)
    arrow(7.0, 4.55, 8.0, 5.05)               # chord synth -> generation
    arrow(7.0, 2.95, 8.0, 4.55)               # prompts -> generation
    arrow(9.35, 4.4, 9.35, 3.4)               # generation -> stitch
    arrow(8.0, 2.65, 7.0, 1.0)                # stitch -> final mix

    # Labels marking which path belongs to which version
    ax.text(7.5, 6.4, "Version 1 + 2", fontsize=8, color="#3d5d75",
            style="italic", ha="center")
    ax.text(7.5, 4.85, "Version 2 only", fontsize=8, color="#a06030",
            style="italic", ha="center")
    ax.text(7.5, 3.25, "Version 1 + 2", fontsize=8, color="#3d5d75",
            style="italic", ha="center")

    ax.text(5.5, 7.15, "Tasiujaq sonification pipeline — two versions delivered",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(5.5, 6.85,
            "Version 1 (blue): envelope-only.   Version 2 (brown): adds microtonal chord seed at generation.",
            ha="center", fontsize=9, style="italic", color="#3d5d75")

    fig.tight_layout()
    out = OUT / "01_pipeline.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[save] {out}")


# ---------- Figure 2: Depth → voice table ----------
def make_depth_voices():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0.2, 8)
    ax.axis("off")

    headers = ["Depth", "Physical character", "Musical voice (v10)", "Role"]
    rows = [
        ("200 cm",  "Volatile (std ~ 3.4 °C)",   "Soft soprano + celesta",        "Melodic top voice"),
        ("300 cm",  "Volatile (std ~ 2.1 °C)",   "Piano + warm Rhodes",           "Melodic mid voice"),
        ("400 cm",  "Moderate (std ~ 1.2 °C)",   "Muted viola + humming",         "Lead lament"),
        ("500 cm",  "Moderate (std ~ 1.1 °C)",   "Wordless choir",                "Harmonic pad"),
        ("700 cm",  "Stable (std ~ 0.7 °C)",     "Taiko + pizzicato cello",       "Rhythmic build"),
        ("900 cm",  "Stable (std ~ 0.36 °C)",    "Cello section + male bass",     "Low foundation"),
        ("1100 cm", "Near-frozen (std ~ 0.36 °C)", "Permafrost + throat singing",  "Sub-bass drone"),
    ]

    # Column x positions
    xs = [0.3, 2.2, 5.1, 8.5]
    col_w = [1.9, 2.85, 3.35, 2.4]
    row_h = 0.85
    y0 = 6.6

    # Header
    for x, w, h in zip(xs, col_w, headers):
        rect = mpatches.Rectangle((x, y0), w, row_h, facecolor="#3d5d75", edgecolor="white")
        ax.add_patch(rect)
        ax.text(x + w / 2, y0 + row_h / 2, h, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")

    # Rows — color-coded by depth (surface = warm, deep = cool blue)
    colors = ["#f7d6c1", "#f5d5b7", "#f0c8a4", "#e8cdb0", "#c9d8e8", "#b3c2da", "#9fb0cc"]
    for i, (row, color) in enumerate(zip(rows, colors)):
        y = y0 - (i + 1) * row_h
        for x, w, val in zip(xs, col_w, row):
            rect = mpatches.Rectangle((x, y), w, row_h, facecolor=color, edgecolor="white")
            ax.add_patch(rect)
            ax.text(x + w / 2, y + row_h / 2, val, ha="center", va="center", fontsize=9)

    ax.text(5.5, 7.6, "Per-depth voice assignment",
            ha="center", fontsize=13, fontweight="bold")

    fig.tight_layout()
    out = OUT / "02_depth_voices.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[save] {out}")


# ---------- Figure 3: Segments timeline ----------
def make_segments_timeline():
    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Top bar: 27-year history
    years = list(range(1993, 2024))
    seg_year_boundaries = [1993, 2000, 2007, 2014, 2023]
    seg_labels = [
        "Pristine\n(1993-2000)",
        "Early warming\n(2000-2007)",
        "Active thaw\n(2007-2014)",
        "Late warming\n(2014-2023)",
    ]
    seg_colors = ["#a4cce0", "#f0c995", "#e89868", "#d96e3f"]

    # Top: historical timeline
    y_top, h = 2.2, 0.75
    for i, (start, end, label, color) in enumerate(
        zip(seg_year_boundaries[:-1], seg_year_boundaries[1:], seg_labels, seg_colors)
    ):
        ax.add_patch(mpatches.Rectangle((start, y_top), end - start, h,
                                         facecolor=color, edgecolor="white", linewidth=2))
        ax.text((start + end) / 2, y_top + h / 2, label,
                ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(1993, y_top + h + 0.25, "27 years of soil temperature record",
            fontsize=10, fontweight="bold")
    ax.text(1993, y_top - 0.3, "1993", fontsize=8, ha="center")
    for yr in [2000, 2007, 2014, 2023]:
        ax.text(yr, y_top - 0.3, str(yr), fontsize=8, ha="center")

    # Mapping arrows
    audio_x_starts = [1993, 2000.5, 2008, 2015.5]
    audio_x_widths = [7, 7, 7, 7]
    audio_seg_t = ["0 - 0:44", "0:44 - 1:28", "1:28 - 2:12", "2:12 - 2:59"]
    y_bottom = 0.6
    h2 = 0.75
    # Audio segments below
    for i, (start, w, color, t_label) in enumerate(
        zip(audio_x_starts, audio_x_widths, seg_colors, audio_seg_t)
    ):
        ax.add_patch(mpatches.Rectangle((start, y_bottom), w, h2,
                                         facecolor=color, edgecolor="white", linewidth=2))
        ax.text(start + w / 2, y_bottom + h2 / 2, t_label,
                ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(1993, y_bottom + h2 + 0.25,
            "3-minute audio piece (4 segments × 47 s, 3 s crossfade)",
            fontsize=10, fontweight="bold")

    # Connecting arrows
    for hist_x, audio_x in zip(
        [(1993 + 2000) / 2, (2000 + 2007) / 2, (2007 + 2014) / 2, (2014 + 2023) / 2],
        [s + w / 2 for s, w in zip(audio_x_starts, audio_x_widths)]
    ):
        ax.annotate("", xy=(audio_x, y_bottom + h2 + 0.04),
                    xytext=(hist_x, y_top - 0.04),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#3d5d75"))

    ax.set_xlim(1992, 2024)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Time mapping — 27 years → 3 minutes",
                 fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    out = OUT / "03_segments_timeline.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[save] {out}")


# ---------- Figure 4: envelope concept (one depth, ~6.7y per segment) ----------
def make_envelope_concept():
    """
    Shows the actual mapping used in generate_3min.py:
    one segment = ~6.7 years of data compressed to 47s of audio.
    """
    df = load_csv()
    d = "200"
    # First 6.7-year segment ( == segment_day_window(n_total, 0) for n_total=9748 )
    n_total = len(df)
    seg_len = n_total // 4
    series = extract_depth_series(df, d)[0:seg_len]
    n_days = len(series)
    n_years = n_days / 365.25

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=False)

    # (a) Raw temperature curve over ~6.7 years.
    # IMPORTANT: the CSV starts in August 1993 (Year=1993, Month=8, MI=0), NOT January.
    # Build the real time axis from the actual Year/Month/Measurement Index of each row.
    seg_df = df.iloc[0:seg_len]
    start_year_decimal = (
        float(seg_df["Year"].iloc[0])
        + (float(seg_df["Month"].iloc[0]) - 1) / 12.0
        + float(seg_df["Measurement Index"].iloc[0]) / 365.25
    )
    years_axis = np.arange(n_days) / 365.25 + start_year_decimal
    axes[0].plot(years_axis, series, color="#3d5d75", lw=0.7)
    axes[0].set_title(
        f"(a) Raw soil temperature — depth 200 cm — segment 1 ({n_years:.1f} years of daily data)",
        loc="left", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_xlabel("Year")
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(years_axis[0], years_axis[-1])
    # Vertical lines at January 1 of each year so the reader can verify that
    # peaks land in summer (mid-year), troughs near January.
    for jan in range(int(np.ceil(years_axis[0])), int(np.floor(years_axis[-1])) + 1):
        axes[0].axvline(jan, color="#bbbbbb", lw=0.5, linestyle=":")

    # (b) Normalized envelope at audio rate (47s @ 44.1kHz, downsampled for display)
    target_n_audio = int(47.0 * 44100)
    env_full = to_envelope(series, target_samples=target_n_audio, floor=0.0,
                            smooth_kernel=int(44100 * 0.5))
    # Downsample for plotting
    plot_step = max(1, target_n_audio // 4000)
    env_plot = env_full[::plot_step]
    t_plot = np.arange(len(env_plot)) * plot_step / 44100.0
    axes[1].plot(t_plot, env_plot, color="#a52a2a", lw=1.0)
    axes[1].fill_between(t_plot, 0, env_plot, alpha=0.2, color="#a52a2a")
    axes[1].set_title(
        f"(b) → Normalized envelope at audio rate (the same {n_years:.1f} years compressed to 47 s)",
        loc="left", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Envelope (0-1)")
    axes[1].set_xlabel("Audio time within segment (s)")
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlim(0, 47)

    # (c) Illustrative audio shaped by envelope
    n_demo = 47 * 4000  # 4kHz mini sample for fast plotting
    rng = np.random.default_rng(0)
    base = rng.standard_normal(n_demo) * 0.4
    env_up = np.interp(np.linspace(0, 1, n_demo),
                       np.linspace(0, 1, len(env_plot)), env_plot)
    audio = base * (0.4 + 0.6 * env_up)
    t_audio = np.linspace(0, 47, n_demo)
    axes[2].plot(t_audio, audio, color="#3d5d75", lw=0.4, alpha=0.6)
    axes[2].plot(t_audio, 0.4 + 0.6 * env_up, color="#a52a2a", lw=1.5,
                 label="Envelope-derived gain")
    axes[2].plot(t_audio, -(0.4 + 0.6 * env_up), color="#a52a2a", lw=1.5)
    axes[2].set_title(
        "(c) → Audio amplitude shaped by the envelope (each seasonal cycle ≈ 7 s within segment)",
        loc="left", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Audio amplitude")
    axes[2].set_xlabel("Audio time within segment (s)")
    axes[2].grid(alpha=0.3)
    axes[2].set_xlim(0, 47)
    axes[2].set_ylim(-1.0, 1.0)
    axes[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = OUT / "04_envelope_concept.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[save] {out}")


# ---------- Figure 5: Mix stack ----------
def make_mix_stack():
    fig, ax = plt.subplots(figsize=(11, 6))

    depths = [200, 300, 400, 500, 700, 900, 1100]
    names = [
        "Soprano + celesta",
        "Piano + Rhodes",
        "Viola + humming",
        "Wordless choir",
        "Taiko + pizz cello",
        "Cello + male bass",
        "Permafrost + throat",
    ]
    gains_db = [-2.4, 5.9, 1.5, 2.3, 13.9, 8.4, 12.1]  # from v10 mix
    # Colors: warm → cool
    colors = ["#f7d6c1", "#f5d5b7", "#f0c8a4", "#e8cdb0", "#c9d8e8", "#b3c2da", "#9fb0cc"]

    y_positions = np.arange(len(depths))[::-1]  # 200 at top
    bar_height = 0.7

    for y, d, n, g, c in zip(y_positions, depths, names, gains_db, colors):
        ax.add_patch(mpatches.Rectangle((0, y - bar_height / 2), 7, bar_height,
                                         facecolor=c, edgecolor="#444", linewidth=0.8))
        ax.text(0.15, y, f"{d} cm", va="center", fontsize=10, fontweight="bold")
        ax.text(2.0, y, n, va="center", fontsize=10)
        ax.text(6.85, y, f"{g:+.1f} dB", va="center", ha="right",
                fontsize=10, family="monospace", color="#404040")

    # Final mix arrow
    ax.annotate("",
                xy=(8.5, 3), xytext=(7.2, 3),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#444"))
    rect = mpatches.FancyBboxPatch(
        (8.6, 2.3), 2.2, 1.4,
        boxstyle="round,pad=0.05",
        facecolor="#dce4f3", edgecolor="#3d5d75", linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(9.7, 3, "Final\n3-min mix",
            ha="center", va="center", fontsize=11, fontweight="bold")

    ax.set_xlim(-0.2, 11)
    ax.set_ylim(-0.5, 8)
    ax.axis("off")
    ax.set_title("Mix balance — per-stem gains (after RMS equalization)",
                 fontsize=13, fontweight="bold", pad=10)

    fig.tight_layout()
    out = OUT / "05_mix_stack.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[save] {out}")


# ---------- Figure 6: chord seed construction (Version 2) ----------
def make_chord_seed():
    """
    Illustrate how the EMD chord becomes the diffusion seed for Version 2.

      (a) All partials per depth (stem plot, color-coded by depth).
      (b) For one depth (200 cm): partials after max_freq cap, weighted by
          1/(rank^exp) — these are the sine amplitudes used in synthesis.
      (c) Where the chord seed enters the diffusion process (sigma schedule).
    """
    import json
    from pathlib import Path

    harmonies = json.loads(Path("outputs/depth_harmonies.json").read_text(encoding="utf-8"))
    depth_list = ["200", "300", "400", "500", "700", "900", "1100"]
    depth_colors = plt.cm.coolwarm(np.linspace(0, 1, len(depth_list)))

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1.2, 1.0], hspace=0.55)

    # (a) All partials per depth — stem plot, log-x axis
    ax_a = fig.add_subplot(gs[0])
    for di, d in enumerate(depth_list):
        freqs = harmonies[d]["raw_freqs"]
        n = len(freqs)
        weights = 1.0 / (np.arange(n) + 1) ** 0.6
        freqs_sorted = sorted(freqs)
        ax_a.scatter(freqs_sorted, [di] * n, s=40,
                     c=[depth_colors[di]], edgecolors="black", lw=0.4, zorder=3)
        for f, _ in zip(freqs_sorted, weights):
            ax_a.plot([f, f], [di - 0.18, di + 0.18], color=depth_colors[di], lw=1.4, zorder=2)
    ax_a.set_yticks(range(len(depth_list)))
    ax_a.set_yticklabels([f"{d} cm" for d in depth_list], fontsize=9)
    ax_a.set_xscale("log")
    ax_a.set_xlim(20, 22000)
    ax_a.set_xlabel("Frequency (Hz, log)")
    ax_a.set_title("(a) All partials per depth, extracted from the microtonal MIDI",
                   loc="left", fontsize=11, fontweight="bold")
    ax_a.grid(alpha=0.3, axis="x")
    # Annotation: shared low fundamental
    ax_a.axvspan(25, 35, alpha=0.18, color="orange", zorder=1)
    ax_a.text(30, len(depth_list) - 0.4,
              "shared\nsub-bass\nfundamental",
              ha="center", va="top", fontsize=8, color="#a06030", style="italic")

    # (b) One depth (200 cm) — partials after max_freq=2000 cap, weighted
    ax_b = fig.add_subplot(gs[1])
    freqs_200 = sorted(harmonies["200"]["raw_freqs"])
    max_freq = 2000.0
    weight_exp = 1.5
    kept = [f for f in freqs_200 if f <= max_freq]
    dropped = [f for f in freqs_200 if f > max_freq]
    weights_kept = [1.0 / (i + 1) ** weight_exp for i in range(len(kept))]
    for f, w in zip(kept, weights_kept):
        ax_b.plot([f, f], [0, w], color="#3d5d75", lw=2.5)
        ax_b.scatter([f], [w], s=50, c="#3d5d75", zorder=3)
        ax_b.annotate(f"{f:.0f} Hz\nw={w:.2f}",
                      xy=(f, w), xytext=(0, 6), textcoords="offset points",
                      ha="center", fontsize=7, color="#3d5d75")
    for f in dropped:
        ax_b.plot([f, f], [0, 0.04], color="#bbbbbb", lw=2.0)
        ax_b.scatter([f], [0.04], s=30, c="white", edgecolors="#bbbbbb", lw=0.8, zorder=3)
    ax_b.axvline(max_freq, color="orange", linestyle="--", lw=1.0, alpha=0.7)
    ax_b.text(max_freq * 1.05, 0.95, f"max_freq cap = {max_freq:.0f} Hz",
              fontsize=8, color="#a06030", style="italic")
    ax_b.set_xscale("log")
    ax_b.set_xlim(20, 22000)
    ax_b.set_ylim(0, 1.15)
    ax_b.set_xlabel("Frequency (Hz, log)")
    ax_b.set_ylabel("Sine amplitude")
    ax_b.set_title(
        "(b) Chord seed for one depth (200 cm) — high partials dropped (grey), low partials weighted",
        loc="left", fontsize=11, fontweight="bold")
    ax_b.grid(alpha=0.3, axis="x")

    # (c) Where the chord seed enters the diffusion process
    ax_c = fig.add_subplot(gs[2])
    n_steps = 200
    sigmas = np.linspace(500, 0.3, n_steps)  # rough sketch of SAO's schedule
    # Real shape: cosine-like; use log to approximate
    step_axis = np.arange(n_steps)
    sigmas = 500 * (0.3 / 500) ** (step_axis / (n_steps - 1))
    ax_c.semilogy(step_axis, sigmas, color="#3d5d75", lw=1.6)
    # Two regions
    strength = 0.85
    start_idx = int(round(n_steps * (1.0 - strength)))
    ax_c.axvspan(0, start_idx, alpha=0.2, color="#bbbbbb")
    ax_c.axvspan(start_idx, n_steps, alpha=0.25, color="#f3ece0")
    ax_c.axvline(start_idx, color="#a06030", lw=1.2)
    ax_c.text(start_idx / 2, 200,
              "skipped:\nfull-noise denoising\n(would erase the seed)",
              ha="center", va="center", fontsize=9, color="#666666")
    ax_c.text((start_idx + n_steps) / 2, 200,
              "denoising starts from\nchord seed + small noise\n(seed has audible effect)",
              ha="center", va="center", fontsize=9, color="#a06030")
    ax_c.set_xlim(0, n_steps)
    ax_c.set_ylim(0.2, 600)
    ax_c.set_xlabel("Diffusion step")
    ax_c.set_ylabel("Noise σ (log)")
    ax_c.set_title(
        f"(c) Chord seed enters the diffusion at step {start_idx}/{n_steps} (strength = {strength})",
        loc="left", fontsize=11, fontweight="bold")
    ax_c.grid(alpha=0.3, which="both")

    fig.suptitle("Version 2 — Microtonal chord seeding of the diffusion process",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    out = OUT / "06_chord_seed.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[save] {out}")


if __name__ == "__main__":
    make_pipeline()
    make_depth_voices()
    make_segments_timeline()
    make_envelope_concept()
    make_mix_stack()
    make_chord_seed()
    print("\nDone. Figures in:", OUT.resolve())
