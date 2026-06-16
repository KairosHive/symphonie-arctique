"""
Soil temperature data utilities for soil → audio conditioning.

CSV format (Tasiujaq):
    Year, Month, Measurement Index, 0, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 700, 900, 1100
    - Daily sampling, 1993-08 through 2023.
    - Each numeric column header is a depth in cm.
    - Many NaN at shallow depths (5, 10, 20 cm have ~54% coverage).
    - Deeper depths (200+ cm) have ~98% coverage.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(r"C:\Users\skite\Downloads\temperature_curves_by_depth_Tasiujaq.csv")
META_COLS = ["Year", "Month", "Measurement Index"]

# Depths with ≥80% coverage in Tasiujaq, sorted shallow → deep
USABLE_DEPTHS = ["0", "50", "100", "150", "200", "300", "400", "500", "700", "900", "1100"]


@dataclass
class Envelope:
    """A normalized envelope curve at a target sample count."""

    depth_cm: int
    values: np.ndarray  # shape (n_target,), float32 in [floor, 1.0]
    raw_min: float  # original temperature range (for metadata / cross-depth scaling)
    raw_max: float
    raw_std: float


def load_csv(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Build a proper monotonic time index. Year-Month-MeasurementIndex maps to a day-ish offset.
    # Measurement Index runs 0..30 within a (year, month). We treat it as day-of-month.
    df = df.copy()
    df["__day_offset"] = (
        (df["Year"] - df["Year"].min()) * 365.25
        + (df["Month"] - 1) * 30.4375
        + df["Measurement Index"]
    )
    df = df.sort_values("__day_offset").reset_index(drop=True)
    return df


def extract_depth_series(df: pd.DataFrame, depth: str | int) -> np.ndarray:
    """
    Extract a depth's temperature series, NaN-filled by linear interpolation
    (with edge-extension at the boundaries).

    Returns float32 array of length len(df).
    """
    col = str(depth)
    s = df[col].copy()
    # Linear interpolate across NaN, then forward/back-fill any remaining edge NaNs.
    s = s.interpolate(method="linear", limit_direction="both")
    return s.to_numpy(dtype=np.float32)


def slice_window(
    series: np.ndarray,
    start: int = 0,
    n: int | None = None,
) -> np.ndarray:
    """Extract a contiguous slice. n=None → from start to end."""
    if n is None:
        return series[start:]
    return series[start : start + n]


def to_envelope(
    series: np.ndarray,
    target_samples: int,
    floor: float = 0.1,
    ceil: float = 1.0,
    smooth_kernel: int = 0,
) -> np.ndarray:
    """
    Resample a soil time series to `target_samples` and normalize to [floor, ceil].

    - Linear interpolation onto the new time grid.
    - Min-max normalize using the series's own range.
    - Optional moving-average smoothing on the resampled signal (kernel in samples).

    Returns float32 array shape (target_samples,).
    """
    series = np.asarray(series, dtype=np.float32)
    src_n = len(series)
    if src_n == 0:
        raise ValueError("empty series")

    # Resample via linear interpolation
    src_x = np.linspace(0.0, 1.0, src_n, dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_samples, dtype=np.float32)
    resampled = np.interp(dst_x, src_x, series).astype(np.float32)

    # Min-max normalize
    lo, hi = float(np.min(resampled)), float(np.max(resampled))
    rng = hi - lo if hi > lo else 1.0
    norm = (resampled - lo) / rng  # [0, 1]
    out = floor + (ceil - floor) * norm  # [floor, ceil]

    # Smooth (simple boxcar via convolution)
    if smooth_kernel and smooth_kernel > 1:
        k = int(smooth_kernel)
        kernel = np.ones(k, dtype=np.float32) / k
        out = np.convolve(out, kernel, mode="same").astype(np.float32)

    return out


def envelope_for_depth(
    df: pd.DataFrame,
    depth: str | int,
    target_samples: int,
    *,
    start_day: int = 0,
    n_days: int | None = None,
    floor: float = 0.1,
    smooth_kernel: int = 0,
) -> Envelope:
    """High-level: load one depth's data, slice a window, build a normalized envelope."""
    raw = extract_depth_series(df, depth)
    window = slice_window(raw, start=start_day, n=n_days)
    values = to_envelope(window, target_samples=target_samples, floor=floor, smooth_kernel=smooth_kernel)
    return Envelope(
        depth_cm=int(depth),
        values=values,
        raw_min=float(np.nanmin(window)),
        raw_max=float(np.nanmax(window)),
        raw_std=float(np.nanstd(window)),
    )


def envelopes_for_all_depths(
    df: pd.DataFrame,
    depths: Iterable[str | int] = USABLE_DEPTHS,
    target_samples: int = 44100,
    **kwargs,
) -> list[Envelope]:
    return [envelope_for_depth(df, d, target_samples=target_samples, **kwargs) for d in depths]


def find_first_full_year(df: pd.DataFrame, depth: str | int) -> tuple[int, int]:
    """
    Find the first contiguous 365-day window in which `depth` has no NaNs.
    Returns (start_index, length=365). Falls back to the longest non-NaN run if no
    365-window exists.
    """
    s = df[str(depth)]
    valid = s.notna().to_numpy()
    n = len(valid)
    target = 365
    # Sliding window of length 365: find first window where all valid
    if n < target:
        target = n
    cumsum = np.concatenate([[0], np.cumsum(valid.astype(int))])
    for i in range(0, n - target + 1):
        window_count = cumsum[i + target] - cumsum[i]
        if window_count == target:
            return i, target
    # Fallback: longest run
    runs = []
    cur_start = -1
    for i, v in enumerate(valid):
        if v and cur_start < 0:
            cur_start = i
        elif not v and cur_start >= 0:
            runs.append((cur_start, i - cur_start))
            cur_start = -1
    if cur_start >= 0:
        runs.append((cur_start, n - cur_start))
    if not runs:
        return 0, n
    best = max(runs, key=lambda x: x[1])
    return best
