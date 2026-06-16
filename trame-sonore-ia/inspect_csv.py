"""
Inspect the soil temperature CSV: time axis, per-depth NaN coverage, value ranges,
and continuity. Outputs a textual summary to stdout.
"""
import sys
import pandas as pd
import numpy as np

CSV = r"C:\Users\skite\Downloads\temperature_curves_by_depth_Tasiujaq.csv"

df = pd.read_csv(CSV)
print(f"=== File: {CSV}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

# Identify depth columns (everything except Year, Month, Measurement Index)
meta_cols = ["Year", "Month", "Measurement Index"]
depth_cols_raw = [c for c in df.columns if c not in meta_cols]
# Sort numerically (column names are strings of integers like "0", "10", "100")
depth_cols = sorted(depth_cols_raw, key=lambda c: int(c))
print(f"Depths (cm, sorted): {depth_cols}")
print()

# Time axis inspection
print("=== Time axis")
print(f"Year range: {df['Year'].min()} – {df['Year'].max()}")
print(f"Year/Month/MeasIdx unique counts:")
print(f"  Years: {df['Year'].nunique()}")
print(f"  Year-Month pairs: {df.groupby(['Year','Month']).ngroups}")
mi = df["Measurement Index"]
print(f"  Measurement Index range: {mi.min()} – {mi.max()}")
# How many measurements per (year, month)?
per_ym = df.groupby(["Year", "Month"]).size()
print(f"  Measurements per (year,month): min={per_ym.min()}, median={int(per_ym.median())}, max={per_ym.max()}")
# Likely sampling: if median is ~30/31, it's daily; if ~720, hourly; etc.
print()

# Per-depth coverage
print("=== Per-depth coverage")
print(f"{'depth(cm)':>10} {'n_valid':>10} {'%filled':>8} {'min':>8} {'max':>8} {'mean':>8} {'std':>8}  "
      f"{'first_valid_idx':>15} {'last_valid_idx':>14} {'longest_continuous_run':>22}")
for d in depth_cols:
    s = df[d]
    valid = s.notna()
    n = int(valid.sum())
    pct = 100.0 * n / len(s)
    if n == 0:
        print(f"{d:>10} {n:>10} {pct:>7.1f}%   (no data)")
        continue
    # Longest continuous run of valid values
    arr = valid.values.astype(int)
    # Count run lengths of 1s
    runs = []
    cur = 0
    for x in arr:
        if x:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    longest = max(runs) if runs else 0
    print(
        f"{d:>10} {n:>10} {pct:>7.1f}% "
        f"{s.min():>8.2f} {s.max():>8.2f} {s.mean():>8.2f} {s.std():>8.2f}  "
        f"{int(valid.idxmax()):>15} {int(valid[::-1].idxmax()):>14} {longest:>22}"
    )
print()

# Suggested usable depths: those with at least 80% coverage
print("=== Recommendation")
usable = [d for d in depth_cols if df[d].notna().mean() >= 0.8]
print(f"Depths with >=80% coverage: {usable}")
mostly_usable = [d for d in depth_cols if df[d].notna().mean() >= 0.5]
print(f"Depths with >=50% coverage: {mostly_usable}")
