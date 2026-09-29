"""Warm-up check (Section 5, RQ2): is the steady state reached within the 20 warm-up iterations?

For every steady-state series of campaign 1 on P1 and P2 (null kernel excluded), compare the median
of the first 50 measured samples with the median of the whole series.
The paper reports the median deviation over the 126 series (0.9%) and the share of series whose
deviation stays below 5% (91%).

Usage (from the repository root):  python scripts/warmup_check.py
Input: data/p1_results.csv, data/p2_results.csv.  Output: data/warmup_check.csv
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIRST = 50

d = pd.concat([pd.read_csv(os.path.join(ROOT, "data", f), keep_default_na=False)
               for f in ["p1_results.csv", "p2_results.csv"]], ignore_index=True)
d = d[d.kernel != "null"]
rows = []
for key, g in d.groupby(["platform", "model", "impl", "device", "kernel", "size", "scenario"]):
    v = g.sort_values("iter").lat_ns.to_numpy(float)
    rows.append(dict(zip(["platform", "model", "impl", "device", "kernel", "size", "scenario"], key),
                     n=len(v), median_first50=np.median(v[:FIRST]), median_all=np.median(v),
                     rel_dev=abs(np.median(v[:FIRST]) / np.median(v) - 1)))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "data", "warmup_check.csv"), index=False)
print(f"series: {len(out)}")
print(f"median deviation of the first {FIRST} samples from the series median: {100 * out.rel_dev.median():.1f}%")
print(f"series with deviation below 5%: {100 * (out.rel_dev < 0.05).mean():.0f}%")
print("largest deviations:")
print(out.sort_values("rel_dev", ascending=False).head(5)[["platform", "impl", "kernel", "size", "scenario", "rel_dev"]].round(3).to_string(index=False))
