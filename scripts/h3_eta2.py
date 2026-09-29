"""H3: share of variance (eta squared, one-way) explained by each factor.

Reproduces the values reported for H3 in the paper (campaign 1 of each platform):
  - median inflation under interference, log(median_intf / median_iso);
  - tail ratio under interference, log(p99.9 / median) in the intf scenario.
Factors: configuration (programming model + implementation + device), kernel, problem size.
On P2 the configuration is confounded with the device (CPU baseline vs. iGPU).

Usage (from the repository root):  python scripts/h3_eta2.py
Input: data/summary.csv (produced by scripts/analysis.py).  Output: data/h3_eta2.csv
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KEY = ["label", "kernel", "size"]
FACTORS = {"configuration": "label", "kernel": "kernel", "size": "size"}


def eta2(df, y, factor):
    """One-way eta squared: between-group sum of squares / total sum of squares."""
    m = df[y].mean()
    ssb = sum(len(g) * (g[y].mean() - m) ** 2 for _, g in df.groupby(factor))
    sst = ((df[y] - m) ** 2).sum()
    return ssb / sst


S = pd.read_csv(os.path.join(ROOT, "data", "summary.csv"))
rows = []
for plat in ["P1", "P2"]:
    s = S[(S.platform == plat) & (S.kernel != "null")]
    iso = s[s.scenario == "iso"].set_index(KEY)
    intf = s[s.scenario == "intf"].set_index(KEY)
    j = iso[["median"]].join(intf[["median", "p999"]], lsuffix="_iso", rsuffix="_intf", how="inner").reset_index()
    j["median_inflation"] = np.log(j.median_intf / j.median_iso)
    j["tail_ratio_intf"] = np.log(j.p999 / j.median_intf)
    for y in ["median_inflation", "tail_ratio_intf"]:
        for name, col in FACTORS.items():
            rows.append(dict(platform=plat, response=y, factor=name, n_series=len(j), eta2=eta2(j, y, col)))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "data", "h3_eta2.csv"), index=False)
print(out.pivot_table(index=["platform", "response"], columns="factor", values="eta2").round(3).to_string())
