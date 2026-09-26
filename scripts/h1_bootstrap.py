"""95% moving-block bootstrap confidence intervals for the H1 ratios of medians
(SYCL->OpenCL versus native OpenCL on the same Intel runtime, platform P1, isolated).

Usage (from the repository root):  python scripts/h1_bootstrap.py
Blocks of 50 consecutive samples preserve the autocorrelation of each series.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
B, L, SEED = 4000, 50, 2027
rng = np.random.default_rng(SEED)


def block_bootstrap_medians(x):
    n = len(x)
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(L)).reshape(B, -1)[:, :n]
    return np.median(x[idx], axis=1)


d = pd.read_csv(os.path.join(ROOT, "data", "p1_results.csv"), keep_default_na=False)
d = d[d.scenario == "iso"]
rows = []
for (kernel, size), g in d.groupby(["kernel", "size"]):
    base = g[g.impl == "intel-ocl"].sort_values("iter").lat_ns.to_numpy(float)
    base_bs = block_bootstrap_medians(base)
    for impl in ["acpp-ocl-jit-usm", "acpp-ocl-jit-buf"]:
        x = g[g.impl == impl].sort_values("iter").lat_ns.to_numpy(float)
        ratio = np.median(x) / np.median(base)
        lo, hi = np.percentile(block_bootstrap_medians(x) / base_bs, [2.5, 97.5])
        rows.append(dict(kernel=kernel, size=size, impl=impl, ratio=ratio, ci_lo=lo, ci_hi=hi))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "data", "h1_bootstrap_ci.csv"), index=False)
print(out.round(2).to_string(index=False))
