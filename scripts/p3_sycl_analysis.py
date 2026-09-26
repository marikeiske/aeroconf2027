#!/usr/bin/env python3
"""H1 on P3: SYCL (Intel oneAPI DPC++ 2025.1 over the Intel OpenCL CPU runtime) vs OpenCL (same runtime,
same device: Ryzen 9 5900X, pinned to one logical CPU). Same statistics as P1: ratio of medians with
95% moving-block bootstrap CI (block 50, 4000 resamples), run-to-run CoV, interference ratios, overhead, EVT."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import analysis
from analysis import load, summarize, pwcet, D
analysis.LABEL.update({
    ("opencl", "intel-ocl", "cpu"): ("OpenCL Intel", "OpenCL"),
    ("sycl", "dpcpp-ocl-jit-usm", "cpu"): ("SYCL DPC++ JIT USM", "SYCL"),
    ("sycl", "dpcpp-ocl-jit-buf", "cpu"): ("SYCL DPC++ JIT buf", "SYCL"),
    ("sycl", "dpcpp-ocl-aot-usm", "cpu"): ("SYCL DPC++ AOT USM", "SYCL")})
B, L, SEED = 4000, 50, 2027
rng = np.random.default_rng(SEED)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)

def bb_medians(x):
    n = len(x); nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(L)).reshape(B, -1)[:, :n]
    return np.median(x[idx], axis=1)

reps = [r for r in (1, 2, 3) if os.path.exists(os.path.join(D, f"p3s_rep{r}.csv"))]
d = pd.concat([load(f"p3s_rep{r}.csv", rep=r) for r in reps], ignore_index=True)
d["kernel"] = d["kernel"].fillna("null")

S = summarize(d[d.rep == 1]); S.to_csv(os.path.join(D, "p3s_summary.csv"), index=False)
print(S[["label", "kernel", "size", "scenario", "median", "p999", "max", "cov", "tail999"]].round(2).to_string())

# H1: ratio of medians vs OpenCL, isolated, rep 1 (same protocol as P1), with block-bootstrap CI
rows = []
iso = d[(d.rep == 1) & (d.scenario == "iso")]
for (k, n), g in iso.groupby(["kernel", "size"]):
    base = g[g.impl == "intel-ocl"].sort_values("iter").lat_ns.to_numpy(float)
    bb = bb_medians(base)
    for impl in ["dpcpp-ocl-jit-usm", "dpcpp-ocl-jit-buf", "dpcpp-ocl-aot-usm"]:
        x = g[g.impl == impl].sort_values("iter").lat_ns.to_numpy(float)
        if len(x) == 0: continue
        lo, hi = np.percentile(bb_medians(x) / bb, [2.5, 97.5])
        rows.append(dict(kernel=k, size=n, impl=impl, ratio=np.median(x) / np.median(base), ci_lo=lo, ci_hi=hi,
                         ocl_median_us=np.median(base) / 1e3))
H1 = pd.DataFrame(rows); H1.to_csv(os.path.join(D, "p3s_h1_bootstrap_ci.csv"), index=False)
print("\n=== H1 on P3 (SYCL DPC++ / OpenCL Intel, same device, iso)"); print(H1.round(3).to_string(index=False))

# H1 robustness: same ratio per repetition
rr = []
for r in reps:
    s = summarize(d[(d.rep == r) & (d.scenario == "iso")]).set_index(["impl", "kernel", "size"])["median"]
    for (impl, k, n), v in s.items():
        if impl.startswith("dpcpp"): rr.append(dict(rep=r, impl=impl, kernel=k, size=n, ratio=v / s[("intel-ocl", k, n)]))
RR = pd.DataFrame(rr).pivot_table(index=["impl", "kernel", "size"], columns="rep", values="ratio")
print("\n=== H1 ratio per repetition"); print(RR.round(3).to_string())

# run-to-run CoV of medians
R = summarize(d.assign(platform=d.platform + "_r" + d.rep.astype(str)))
rob = R.groupby(["label", "kernel", "size", "scenario"])["median"].agg(lambda x: x.std() / x.mean())
print("\nrun-to-run CoV of medians: mean %.3f, max %.3f" % (rob.mean(), rob.max()))

# H3: interference ratios
a = S[S.scenario == "iso"].set_index(["label", "kernel", "size"]); b = S[S.scenario == "intf"].set_index(["label", "kernel", "size"])
j = pd.DataFrame({"r_med": b["median"] / a["median"], "r_p999": b["p999"] / a["p999"]}).dropna()
print("\n=== H3 intf/iso"); print(j.round(3).to_string())

# H2: overhead
ov = pd.read_csv(os.path.join(D, "p3s_overhead.csv.meta.csv"))
agg = ov.groupby(["impl", "scenario"])[["t_init_ns", "t_build_ns", "t_first_ns"]].median() / 1e6
agg["startup_total_ms"] = agg.sum(axis=1)
print("\n=== H2 overhead (ms, median of 30 fresh processes)"); print(agg.round(2).to_string())
agg.to_csv(os.path.join(D, "p3s_overhead_summary.csv"))

# EVT
rows = []
for key, x in d[(d.rep == 1)].groupby(["impl", "kernel", "size", "scenario"]):
    v = x.lat_ns.values / 1e3
    if len(v) < 200: continue
    r = pwcet(v)
    rows.append(dict(zip(["impl", "kernel", "size", "scenario"], key), xi=r["xi"], lb=r["lb_p"], ks=r["ks_p"], gof=r["gof_p"],
                     valid=(r["lb_p"] > .05) and (r["ks_p"] > .05) and (r["gof_p"] > .05)))
E = pd.DataFrame(rows); E.to_csv(os.path.join(D, "p3s_evt.csv"), index=False)
print(f"\n=== EVT: {len(E)} series | indep. rejected {(E.lb<=.05).mean():.0%} | i.d. rejected {(E.ks<=.05).mean():.0%} "
      f"| GoF rejected {(E.gof<=.05).mean():.0%} | xi>0 {(E.xi>0).mean():.0%} | valid {E.valid.sum()}/{len(E)}")
