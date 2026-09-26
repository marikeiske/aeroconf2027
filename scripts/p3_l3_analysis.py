#!/usr/bin/env python3
"""H3 on P3: co-runner on the same CCD (shared L3, intf_l3) vs on the other CCD (intf, earlier campaigns)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import analysis
from analysis import load, summarize, pwcet, D
analysis.LABEL.update({("opencl", "nvidia-ocl", "gpu"): ("OpenCL NVIDIA dGPU", "OpenCL"),
                       ("opencl", "intel-ocl", "cpu"): ("OpenCL Intel", "OpenCL"),
                       ("sycl", "dpcpp-ocl-jit-usm", "cpu"): ("SYCL DPC++ JIT USM", "SYCL")})
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
K = ["label", "kernel", "size"]

def ratios(S, sc):
    a = S[S.scenario == "iso"].set_index(K); b = S[S.scenario == sc].set_index(K)
    return pd.DataFrame({"med": b["median"] / a["median"], "p999": b["p999"] / a["p999"], "e2e": b["e2e_median"] / a["e2e_median"]}).dropna(how="all")

l3 = pd.concat([load(f"p3l3_rep{r}.csv", rep=r) for r in (1, 2, 3) if os.path.exists(os.path.join(D, f"p3l3_rep{r}.csv"))], ignore_index=True)
l3["kernel"] = l3["kernel"].fillna("null")
# per-repetition ratios (same-CCD), then median across repetitions
per = []
for r, g in l3.groupby("rep"):
    x = ratios(summarize(g), "intf_l3"); x["rep"] = r; per.append(x.reset_index())
P = pd.concat(per)
same = P.groupby(K)[["med", "p999", "e2e"]].median()
rng = P.groupby(K)["med"].agg(["min", "max"]).rename(columns={"min": "med_min", "max": "med_max"})

# cross-CCD (earlier campaigns): p3_rep* (serial, NVIDIA) and p3s_rep* (Intel OCL, SYCL)
x1 = pd.concat([load(f"p3_rep{r}.csv", rep=r) for r in (1, 2, 3)], ignore_index=True)
x2 = pd.concat([load(f"p3s_rep{r}.csv", rep=r) for r in (1, 2, 3)], ignore_index=True)
xc = pd.concat([x1, x2], ignore_index=True); xc["kernel"] = xc["kernel"].fillna("null")
xc = xc[xc.impl.isin(["serial", "nvidia-ocl", "intel-ocl", "dpcpp-ocl-jit-usm"])]
perx = []
for r, g in xc.groupby("rep"):
    x = ratios(summarize(g), "intf"); x["rep"] = r; perx.append(x.reset_index())
cross = pd.concat(perx).groupby(K)[["med", "p999", "e2e"]].median()

T = same.join(rng).join(cross, rsuffix="_xccd")
T = T[["med", "med_min", "med_max", "med_xccd", "p999", "p999_xccd", "e2e", "e2e_xccd"]]
print("=== intf/iso ratios (median over 3 repetitions): same CCD (shared L3) vs other CCD")
print(T.round(3).to_string())
T.to_csv(os.path.join(D, "p3_l3_vs_xccd.csv"))
S1 = summarize(l3[l3.rep == 1])
print("\n=== rep 1 summary"); print(S1[["label", "kernel", "size", "scenario", "median", "p999", "max", "tail999"]].round(2).to_string())
