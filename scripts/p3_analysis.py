#!/usr/bin/env python3
"""p3_analysis.py - same statistics as analysis.py, applied to platform P3 (discrete NVIDIA GPU),
and compared with P2 (AMD iGPU sharing DRAM) for H1/H2/H3."""
import os, glob, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import load, summarize, pwcet, D

LABEL_EXTRA = {("opencl", "nvidia-ocl", "gpu"): ("OpenCL NVIDIA dGPU", "OpenCL")}
import analysis; analysis.LABEL.update(LABEL_EXTRA)

p3 = pd.concat([load(f"p3_rep{r}.csv", rep=r) for r in (1, 2, 3) if os.path.exists(os.path.join(D, f"p3_rep{r}.csv"))], ignore_index=True)
p2 = pd.concat([load("p2_results.csv", rep=1), load("p2_rep2.csv", rep=2), load("p2_rep3.csv", rep=3)], ignore_index=True)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)

S3 = summarize(p3[p3.rep == 1]); S3.to_csv(os.path.join(D, "p3_summary.csv"), index=False)
S2 = summarize(p2[p2.rep == 1])
print("=== P3 summary (rep 1)")
print(S3[["label", "kernel", "size", "scenario", "n", "median", "p99", "p999", "max", "cov", "tail999", "e2e_median", "gflops"]].round(3).to_string())

# run-to-run robustness
if p3.rep.nunique() > 1:
    R = summarize(p3.assign(platform=p3.platform + "_r" + p3.rep.astype(str)))
    R["base"] = R.platform.str.replace(r"_r\d", "", regex=True)
    rob = R.groupby(["label", "kernel", "size", "scenario"])["median"].agg(lambda x: x.std() / x.mean()).reset_index()
    rob.to_csv(os.path.join(D, "p3_rep_robustness.csv"), index=False)
    print("\nP3 run-to-run CoV of medians: mean %.3f, max %.3f (%d reps)" % (rob["median"].mean(), rob["median"].max(), p3.rep.nunique()))

# H3: interference ratios intf/iso, P3 vs P2
def ratios(S):
    k = ["label", "kernel", "size"]
    a = S[S.scenario == "iso"].set_index(k); b = S[S.scenario == "intf"].set_index(k)
    j = a[["median", "p999", "e2e_median"]].join(b[["median", "p999", "e2e_median"]], lsuffix="_iso", rsuffix="_intf", how="inner")
    j["r_med"] = j.median_intf / j.median_iso; j["r_p999"] = j.p999_intf / j.p999_iso
    j["r_e2e"] = j.e2e_median_intf / j.e2e_median_iso
    return j[["median_iso", "r_med", "r_p999", "r_e2e"]].reset_index()
print("\n=== H3 interference ratios (intf/iso), P3"); r3 = ratios(S3); print(r3.round(3).to_string())
print("\n=== H3 interference ratios (intf/iso), P2 (for comparison)"); print(ratios(S2).round(3).to_string())
r3.to_csv(os.path.join(D, "p3_interference.csv"), index=False)

# speed-up of the dGPU over serial and over the P2 iGPU (medians, iso)
iso3 = S3[S3.scenario == "iso"].set_index(["label", "kernel", "size"])["median"]
iso2 = S2[S2.scenario == "iso"].set_index(["label", "kernel", "size"])["median"]
print("\n=== P3 iso medians (us) and ratios")
for (k, n) in sorted({(k, n) for _, k, n in iso3.index if k != "null"}):
    g = iso3.get(("OpenCL NVIDIA dGPU", k, n)); s = iso3.get(("C++ serial", k, n)); o = iso3.get(("C++ OpenMP", k, n))
    ig = iso2.get(("OpenCL AMD iGPU", k, n))
    print(f"{k:7s} {n:6d}  dGPU={g:10.1f}  serial={s:10.1f}  omp={o if o else float('nan'):10.1f}  serial/dGPU={s/g:7.1f}  P2iGPU/P3dGPU={ig/g if ig else float('nan'):6.1f}")

# H2: overhead
ov = pd.read_csv(os.path.join(D, "p3_overhead.csv.meta.csv"))
ov["cfg"] = ov.model + "/" + ov.impl + "/" + ov.kernel + "/" + ov.scenario
agg = ov.groupby("cfg")[["t_init_ns", "t_build_ns", "t_first_ns"]].median() / 1e6
print("\n=== H2 overhead (ms, median of 30 fresh processes)"); print(agg.round(3).to_string())
agg.to_csv(os.path.join(D, "p3_overhead_summary.csv"))

# EVT on every steady-state series of rep 1 (same protocol as P1/P2)
rows = []
for key, x in p3[(p3.rep == 1) & (p3.scenario.isin(["iso", "intf"]))].groupby(["model", "impl", "device", "kernel", "size", "scenario"]):
    v = x.lat_ns.values / 1e3
    if len(v) < 200: continue
    r = pwcet(v); med = np.median(v)
    rows.append(dict(zip(["model", "impl", "device", "kernel", "size", "scenario"], key), median=med, mowcet=r["mowcet"],
                     pw6=r["pwcet_1e-06"], xi=r["xi"], lb=r["lb_p"], ks=r["ks_p"], gof=r["gof_p"],
                     valid=(r["lb_p"] > .05) and (r["ks_p"] > .05) and (r["gof_p"] > .05)))
E = pd.DataFrame(rows); E.to_csv(os.path.join(D, "p3_evt.csv"), index=False)
n = len(E)
print(f"\n=== EVT on P3: {n} series | independence rejected {(E.lb<=.05).mean():.0%} | i.d. rejected {(E.ks<=.05).mean():.0%} "
      f"| GoF rejected {(E.gof<=.05).mean():.0%} | xi>0 {(E.xi>0).mean():.0%} | valid {E.valid.sum()}/{n}")
