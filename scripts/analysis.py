#!/usr/bin/env python3
"""analysis.py - statistics, hypothesis tests, EVT/pWCET and figures for the AeroConf 2027 paper."""
import os, glob
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures", "png")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- style (print, IEEE, >=10 pt)
COL = {"Native": "#2a78d6", "OpenCL": "#eb6834", "SYCL": "#1baf7a"}   # validated slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})

LABEL = {  # (model, impl, device) -> short label, family
    ("cpp", "serial", "cpu"): ("C++ serial", "Native"),
    ("cpp", "openmp", "cpu"): ("C++ OpenMP", "Native"),
    ("opencl", "pocl", "cpu"): ("OpenCL PoCL", "OpenCL"),
    ("opencl", "intel-ocl", "cpu"): ("OpenCL Intel", "OpenCL"),
    ("opencl", "amd-ocl", "gpu"): ("OpenCL AMD iGPU", "OpenCL"),
    ("sycl", "acpp-omp-jit-usm", "cpu"): ("SYCL OMP JIT USM", "SYCL"),
    ("sycl", "acpp-omp-jit-buf", "cpu"): ("SYCL OMP JIT buf", "SYCL"),
    ("sycl", "acpp-omp-aot-usm", "cpu"): ("SYCL OMP AOT USM", "SYCL"),
    ("sycl", "acpp-ocl-jit-usm", "cpu"): ("SYCL→OCL Intel USM", "SYCL"),
    ("sycl", "acpp-ocl-jit-buf", "cpu"): ("SYCL→OCL Intel buf", "SYCL"),
}
FLOP = {"gemm": lambda n: 2.0 * n ** 3, "conv": lambda n: 50.0 * n * n, "kalman": lambda n: 1000.0 * n}
KNAME = {"gemm": "GEMM", "conv": "Conv2D", "kalman": "Kalman", "null": "Null"}

def load(pattern, rep=None):
    fs = sorted(glob.glob(os.path.join(D, pattern)))
    out = []
    for i, f in enumerate(fs):
        x = pd.read_csv(f, keep_default_na=False, na_values=[""]); x["rep"] = rep if rep is not None else i + 1; out.append(x)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def summarize(df):
    g = df.groupby(["platform", "model", "impl", "device", "kernel", "size", "scenario"])
    def f(x):
        v = x["lat_ns"].values / 1e3  # us
        e = x["e2e_ns"].values / 1e3
        med = np.median(v)
        return pd.Series({"n": len(v), "min": v.min(), "median": med, "mean": v.mean(), "std": v.std(ddof=1),
                          "cov": v.std(ddof=1) / v.mean(), "p99": np.percentile(v, 99), "p999": np.percentile(v, 99.9),
                          "max": v.max(), "iqr": np.subtract(*np.percentile(v, [75, 25])),
                          "e2e_median": np.median(e[e > 0]) if (e > 0).any() else np.nan,
                          "tail999": np.percentile(v, 99.9) / med, "maxmed": v.max() / med})
    s = g.apply(f).reset_index()
    s["label"] = [LABEL.get((m, i, d), (f"{m}/{i}", "Other"))[0] for m, i, d in zip(s.model, s.impl, s.device)]
    s["family"] = [LABEL.get((m, i, d), ("", "Other"))[1] for m, i, d in zip(s.model, s.impl, s.device)]
    s["gflops"] = [FLOP[k](n) / (med * 1e3) if k in FLOP else np.nan for k, n, med in zip(s.kernel, s["size"], s["median"])]
    return s

# ---------------------------------------------------------------- EVT (POT / GPD) pWCET
def pwcet(v, probs=(1e-3, 1e-6, 1e-9), q=0.90):
    """Peaks-over-threshold with a Generalized Pareto fit. Returns dict with tests and quantiles.
    Exceedance probability is per execution (per job)."""
    v = np.asarray(v, float)
    u = np.quantile(v, q)
    exc = v[v > u] - u
    res = {"n": len(v), "u": u, "k": len(exc)}
    # i.i.d. checks (MBPTA practice): lag-1 autocorrelation (Ljung-Box, lag 10) and KS between halves
    x = v - v.mean()
    ac = [np.sum(x[:-l] * x[l:]) / np.sum(x * x) for l in range(1, 11)]
    Q = len(v) * (len(v) + 2) * sum(a * a / (len(v) - l) for l, a in zip(range(1, 11), ac))
    res["lb_p"] = 1 - stats.chi2.cdf(Q, 10)
    res["ks_p"] = stats.ks_2samp(v[: len(v) // 2], v[len(v) // 2:]).pvalue
    c, loc, scale = stats.genpareto.fit(exc, floc=0)
    res["xi"] = c; res["sigma"] = scale
    res["gof_p"] = stats.kstest(exc, "genpareto", args=(c, 0, scale)).pvalue
    zeta = len(exc) / len(v)
    for p in probs:
        res[f"pwcet_{p:.0e}"] = u + stats.genpareto.ppf(1 - p / zeta, c, 0, scale) if p < zeta else np.quantile(v, 1 - p)
    res["mowcet"] = v.max()
    return res

if __name__ == "__main__":
    p1 = load("p1_results.csv", rep=1)
    p2 = pd.concat([load("p2_results.csv", rep=1), load("p2_rep2.csv", rep=2), load("p2_rep3.csv", rep=3)], ignore_index=True)
    allr = pd.concat([p1, p2], ignore_index=True)
    S = summarize(allr[allr.rep == 1])
    S.to_csv(os.path.join(D, "summary.csv"), index=False)
    # repetition robustness on P2: coefficient of variation of the medians across the 3 campaigns
    S2 = summarize(p2.assign(platform=p2.platform + "_r" + p2.rep.astype(str)))
    S2["base"] = S2.platform.str.replace(r"_r\d", "", regex=True)
    rob = S2.groupby(["base", "model", "impl", "device", "kernel", "size", "scenario"])["median"].agg(lambda x: x.std() / x.mean()).reset_index()
    rob.to_csv(os.path.join(D, "p2_rep_robustness.csv"), index=False)
    print("P2 run-to-run CoV of medians: mean %.3f, max %.3f" % (rob["median"].mean(), rob["median"].max()))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(S[["platform", "label", "kernel", "size", "scenario", "n", "median", "p99", "p999", "max", "cov", "tail999", "e2e_median", "gflops"]].round(3).to_string())
