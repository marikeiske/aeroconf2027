#!/usr/bin/env python3
"""figs.py - publication figures (300 dpi PNG, print, light surface)."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from analysis import COL, INK, INK2, GRID, KNAME, load, summarize, pwcet, D, FIG
from scipy import stats

DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#b52f2e"])
ORDER_P1 = ["C++ serial", "OpenCL PoCL", "OpenCL Intel", "SYCL OMP AOT USM", "SYCL OMP JIT USM", "SYCL OMP JIT buf",
            "SYCL→OCL Intel USM", "SYCL→OCL Intel buf"]
KS = [("gemm", 256), ("gemm", 512), ("conv", 512), ("conv", 1024), ("kalman", 4096), ("kalman", 32768)]
KSLAB = [{"gemm":"GEMM","conv":"Conv","kalman":"KF"}[k] + f"\n{n}" for k, n in KS]

def save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig); print("saved", name)

# ---------------------------------------------------------------- Fig 1: pipeline / method diagram
def fig_method():
    fig, ax = plt.subplots(figsize=(7.16, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")
    def box(x, y, w, h, t, fc="#ffffff", ec=INK2, bold=False, fs=7.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal")
    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=INK2, lw=0.9))
    ax.text(1, 40, "Flight-autonomy pipeline (workloads)", fontsize=9, color=INK2, style="italic")
    box(1, 27, 15, 10, "Sensors\nradar · EO/IR\ncamera", fc="#f6f6f4")
    box(22, 27, 20, 10, "K2 Conv2D 5×5\nperception\npre-processing")
    box(48, 27, 20, 10, "K3 Batched Kalman\ntwo-sensor\ntrack fusion")
    box(74, 27, 24, 10, "K1 GEMM\ndense linear algebra\n(guidance / trajectory)")
    arr(16.5, 32, 21.5, 32); arr(42.5, 32, 47.5, 32); arr(68.5, 32, 73.5, 32)
    ax.text(1, 22, "Experimental factors", fontsize=9, color=INK2, style="italic")
    box(1, 9, 22, 11, "Programming model\nC++ / OpenMP\nOpenCL (PoCL, Intel, AMD)\nSYCL (AdaptiveCpp)", fs=7.0)
    box(25, 9, 23, 11, "Platform\nP1: Xeon 2 vCPU (KVM)\nP2: Ryzen APU + iGPU", fs=7.0)
    box(50.5, 9, 20, 11, "Scenario\niso: isolated\nintf: memory co-runner\n(AMC 20-193\ninterference channel)", fs=7.0)
    box(73, 9, 25, 11, "Metrics\nthroughput · latency\ninit/build/first launch\njitter · tail · pWCET (EVT)", fs=7.0)
    ax.text(50, 2.5, "Each configuration: numerically cross-validated, 20 warm-up + 300–5000 timed samples",
            ha="center", fontsize=8, color=INK2)
    save(fig, "fig1_method.png")

# ---------------------------------------------------------------- Fig 2: normalized median latency heatmap (P1 iso) + P2
def fig_perf(S):
    s = S[(S.platform == "P1") & (S.scenario == "iso") & (S.kernel != "null")]
    M = np.full((len(ORDER_P1), len(KS)), np.nan)
    for j, (k, n) in enumerate(KS):
        base = s[(s.label == "C++ serial") & (s.kernel == k) & (s["size"] == n)]["median"].values[0]
        for i, lab in enumerate(ORDER_P1):
            v = s[(s.label == lab) & (s.kernel == k) & (s["size"] == n)]["median"].values
            if len(v): M[i, j] = v[0] / base
    fig, ax = plt.subplots(figsize=(3.45, 3.3))
    im = ax.imshow(np.log2(M), cmap=DIV, norm=TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; lv = abs(np.log2(v))
            ax.text(j, i, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if lv > 1.9 else INK)
    ax.set_xticks(range(len(KS))); ax.set_xticklabels(KSLAB, fontsize=7.5)
    ax.set_yticks(range(len(ORDER_P1))); ax.set_yticklabels(ORDER_P1, fontsize=8)
    ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb.ax.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"], fontsize=7.5); cb.outline.set_visible(False)
    cb.set_label("median latency / C++ serial", fontsize=8)
    save(fig, "fig2_perf_p1.png")
    return M

# ---------------------------------------------------------------- Fig 3: ECDFs iso vs intf (P2)
def fig_ecdf(R):
    cases = [("gemm", 512, "GEMM 512"), ("kalman", 32768, "Kalman 32768")]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, (k, n, t) in zip(axs, cases):
        for (m, impl, dev, fam, lab) in [("cpp", "serial", "cpu", "Native", "C++ serial (CPU0)"),
                                          ("opencl", "amd-ocl", "gpu", "OpenCL", "OpenCL AMD iGPU")]:
            for sc, ls in [("iso", "-"), ("intf", "--")]:
                v = R[(R.platform == "P2") & (R.rep == 1) & (R.model == m) & (R.impl == impl) & (R.kernel == k) &
                      (R["size"] == n) & (R.scenario == sc)]["lat_ns"].values / 1e3
                v = np.sort(v); y = np.arange(1, len(v) + 1) / len(v)
                ax.plot(v, y, ls=ls, color=COL[fam], lw=1.6, label=f"{lab}, {sc}")
        ax.set_xscale("log"); ax.set_title(t, color=INK); ax.set_xlabel("latency (µs, log)")
    axs[0].set_ylabel("ECDF")
    h, l = axs[0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.32, wspace=0.08); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    save(fig, "fig3_ecdf_p2.png")

# ---------------------------------------------------------------- Fig 4: runtime phase costs
def fig_overhead(meta, S):
    rows = []
    for _, r in meta.iterrows():
        rows.append(r)
    m = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    y = np.arange(len(m))
    ax.scatter(m["t_init_ms"], y, s=36, color=COL["Native"], marker="o", label="init (platform/context/queue)", zorder=3, edgecolor="white", lw=1)
    ax.scatter(m["t_build_ms"], y, s=36, color=COL["OpenCL"], marker="s", label="build (OpenCL online compile)", zorder=3, edgecolor="white", lw=1)
    ax.scatter(m["t_first_ms"], y, s=40, color=COL["SYCL"], marker="D", label="first launch (incl. SYCL JIT)", zorder=3, edgecolor="white", lw=1)
    ax.scatter(m["steady_ms"], y, s=60, color=INK, marker="|", label="steady-state median", zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(m["label"], fontsize=8); ax.set_xscale("log")
    ax.set_xlabel("time (ms, log) — GEMM 256"); ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.25, -0.2), frameon=False, fontsize=7.5, ncol=2)
    save(fig, "fig4_overhead.png")

# ---------------------------------------------------------------- Fig 5: tail inflation dumbbell
def fig_tail(S):
    s = S[(S.kernel != "null")]
    g = s.groupby(["platform", "label", "family", "scenario"])["tail999"].median().unstack("scenario").reset_index()
    g = g.dropna(subset=["iso", "intf"]); g["name"] = g.platform + "  " + g.label
    g = g.sort_values(["platform", "intf"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.45, 3.6))
    for i, r in g.iterrows():
        ax.plot([r.iso, r.intf], [i, i], color=GRID, lw=2.2, zorder=1)
        ax.scatter(r.iso, i, s=34, facecolor="white", edgecolor=COL[r.family], lw=1.6, zorder=3)
        ax.scatter(r.intf, i, s=34, color=COL[r.family], zorder=3)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.name, fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlabel("tail ratio p99.9 / median (log)")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator([1.5, 2, 3, 4, 6, 8])); ax.xaxis.set_major_formatter(FixedFormatter(["1.5", "2", "3", "4", "6", "8"])); ax.xaxis.set_minor_formatter(NullFormatter())
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, label="isolated"),
          Line2D([], [], marker="o", ls="", color=INK2, label="with interference")] + \
         [Line2D([], [], marker="s", ls="", color=COL[f], label=f) for f in ["Native", "OpenCL", "SYCL"]]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=3, frameon=False, fontsize=7.5)
    save(fig, "fig5_tail.png")
    return g

# ---------------------------------------------------------------- Fig 6: EVT exceedance plot
def fig_evt(R, cases):
    fig, axs = plt.subplots(1, len(cases), figsize=(7.0, 2.5), sharey=True)
    out = []
    for ax, (plat, m, impl, k, n, title, fam) in zip(axs, cases):
        for sc, ls in [("iso", "-"), ("intf", "--")]:
            v = R[(R.platform == plat) & (R.rep == 1) & (R.model == m) & (R.impl == impl) & (R.kernel == k) &
                  (R["size"] == n) & (R.scenario == sc)]["lat_ns"].values / 1e3
            if not len(v): continue
            e = pwcet(v); out.append(dict(platform=plat, config=title, scenario=sc, **e))
            vs = np.sort(v); cc = 1 - np.arange(1, len(vs) + 1) / (len(vs) + 1)
            ax.plot(vs, cc, ls="", marker="o", ms=2.2, color=COL[fam], alpha=0.55 if sc == "iso" else 0.9,
                    mfc="white" if sc == "iso" else COL[fam])
            u, xi, sg, z = e["u"], e["xi"], e["sigma"], e["k"] / e["n"]
            xs = np.linspace(u, max(e["pwcet_1e-06"], vs[-1]) * 1.05, 200)
            ax.plot(xs, z * stats.genpareto.sf(xs - u, xi, 0, sg), ls=ls, color=INK, lw=1.2,
                    label=f"{sc}: pWCET(10⁻⁶)={e['pwcet_1e-06'] / 1e3:.1f} ms" + ("" if min(e["lb_p"], e["ks_p"], e["gof_p"]) > 0.05 else " (i.i.d. rejected)"))
        ax.set_yscale("log"); ax.set_xscale("log"); ax.set_ylim(1e-7, 1.2); ax.set_title(title, color=INK, fontsize=9)
        ax.set_xlabel("latency (µs, log)"); ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.27))
    axs[0].set_ylabel("exceedance probability")
    save(fig, "fig6_evt.png")
    return pd.DataFrame(out)
