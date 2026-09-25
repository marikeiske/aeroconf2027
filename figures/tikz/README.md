# TikZ/pgfplots figures: AeroConf 2027 paper

Every figure in the paper, redone in editable TikZ/pgfplots and drawn from the real measurement data.

## How to use in Overleaf
1. Upload the whole `tikz_figures/` folder to the root of the Overleaf project.
2. In the paper preamble:
   ```latex
   \newcommand{\figdata}{tikz_figures/data}
   \input{tikz_figures/aeroconf-figstyle}
   ```
3. Where each figure goes:
   ```latex
   \begin{figure}[t]                 % column-width figure
     \centering
     \input{tikz_figures/fig3_perf_heatmap}
     \caption{Median kernel latency on P1 ...}
     \label{fig:perf}
   \end{figure}

   \begin{figure*}[t]                % full-width figure (figure*)
     \centering
     \input{tikz_figures/fig6_ecdf}
     \caption{Latency distributions on P2 ...}
   \end{figure*}
   ```
4. `preview.tex` compiles all figures at once. Compile it from inside the folder, or set it as the main document in Overleaf to test.

Tip: to speed up Overleaf compilation, you can enable `\usetikzlibrary{external}` + `\tikzexternalize` in the preamble.

## Files
| Figure | File | Width | Data (editable) |
|---|---|---|---|
| 1 MSL (bubble chart) | `fig1_msl.tex` | column | counts in the `\bubble{x}{y}{n}` lines |
| 2 Method | `fig2_method.tex` | full | TikZ text/nodes |
| 3 Normalized heatmap | `fig3_perf_heatmap.tex` | column | `data/fig3_heatmap.dat` |
| 4 Runtime overhead | `fig4_overhead.tex` | column | coordinates in the file (= Table 3) |
| 5 Tail ratio | `fig5_tail.tex` | column | `\dumbbell{row}{iso}{intf}{color}` lines |
| 6 ECDF P2 | `fig6_ecdf.tex` | full | `data/fig6_*.dat` |
| 7 EVT/pWCET | `fig7_evt.tex` | full | `data/fig7_*.dat` |

- Colors: set once in `aeroconf-figstyle.tex` (`cNative`, `cOpenCL`, `cSYCL`). The palette is colorblind-safe.
- Fonts follow the document (`\footnotesize`/`\scriptsize`), which makes them compatible with the IEEE template.
- The `.dat` files were exported by `scripts/export_tikz_data.py` (reproducibility package) from the raw measurements. If the experiments are rerun, just re-export them.
- Requirements: TikZ/pgfplots with `compat=1.18` (standard in Overleaf and TeX Live 2021+).
