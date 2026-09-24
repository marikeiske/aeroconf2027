# Evaluation of High-Level Programming Models for Critical Heterogeneous Avionics

Artifact of the paper submitted to the **2027 IEEE Aerospace Conference** (Big Sky, MT, USA):

> M. K. de M. Moreira and D. S. Loubach, "Evaluation of High-Level Programming Models for Critical Heterogeneous Avionics," in *Proc. IEEE Aerospace Conference*, 2027. *(under review)*

This repository contains everything needed to reproduce the paper's results: the benchmark suite (native C++/OpenMP, OpenCL 1.2, and SYCL 2020), the measurement campaigns, the raw per-iteration latency data, the analysis scripts (statistics, hypothesis tests, EVT/pWCET), the figures (PNG and editable TikZ/pgfplots), and the classified list of the studies selected by the systematic mapping.

## Repository layout

```
src/        benchmark suite
  common.hpp        shared harness: timing, deterministic data, reference kernels, CSV output
  kernels.cl        OpenCL C kernels (GEMM, Conv2D 5x5, batched two-sensor Kalman filter, null kernel)
  bench_cpp.cpp     native baseline (serial / OpenMP)
  bench_ocl.cpp     OpenCL host program (C API)
  bench_sycl.cpp    SYCL 2020 program (USM or buffers, selected at run time)
  enemy.cpp         memory-bandwidth co-runner (AMC 20-193 interference channel)
scripts/    campaigns and analysis
  run_p1_campaign.sh, overhead_p1.sh      Linux platform (P1)
  run_p2_campaign.ps1, overhead_p2.ps1    Windows APU platform (P2)
  analysis.py        summary statistics, run-to-run robustness, EVT/pWCET
  make_figs.py, figs.py, fig_msl.py      publication figures (PNG)
data/       raw and processed measurements (see "Data dictionary")
figures/    png/ (as in the paper) and tikz/ (editable pgfplots sources, .dat files,
            and export_tikz_data.py, which regenerates the .dat files from data/)
msl/        msl_included.csv: 24 primary studies with their MQ1–MQ5 classification
```

## Platforms used in the paper

| | P1 | P2 |
|---|---|---|
| Hardware | Intel Xeon (Cascade Lake), 2 vCPU, KVM guest | AMD Ryzen 5 7520U (4C/8T) + Radeon 610M iGPU (gfx1036) |
| OS | Ubuntu 24.04, Linux 6.18 | Windows 11 |
| Toolchains | GCC 13.3, PoCL 5.0, Intel OpenCL CPU runtime 2026.1, AdaptiveCpp 25.02 (LLVM 18) | MinGW-w64 GCC 13.2, AMD APP OpenCL 3652.0 |

## Building

Linux (P1):
```bash
mkdir build && cp src/kernels.cl build/ && cd build
g++ -O3 -fopenmp ../src/bench_cpp.cpp -o bench_cpp
g++ -O3 ../src/bench_ocl.cpp -lOpenCL -o bench_ocl
g++ -O2 -pthread ../src/enemy.cpp -o enemy
acpp -O3 --acpp-targets=generic -DACPP_FLAVOR='"jit"' ../src/bench_sycl.cpp -o bench_sycl_generic
acpp -O3 --acpp-targets=omp     -DACPP_FLAVOR='"aot"' ../src/bench_sycl.cpp -o bench_sycl_omp
```

Windows (P2, MinGW-w64; OpenCL headers from KhronosGroup/OpenCL-Headers in `CL/`):
```powershell
mkdir build; copy src\kernels.cl build\; cd build
g++ -O3 -march=znver2 -fopenmp ..\src\bench_cpp.cpp -o bench_cpp.exe
g++ -O3 -I. ..\src\bench_ocl.cpp C:\Windows\System32\OpenCL.dll -o bench_ocl.exe
g++ -O2 -pthread ..\src\enemy.cpp -o enemy.exe
```

## Running

```
<binary> -k gemm|conv|kalman|null -n SIZE -i ITERS -w WARMUP -p PLATFORM -s iso|intf -o OUT.csv -d DEVICE
```
- `bench_cpp`: `-d serial` or `-d omp`
- `bench_ocl`: `-d "<platform substring>:cpu|gpu"`, e.g. `Intel:cpu`, `Portable:cpu`, `AMD:gpu`
- `bench_sycl_*`: `-d usm` or `-d buf`; select the back-end with `ACPP_VISIBILITY_MASK=omp|ocl`

Every run validates its output against a serial reference and appends one row per iteration to `OUT.csv` and one row per process to `OUT.csv.meta.csv`. The campaign scripts run inside `build/` and write their CSV files there; move them to `data/` before running the analysis. They pin the critical partition to one CPU and start the co-runner on the other cores for the `intf` scenario.

## Reproducing the analysis and figures

```bash
pip install -r requirements.txt
python scripts/analysis.py      # data/summary.csv + robustness
python scripts/make_figs.py     # figures/png/fig1..fig6
python scripts/fig_msl.py       # figures/png/fig7_msl.png (Figure 1 in the paper)
```
```bash
python figures/tikz/export_tikz_data.py   # figures/tikz/data/*.dat
```
The TikZ versions are in `figures/tikz/` (see its README for use in Overleaf).

## Data dictionary

- `p1_results.csv`, `p2_results.csv`, `p2_rep2.csv`, `p2_rep3.csv`: one row per measured iteration.
  Columns: `platform, model, impl, device, kernel, size, scenario, iter, lat_ns` (kernel latency, data resident), `e2e_ns` (including host↔device transfers; −1 for the null kernel).
- `*.meta.csv`: one row per process. Columns: `t_init_ns` (platform/context/queue), `t_build_ns` (OpenCL program build), `t_first_ns` (first launch, including SYCL JIT), `max_rel_err` (validation against the serial reference).
- `p1_overhead*.csv`, `p2_overhead*.csv`: start-up study; `scenario` = `cold`/`warm` refers to the persistent on-disk kernel cache.
- `summary.csv`, `evt.csv`, `overhead_summary.csv`, `p2_rep_robustness.csv`: processed results used in the paper.

## License

- Code (`src/`, `scripts/`): MIT License (`LICENSE`).
- Data, figures, and the mapping list (`data/`, `figures/`, `msl/`): Creative Commons Attribution 4.0 (`LICENSE-DATA`).

## Acknowledgements

Laboratory infrastructure provided by the Computer Science Division (IEC) of the Aeronautics Institute of Technology (ITA). This work was supported by FAPESP and developed within the Engineering Research Center for Air Mobility of the Future (Flymov), a partnership among ITA, Embraer, and FAPESP.

## How to cite

See `CITATION.cff` (GitHub shows a "Cite this repository" button).
