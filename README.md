# Evaluation of High-Level Programming Models for Critical Heterogeneous Avionics

Artifact of the paper submitted to the **2027 IEEE Aerospace Conference** (Big Sky, MT, USA):

> M. K. de M. Moreira and D. S. Loubach, "Evaluation of High-Level Programming Models for Critical Heterogeneous Avionics," in *Proc. IEEE Aerospace Conference*, 2027. *(under review)*

This repository contains everything needed to reproduce the paper's results: the benchmark suite (native C++/OpenMP, OpenCL 1.2, and SYCL 2020), the measurement campaigns, the raw per-iteration latency data, the analysis scripts (statistics, hypothesis tests, EVT/pWCET), the figures (PNG, PDF and editable TikZ/pgfplots), and the classified list of the studies selected by the systematic mapping.

Version 1.1 adds a supplementary platform, **P3** (desktop CPU + discrete NVIDIA GPU), with a second SYCL implementation (Intel oneAPI DPC++) and a second co-runner placement. **P3 results are not part of the paper version under review**; they are provided as additional evidence and are described in the section "Supplementary platform P3" below.

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
  run_p3_*.ps1/.bat, build_p3_sycl.bat,   Windows desktop + discrete GPU platform (P3, supplementary)
    p3_fp_control.bat, machine_inventory.ps1
  analysis.py        summary statistics, run-to-run robustness, EVT/pWCET
  h1_bootstrap.py    95% moving-block bootstrap CIs for the H1 ratios (Table 2 of the paper)
  h3_eta2.py         eta squared of configuration, kernel and size for H3 (Section 5)
  warmup_check.py    deviation of the first 50 samples from the series median (warm-up check, Section 5)
  p3_analysis.py, p3_sycl_analysis.py, p3_l3_analysis.py   analysis of the P3 campaigns
  make_figs.py, figs.py, fig_msl.py      publication figures (PNG)
data/       raw and processed measurements (see "Data dictionary")
figures/    png/, pdf/ (as in the paper) and tikz/ (editable pgfplots sources, .dat files,
            and export_tikz_data.py, which regenerates the .dat files from data/)
msl/        msl_included.csv: 24 primary studies with their MQ1–MQ5 classification
```

## Platforms used in the paper

| | P1 | P2 |
|---|---|---|
| Hardware | Intel Xeon (Cascade Lake), 2 vCPU, KVM guest | AMD Ryzen 5 7520U (4C/8T) + Radeon 610M iGPU (gfx1036) |
| OS | Ubuntu 24.04, Linux 6.18 | Windows 11 |
| Toolchains | GCC 13.3, PoCL 5.0, Intel OpenCL CPU runtime 2026.1, AdaptiveCpp 25.02 (LLVM 18) | MinGW-w64 GCC 13.2, AMD APP OpenCL 3652.0 |

## Supplementary platform P3

| | P3 |
|---|---|
| Hardware | AMD Ryzen 9 5900X (12C/24T, two CCDs with separate L3) + NVIDIA GeForce RTX 4060 Ti 8 GB |
| OS | Windows 11 Pro (build 26200), High-performance power plan |
| Toolchains | MinGW-w64 GCC 13.2 (cross-compiled, static), NVIDIA OpenCL (driver 616.92), Intel oneAPI DPC++ 2025.1.1 + Intel OpenCL CPU runtime 2025.19 |

Three campaigns, each with three repetitions (details in `data/sysinfo_p3.txt`):
1. `run_p3_campaign.ps1`: C++ serial/OpenMP and OpenCL on the discrete GPU; start-up overhead with the NVIDIA kernel cache enabled (`ovh`) and disabled (`ovh_cold`, `CUDA_CACHE_DISABLE=1`).
2. `run_p3_sycl_campaign.bat`: H1 with a second SYCL implementation. SYCL (DPC++, JIT and AOT, USM and buffers) versus OpenCL on the **same** Intel OpenCL CPU runtime and device. Start-up with the SYCL persistent cache off (`ovh_cold`) and on (`ovh_warm`). `p3_fp_control.bat` repeats the comparison with `/fp:precise` as a control for the compiler's default floating-point model.
3. `run_p3_l3_campaign.bat`: the co-runner on the same CCD as the critical thread (`intf_l3`, shared L3), versus the other CCD (`intf`, campaigns 1 and 2).

SYCL on the NVIDIA GPU was not evaluated: neither AdaptiveCpp nor DPC++ offers a validated CUDA back-end on Windows.

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

Windows (P3): the C++/OpenCL binaries were cross-compiled on Linux with MinGW-w64 (an import library for `OpenCL.dll` is generated with `dlltool` from the list of used entry points):
```bash
x86_64-w64-mingw32-g++-posix -O2 -std=c++17 -static -fopenmp src/bench_cpp.cpp -o bench_cpp.exe
x86_64-w64-mingw32-g++-posix -O2 -std=c++17 -static -I. -DCL_TARGET_OPENCL_VERSION=120 src/bench_ocl.cpp -L. -lOpenCL -o bench_ocl.exe
x86_64-w64-mingw32-g++-posix -O2 -std=c++17 -static src/enemy.cpp -o enemy.exe
```
The SYCL binaries are built on the target with `scripts\build_p3_sycl.bat` (Visual Studio Build Tools 2022 + oneAPI DPC++; run it from `build\`).

## Running

```
<binary> -k gemm|conv|kalman|null -n SIZE -i ITERS -w WARMUP -p PLATFORM -s iso|intf -o OUT.csv -d DEVICE
```
- `bench_cpp`: `-d serial` or `-d omp`
- `bench_ocl`: `-d "<platform substring>:cpu|gpu"`, e.g. `Intel:cpu`, `Portable:cpu`, `AMD:gpu`, `NVIDIA:gpu`
- `bench_sycl_*`: `-d usm` or `-d buf`; select the back-end with `ACPP_VISIBILITY_MASK=omp|ocl` (AdaptiveCpp) or `ONEAPI_DEVICE_SELECTOR=opencl:cpu` (DPC++)

Every run validates its output against a serial reference and appends one row per iteration to `OUT.csv` and one row per process to `OUT.csv.meta.csv`. The campaign scripts run inside `build/` and write their CSV files there; move them to `data/` before running the analysis. They pin the critical partition to one CPU and start the co-runner on the other cores for the `intf` scenario.

## Reproducing the analysis and figures

```bash
pip install -r requirements.txt
python scripts/analysis.py      # data/summary.csv + robustness
python scripts/make_figs.py     # figures/png/fig1..fig6
python scripts/fig_msl.py       # figures/png/fig7_msl.png (Figure 1 in the paper)
python scripts/h1_bootstrap.py  # data/h1_bootstrap_ci.csv (Table 2)
python scripts/h3_eta2.py       # data/h3_eta2.csv (H3: eta squared per factor)
python scripts/warmup_check.py  # data/warmup_check.csv (warm-up check)
python scripts/p3_analysis.py; python scripts/p3_sycl_analysis.py; python scripts/p3_l3_analysis.py   # P3
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
- `summary.csv`, `evt.csv`, `overhead_summary.csv`, `p2_rep_robustness.csv`, `h1_bootstrap_ci.csv`, `h3_eta2.csv`, `warmup_check.csv`: processed results used in the paper.
- P3 (supplementary), same column layout:
  - `p3_rep{1,2,3}.csv`, `p3_overhead.csv`: campaign 1 (`impl` = `serial`, `openmp`, `nvidia-ocl`).
  - `p3s_rep{1,2,3}.csv`, `p3s_overhead.csv`, `p3s_fpcheck.csv`: campaign 2 (`impl` = `intel-ocl`, `dpcpp-ocl-{jit,aot}-{usm,buf}`, `dpcpp-ocl-jitprecise-usm`).
  - `p3l3_rep{1,2,3}.csv`: campaign 3 (`scenario` = `iso`, `intf_l3`).
  - Processed: `p3_summary.csv`, `p3_interference.csv`, `p3_overhead_summary.csv`, `p3_evt.csv`, `p3_rep_robustness.csv`, `p3s_summary.csv`, `p3s_h1_bootstrap_ci.csv`, `p3s_overhead_summary.csv`, `p3s_evt.csv`, `p3_l3_vs_xccd.csv`.
  - `sysinfo_p3.txt` (machine and pinning) and `logs/` (campaign logs).

## Changelog

- **1.1.1** (2026-09-29): `h3_eta2.py` and `warmup_check.py` added, with their outputs, so that every number in Section 5 is reproducible from a script.
- **1.1.0** (2026-09-26)
  - Repository layout moved to the root.
  - Stricter input validation in the harness. It only affects invalid command lines; all campaigns use valid ones.
  - `h1_bootstrap.py` and `h1_bootstrap_ci.csv` added.
  - PDF figures added and TikZ figures updated.
  - Supplementary platform P3 added: scripts, raw data and analysis.
  - `bench_ocl` labels the NVIDIA platform as `nvidia-ocl`, and `bench_sycl` labels DPC++ builds as `dpcpp-*`.
- **1.0.0**: artifact of the submitted paper (P1, P2).

## License
[![DOI](https://zenodo.org/badge/1385724567.svg)](https://doi.org/10.5281/zenodo.22942913)
- Code (`src/`, `scripts/`): MIT License (`LICENSE`).
- Data, figures, and the mapping list (`data/`, `figures/`, `msl/`): Creative Commons Attribution 4.0 (`LICENSE-DATA`).

## Acknowledgements

Laboratory infrastructure provided by the Computer Science Division (IEC) of the Aeronautics Institute of Technology (ITA). This work was supported by FAPESP and developed within the Engineering Research Center for Air Mobility of the Future (Flymov), a partnership among ITA, Embraer, and FAPESP.

## How to cite

See `CITATION.cff` (GitHub shows a "Cite this repository" button).
