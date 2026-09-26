@echo off
rem build_p3_sycl.bat - builds the SYCL benchmark with Intel oneAPI DPC++ (JIT and AOT for x86-64 CPU)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 >nul
cd /d "%~dp0"
echo === sycl-ls
sycl-ls
echo === build JIT (SPIR-V, compiled by the OpenCL CPU runtime at run time)
icx -fsycl /O2 /EHsc /std:c++17 /I ..\src ..\src\bench_sycl.cpp -o bench_sycl_dpcpp_jit.exe
echo === build AOT (spir64_x86_64, compiled ahead of time)
icx -fsycl -fsycl-targets=spir64_x86_64 -DACPP_FLAVOR=\"aot\" /O2 /EHsc /std:c++17 /I ..\src ..\src\bench_sycl.cpp -o bench_sycl_dpcpp_aot.exe
echo === compiler
icx --version
dir /b *.exe
