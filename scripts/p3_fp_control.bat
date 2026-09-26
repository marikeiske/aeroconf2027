@echo off
rem p3_fp_control.bat - control experiment: is the SYCL speed-up caused by icx's default fast floating-point model?
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 >nul
cd /d "%~dp0"
icx -fsycl /O2 /EHsc /std:c++17 /fp:precise -D_CRT_SECURE_NO_WARNINGS -DACPP_FLAVOR=\"jitprecise\" /I ..\src ..\src\bench_sycl.cpp -o bench_sycl_dpcpp_jitprecise.exe 2>nul
set OCL_ICD_FILENAMES=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\intelocl64.dll
set ONEAPI_DEVICE_SELECTOR=opencl:cpu
set O=results_p3_sycl\p3s_fpcheck.csv
del %O% %O%.meta.csv 2>nul
for %%K in ("gemm 256 1000" "gemm 512 300" "conv 512 1000" "conv 1024 500" "kalman 4096 1000" "kalman 32768 500") do (
  for /f "tokens=1-3" %%a in (%%K) do (
    start "" /wait /b /affinity 1 bench_ocl.exe -k %%a -n %%b -i %%c -w 20 -p P3 -s iso -o %O% -d Intel:cpu
    start "" /wait /b /affinity 1 bench_sycl_dpcpp_jit.exe -k %%a -n %%b -i %%c -w 20 -p P3 -s iso -o %O% -d usm
    start "" /wait /b /affinity 1 bench_sycl_dpcpp_jitprecise.exe -k %%a -n %%b -i %%c -w 20 -p P3 -s iso -o %O% -d usm
  )
)
type %O%.meta.csv
