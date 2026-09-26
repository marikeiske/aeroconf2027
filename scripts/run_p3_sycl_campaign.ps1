# run_p3_sycl_campaign.ps1 - H1 on P3: SYCL (Intel oneAPI DPC++ 2025.1, OpenCL CPU back-end) vs OpenCL (same
# Intel OpenCL CPU runtime) on the same device (Ryzen 9 5900X). Same protocol as P1/P2/P3:
# host thread + device pinned to logical CPU 0 (mask 0x1), co-runner on the last 4 logical CPUs.
# Must be started from run_p3_sycl_campaign.bat (loads the oneAPI environment).
param([int]$reps = 3)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path results_p3_sycl | Out-Null
$log = "results_p3_sycl\campaign.log"
function L($s) { $s | Tee-Object -FilePath $log -Append | Out-Host }
$env:ONEAPI_DEVICE_SELECTOR = "opencl:cpu"
$N = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
[int64]$enemyMask = ([int64]1 -shl $N) - ([int64]1 -shl ($N - 4))
L "=== P3-SYCL start $(Get-Date -Format o) | OCL_ICD_FILENAMES=$env:OCL_ICD_FILENAMES | enemy mask=0x$('{0:X}' -f $enemyMask)"
& icx --version 2>&1 | Select-Object -First 1 | ForEach-Object { L "compiler: $_" }

$plan = @(
  @{k="gemm";n=256;i=1000}, @{k="gemm";n=512;i=300},
  @{k="conv";n=512;i=1000}, @{k="conv";n=1024;i=500},
  @{k="kalman";n=4096;i=1000}, @{k="kalman";n=32768;i=500})
$cfgs = @(
  @{exe=".\bench_ocl.exe"; d="Intel:cpu"},
  @{exe=".\bench_sycl_dpcpp_jit.exe"; d="usm"},
  @{exe=".\bench_sycl_dpcpp_jit.exe"; d="buf"},
  @{exe=".\bench_sycl_dpcpp_aot.exe"; d="usm"})
function Run-Pinned($exe, $argstr, $mask) { cmd /c "start `"`" /wait /b /affinity $mask $exe $argstr" | Out-Host }
function Start-Enemy {
  $p = Start-Process -FilePath ".\enemy.exe" -ArgumentList "64 100000 4" -WindowStyle Hidden -PassThru
  $p.ProcessorAffinity = [IntPtr]$enemyMask; $p }

for ($r = 1; $r -le $reps; $r++) {
  $out = "results_p3_sycl\p3s_rep$r.csv"
  L "--- repetition $r -> $out  $(Get-Date -Format T)"
  foreach ($sc in @("iso","intf")) {
    foreach ($c in $plan) {
      $en = $null
      if ($sc -eq "intf") { $en = Start-Enemy; Start-Sleep -Seconds 2 }
      foreach ($g in $cfgs) {
        Run-Pinned $g.exe "-k $($c.k) -n $($c.n) -i $($c.i) -w 20 -p P3 -s $sc -o $out -d $($g.d)" 1 }
      if ($en) { Stop-Process -Id $en.Id -Force }
    }
    $en = $null
    if ($sc -eq "intf") { $en = Start-Enemy; Start-Sleep -Seconds 2 }
    foreach ($g in $cfgs) { Run-Pinned $g.exe "-k null -n 1 -i 5000 -w 50 -p P3 -s $sc -o $out -d $($g.d)" 1 }
    if ($en) { Stop-Process -Id $en.Id -Force }
  }
}

# start-up / JIT overhead: 30 fresh processes per configuration
$ovh = "results_p3_sycl\p3s_overhead.csv"
L "--- overhead $(Get-Date -Format T)"
for ($r = 0; $r -lt 30; $r++) {
  Run-Pinned ".\bench_ocl.exe" "-k gemm -n 256 -i 5 -w 0 -p P3 -s ovh -o $ovh -d Intel:cpu" 1
  Remove-Item Env:SYCL_CACHE_PERSISTENT -ErrorAction SilentlyContinue
  Run-Pinned ".\bench_sycl_dpcpp_jit.exe" "-k gemm -n 256 -i 5 -w 0 -p P3 -s ovh_cold -o $ovh -d usm" 1
  $env:SYCL_CACHE_PERSISTENT = "1"
  Run-Pinned ".\bench_sycl_dpcpp_jit.exe" "-k gemm -n 256 -i 5 -w 0 -p P3 -s ovh_warm -o $ovh -d usm" 1
  Remove-Item Env:SYCL_CACHE_PERSISTENT
  Run-Pinned ".\bench_sycl_dpcpp_aot.exe" "-k gemm -n 256 -i 5 -w 0 -p P3 -s ovh -o $ovh -d usm" 1
}
L "=== P3-SYCL end $(Get-Date -Format o)"
Compress-Archive -Force -Path results_p3_sycl\* -DestinationPath results_p3_sycl.zip
L "Results packed in results_p3_sycl.zip"
