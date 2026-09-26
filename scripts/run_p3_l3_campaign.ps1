# run_p3_l3_campaign.ps1 - H3 on P3 with the co-runner on the SAME CCD as the critical thread (shared L3),
# mirroring the P2 geometry. Critical thread: logical CPU 0 (core 0, CCD0).
# Co-runner: 4 threads on logical CPUs 2,4,6,8 (cores 1-4, CCD0; SMT siblings of core 0 left idle).
# Scenarios: iso (baseline re-measured in the same session) and intf_l3.
param([int]$reps = 3)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path results_p3_l3 | Out-Null
$log = "results_p3_l3\campaign.log"
function L($s) { $s | Tee-Object -FilePath $log -Append | Out-Host }
$env:ONEAPI_DEVICE_SELECTOR = "opencl:cpu"
[int64]$enemyMask = 0x154
L "=== P3-L3 start $(Get-Date -Format o) | enemy mask=0x$('{0:X}' -f $enemyMask) (CCD0, shared L3)"
$plan = @(
  @{k="gemm";n=256;i=1000}, @{k="gemm";n=512;i=300},
  @{k="conv";n=512;i=1000}, @{k="conv";n=1024;i=500},
  @{k="kalman";n=4096;i=1000}, @{k="kalman";n=32768;i=500})
$cfgs = @(
  @{exe=".\bench_cpp.exe"; d="serial"},
  @{exe=".\bench_ocl.exe"; d="Intel:cpu"},
  @{exe=".\bench_sycl_dpcpp_jit.exe"; d="usm"},
  @{exe=".\bench_ocl.exe"; d="NVIDIA:gpu"})
function Run-Pinned($exe, $argstr, $mask) { cmd /c "start `"`" /wait /b /affinity $mask $exe $argstr" | Out-Host }
function Start-Enemy {
  $p = Start-Process -FilePath ".\enemy.exe" -ArgumentList "64 100000 4" -WindowStyle Hidden -PassThru
  $p.ProcessorAffinity = [IntPtr]$enemyMask; $p }
for ($r = 1; $r -le $reps; $r++) {
  $out = "results_p3_l3\p3l3_rep$r.csv"
  L "--- repetition $r -> $out  $(Get-Date -Format T)"
  foreach ($sc in @("iso","intf_l3")) {
    foreach ($c in $plan) {
      $en = $null
      if ($sc -ne "iso") { $en = Start-Enemy; Start-Sleep -Seconds 2 }
      foreach ($g in $cfgs) { Run-Pinned $g.exe "-k $($c.k) -n $($c.n) -i $($c.i) -w 20 -p P3 -s $sc -o $out -d $($g.d)" 1 }
      if ($en) { Stop-Process -Id $en.Id -Force }
    }
    $en = $null
    if ($sc -ne "iso") { $en = Start-Enemy; Start-Sleep -Seconds 2 }
    foreach ($g in $cfgs) { if ($g.d -ne "serial") { Run-Pinned $g.exe "-k null -n 1 -i 5000 -w 50 -p P3 -s $sc -o $out -d $($g.d)" 1 } }
    if ($en) { Stop-Process -Id $en.Id -Force }
  }
}
L "=== P3-L3 end $(Get-Date -Format o)"
Compress-Archive -Force -Path results_p3_l3\* -DestinationPath results_p3_l3.zip
L "Results packed in results_p3_l3.zip"
