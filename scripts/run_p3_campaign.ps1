# run_p3_campaign.ps1 - measurement campaign on platform P3 (Windows desktop, discrete NVIDIA GPU)
# Same kernels, sizes, iterations, pinning and co-runner as the P2 campaign (run_p2_campaign.ps1).
# Nothing is installed and no system setting is changed; everything stays in this folder.
param([int]$reps = 3)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:Path = "$PSScriptRoot;$env:Path"
New-Item -ItemType Directory -Force -Path results_p3 | Out-Null
$log = "results_p3\campaign.log"
function L($s) { $s | Tee-Object -FilePath $log -Append | Out-Host }

# ---- 0. machine inventory (read-only)
& "$PSScriptRoot\machine_inventory.ps1" | Out-Null
Move-Item -Force machine_inventory.txt results_p3\ -ErrorAction SilentlyContinue
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  nvidia-smi -q > results_p3\nvidia_smi_before.txt 2>&1 }

# ---- 1. pinning: critical thread on logical CPU 0; co-runner on the last 4 logical CPUs
$N = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
[int64]$enemyMask = ([int64]1 -shl $N) - ([int64]1 -shl ($N - 4))
L "=== P3 start $(Get-Date -Format o) | logical CPUs=$N | enemy mask=0x$('{0:X}' -f $enemyMask) | reps=$reps"

# ---- 2. which OpenCL devices exist here?
$devs = @()
foreach ($h in @("NVIDIA:gpu","Intel:cpu","Intel:gpu","AMD:cpu","AMD:gpu","Portable:cpu")) {
  & .\bench_ocl.exe -k null -n 1 -i 2 -w 0 -p probe -s probe -o results_p3\probe.csv -d $h 2>&1 | Out-Null
  if ($LASTEXITCODE -eq 0) { $devs += $h; L "OpenCL device found: $h" }
}
Remove-Item results_p3\probe.csv* -ErrorAction SilentlyContinue
if ($devs.Count -eq 0) { L "WARNING: no OpenCL device found - only C++ will run" }

$plan = @(
  @{k="gemm";n=256;i=1000}, @{k="gemm";n=512;i=300},
  @{k="conv";n=512;i=1000}, @{k="conv";n=1024;i=500},
  @{k="kalman";n=4096;i=1000}, @{k="kalman";n=32768;i=500})
function Run-Pinned($exe, $argstr, $mask) { cmd /c "start `"`" /wait /b /affinity $mask $exe $argstr" | Out-Host }
function Start-Enemy {
  $p = Start-Process -FilePath ".\enemy.exe" -ArgumentList "64 100000 4" -WindowStyle Hidden -PassThru
  $p.ProcessorAffinity = [IntPtr]$enemyMask; $p }

# ---- 3. steady-state campaign (iso + intf), $reps repetitions
for ($r = 1; $r -le $reps; $r++) {
  $out = "results_p3\p3_rep$r.csv"
  L "--- repetition $r -> $out  $(Get-Date -Format T)"
  foreach ($sc in @("iso","intf")) {
    foreach ($c in $plan) {
      $en = $null
      if ($sc -eq "intf") { $en = Start-Enemy; Start-Sleep -Seconds 2 }
      $a = "-k $($c.k) -n $($c.n) -i $($c.i) -w 20 -p P3 -s $sc -o $out"
      Run-Pinned ".\bench_cpp.exe" "$a -d serial" 1
      foreach ($d in $devs) { Run-Pinned ".\bench_ocl.exe" "$a -d $d" 1 }
      if ($sc -eq "iso") { & .\bench_cpp.exe -k $c.k -n $c.n -i $c.i -w 20 -p P3 -s all -o $out -d omp | Out-Host }
      if ($en) { Stop-Process -Id $en.Id -Force }
    }
    $en = $null
    if ($sc -eq "intf") { $en = Start-Enemy; Start-Sleep -Seconds 2 }
    foreach ($d in $devs) { Run-Pinned ".\bench_ocl.exe" "-k null -n 1 -i 5000 -w 50 -p P3 -s $sc -o $out -d $d" 1 }
    if ($en) { Stop-Process -Id $en.Id -Force }
  }
}

# ---- 4. start-up / build / first-launch overhead: 30 fresh processes per configuration
#      NVIDIA keeps a compiled-kernel cache; "ovh" = normal (warm cache), "ovh_cold" = cache disabled
$ovh = "results_p3\p3_overhead.csv"
L "--- overhead $(Get-Date -Format T)"
for ($r = 0; $r -lt 30; $r++) {
  foreach ($d in $devs) {
    Run-Pinned ".\bench_ocl.exe" "-k gemm -n 256 -i 5 -w 0 -p P3 -s ovh -o $ovh -d $d" 1
    Run-Pinned ".\bench_ocl.exe" "-k null -n 1 -i 5 -w 0 -p P3 -s ovh -o $ovh -d $d" 1
  }
  if ($devs -contains "NVIDIA:gpu") {
    $env:CUDA_CACHE_DISABLE = "1"
    Run-Pinned ".\bench_ocl.exe" "-k gemm -n 256 -i 5 -w 0 -p P3 -s ovh_cold -o $ovh -d NVIDIA:gpu" 1
    Remove-Item Env:CUDA_CACHE_DISABLE
  }
  Run-Pinned ".\bench_cpp.exe" "-k gemm -n 256 -i 5 -w 0 -p P3 -s ovh -o $ovh -d serial" 1
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  nvidia-smi -q > results_p3\nvidia_smi_after.txt 2>&1 }
L "=== P3 end $(Get-Date -Format o)"
Compress-Archive -Force -Path results_p3\* -DestinationPath results_p3.zip
L "Results packed in results_p3.zip"
