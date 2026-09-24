param([string]$out = "p2_results.csv")
# run_p2_campaign.ps1 - measurement campaign on platform P2 (AMD Ryzen 5 7520U + Radeon 610M iGPU)
# Critical-partition host thread pinned to logical CPU 0 (mask 0x1).
# Interference: enemy.exe with 4 threads pinned to cores 2-3 (logical 4-7, mask 0xF0),
# streaming a 64 MiB buffer per thread -> contention on the shared DRAM controller.
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..\build")
Remove-Item $out, "$out.meta.csv" -ErrorAction SilentlyContinue
$plan = @(
  @{k="gemm";n=256;i=1000}, @{k="gemm";n=512;i=300},
  @{k="conv";n=512;i=1000}, @{k="conv";n=1024;i=500},
  @{k="kalman";n=4096;i=1000}, @{k="kalman";n=32768;i=500})

function Run-Pinned($exe, $argstr, $mask) {
  cmd /c "start `"`" /wait /b /affinity $mask $exe $argstr" | Out-Host
}
function Start-Enemy { Start-Process -FilePath ".\enemy.exe" -ArgumentList "64 100000 4" -WindowStyle Hidden -PassThru | ForEach-Object { $_.ProcessorAffinity = 0xF0; $_ } }

"=== P2 campaign start $(Get-Date -Format o)" | Tee-Object -FilePath campaign.log
foreach ($sc in @("iso","intf")) {
  foreach ($c in $plan) {
    $en = $null
    if ($sc -eq "intf") { $en = Start-Enemy; Start-Sleep -Seconds 2 }
    $a = "-k $($c.k) -n $($c.n) -i $($c.i) -w 20 -p P2 -s $sc -o $out"
    Run-Pinned ".\bench_cpp.exe" "$a -d serial" 1
    Run-Pinned ".\bench_ocl.exe" "$a -d AMD:gpu" 1
    if ($sc -eq "iso") { & .\bench_cpp.exe -k $c.k -n $c.n -i $c.i -w 20 -p P2 -s all -o $out -d omp | Out-Host }
    if ($en) { Stop-Process -Id $en.Id -Force }
  }
  $en = $null
  if ($sc -eq "intf") { $en = Start-Enemy; Start-Sleep -Seconds 2 }
  Run-Pinned ".\bench_ocl.exe" "-k null -n 1 -i 5000 -w 50 -p P2 -s $sc -o $out -d AMD:gpu" 1
  if ($en) { Stop-Process -Id $en.Id -Force }
}
"=== P2 campaign end $(Get-Date -Format o)" | Tee-Object -FilePath campaign.log -Append
