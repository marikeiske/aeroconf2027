# machine_inventory.ps1 -- READ-ONLY machine inventory (installs and changes nothing)
# Usage: powershell -ExecutionPolicy Bypass -File .\machine_inventory.ps1
$out = Join-Path $PSScriptRoot "machine_inventory.txt"
function W($s) { $s | Tee-Object -FilePath $out -Append }
if (Test-Path $out) { Remove-Item $out }

W "=== System ==="
$os = Get-CimInstance Win32_OperatingSystem
W ("{0} build {1}  ({2})" -f $os.Caption, $os.BuildNumber, $os.OSArchitecture)
W ("Total RAM: {0:N1} GB" -f ($os.TotalVisibleMemorySize/1MB))

W "`n=== CPU ==="
Get-CimInstance Win32_Processor | ForEach-Object {
  W ("{0} | {1} cores / {2} threads | {3} MHz" -f $_.Name.Trim(), $_.NumberOfCores, $_.NumberOfLogicalProcessors, $_.MaxClockSpeed) }

W "`n=== GPUs ==="
Get-CimInstance Win32_VideoController | ForEach-Object { W ("{0} | driver {1}" -f $_.Name, $_.DriverVersion) }
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) { W (& nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv 2>&1 | Out-String) }
else { W "nvidia-smi: not found in PATH" }

W "`n=== Power plan ==="
W ((powercfg /getactivescheme) | Out-String)

W "`n=== Registered OpenCL ICDs ==="
foreach ($k in "HKLM:\SOFTWARE\Khronos\OpenCL\Vendors","HKLM:\SOFTWARE\WOW6432Node\Khronos\OpenCL\Vendors") {
  if (Test-Path $k) { (Get-Item $k).Property | ForEach-Object { W "  $_" } } }
W ("OpenCL.dll in System32: {0}" -f (Test-Path "$env:WINDIR\System32\OpenCL.dll"))

W "`n=== Tools in PATH ==="
foreach ($t in "g++","gcc","clang++","cl","nvcc","cmake","python","py","git","acpp","icpx") {
  $c = Get-Command $t -ErrorAction SilentlyContinue
  W ("{0,-8} {1}" -f $t, $(if ($c) { $c.Source } else { "-" })) }
if (Get-Command g++ -ErrorAction SilentlyContinue) { W ((g++ --version | Select-Object -First 1) | Out-String) }
if (Get-Command nvcc -ErrorAction SilentlyContinue) { W ((nvcc --version | Select-Object -Last 2) | Out-String) }
if ($env:CUDA_PATH) { W "CUDA_PATH = $env:CUDA_PATH" }

W "`n=== Virtualization (affects timing) ==="
W ("Hyper-V/VBS active: {0}" -f ((Get-CimInstance -Namespace root\Microsoft\Windows\DeviceGuard -ClassName Win32_DeviceGuard -ErrorAction SilentlyContinue).VirtualizationBasedSecurityStatus))
W "`nDone. Output written to: $out"
