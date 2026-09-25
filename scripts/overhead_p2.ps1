# overhead_p2.ps1 - start-up / build / first-launch overhead, 30 fresh processes per config
Set-Location (Join-Path $PSScriptRoot "..\build")
$out = "p2_overhead.csv"
Remove-Item $out, "$out.meta.csv" -ErrorAction SilentlyContinue
for ($r = 0; $r -lt 30; $r++) {
  cmd /c "start `"`" /wait /b /affinity 1 .\bench_ocl.exe -k gemm -n 256 -i 5 -w 0 -p P2 -s ovh -o $out -d AMD:gpu" | Out-Null
  cmd /c "start `"`" /wait /b /affinity 1 .\bench_ocl.exe -k null -n 1 -i 5 -w 0 -p P2 -s ovh -o $out -d AMD:gpu" | Out-Null
  cmd /c "start `"`" /wait /b /affinity 1 .\bench_cpp.exe -k gemm -n 256 -i 5 -w 0 -p P2 -s ovh -o $out -d serial" | Out-Null
}
"overhead done $(Get-Date -Format o)" | Out-File overhead_done.txt
