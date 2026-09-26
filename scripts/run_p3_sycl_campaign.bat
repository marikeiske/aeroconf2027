@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 >nul
cd /d "%~dp0"
set OCL_ICD_FILENAMES=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\intelocl64.dll
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_p3_sycl_campaign.ps1"
