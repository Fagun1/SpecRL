@echo off
rem Starts the live demo dashboard. Edit the two model paths if your files are elsewhere.
cd /d "%~dp0"
set TARGET=%USERPROFILE%\Downloads\Qwen3-4B-Q4_K_M.gguf
set DRAFT=%USERPROFILE%\Downloads\Qwen3-0.6B-Q8_0.gguf
python demo_app.py --target "%TARGET%" --draft "%DRAFT%" --open
pause
