@echo off
REM === Set paths ===
set PYTHON_ENV=C:\Users\pkoenig2.ORE-PKOENIG01\anaconda3\envs\pyimagej
set SCRIPT_PATH=C:\Users\pkoenig2\SpudInstance\SpudMaster\processor_dataframe_script.py
set LOG_DIR=C:\Users\pkoenig2\SpudInstance\SpudMaster

REM === Launch the script ===
start "" /b "%PYTHON_ENV%\python.exe" "%SCRIPT_PATH%" > "%LOG_DIR%\output.log" 2> "%LOG_DIR%\error.log"
