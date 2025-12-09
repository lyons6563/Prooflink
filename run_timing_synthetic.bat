@echo off
REM === ProofLink - Contribution Timing Analyzer (Synthetic Defaults) ===

REM Set project root
set "PROJECT_ROOT=C:\Users\Sam Lyons\Documents\dev"

REM Call the Python inside your virtual environment directly (no activation needed)
"%PROJECT_ROOT%\src\.venv\Scripts\python.exe" ^
  "%PROJECT_ROOT%\src\contribution_timing_analyzer_v2.py"

echo.
echo =====================================================================
echo   Timing analyzer completed using synthetic ADP/Empower sample files.
echo   Output: %PROJECT_ROOT%\data\processed\late_contributions.csv
echo =====================================================================
echo.
pause
