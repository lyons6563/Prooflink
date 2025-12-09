@echo off
REM === ProofLink - Contribution Timing Analyzer (Prompted Files) ===

setlocal ENABLEDELAYEDEXPANSION

REM Set project root
set "PROJECT_ROOT=C:\Users\Sam Lyons\Documents\dev"
set "RAW_DIR=%PROJECT_ROOT%\data\raw"

echo.
echo =====================================================================
echo   ProofLink Contribution Timing Analyzer - Field Run
echo =====================================================================
echo.
echo   All files should be placed in:
echo     %RAW_DIR%
echo.

REM Ask for payroll file name
set "PAYROLL_FILE="
set /p PAYROLL_FILE=Enter payroll file name in data\raw (e.g. payroll_jan15.csv): 

IF "%PAYROLL_FILE%"=="" (
    echo.
    echo [ERROR] Payroll file name is required.
    echo.
    pause
    exit /b 1
)

REM Ask for RK file name (optional)
set "RK_FILE="
set /p RK_FILE=Enter RK file name in data\raw (leave blank to use default synthetic RK): 

REM Build full paths
set "PAYROLL_PATH=%RAW_DIR%\%PAYROLL_FILE%"
set "RK_PATH=%RAW_DIR%\%RK_FILE%"

REM Basic existence check for payroll
if not exist "%PAYROLL_PATH%" (
    echo.
    echo [ERROR] Payroll file not found:
    echo         %PAYROLL_PATH%
    echo.
    echo   Make sure the file is saved in:
    echo         %RAW_DIR%
    echo   and that you typed the name correctly.
    echo.
    pause
    exit /b 1
)

echo.
echo ---------------------------------------------------------------------
echo   Running timing analyzer...
echo   Payroll:     %PAYROLL_PATH%
if "%RK_FILE%"=="" (
    echo   Recordkeeper: using DEFAULT synthetic RK
) else (
    echo   Recordkeeper: %RK_PATH%
)
echo ---------------------------------------------------------------------
echo.

REM Decide which Python command to run (1-arg vs 2-arg mode)
if "%RK_FILE%"=="" (
    "%PROJECT_ROOT%\src\.venv\Scripts\python.exe" ^
        "%PROJECT_ROOT%\src\contribution_timing_analyzer_v2.py" ^
        "%PAYROLL_PATH%"
) else (
    if not exist "%RK_PATH%" (
        echo.
        echo [ERROR] Recordkeeper file not found:
        echo         %RK_PATH%
        echo.
        echo   Fix the RK file name or move the file into:
        echo         %RAW_DIR%
        echo.
        pause
        exit /b 1
    )

    "%PROJECT_ROOT%\src\.venv\Scripts\python.exe" ^
        "%PROJECT_ROOT%\src\contribution_timing_analyzer_v2.py" ^
        "%PAYROLL_PATH%" ^
        "%RK_PATH%"
)

echo.
echo =====================================================================
echo   Timing analyzer finished.
echo   Output: %PROJECT_ROOT%\data\processed\late_contributions.csv
echo =====================================================================
echo.
pause

endlocal
