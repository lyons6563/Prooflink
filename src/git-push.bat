@echo off
REM -------------------------------------------
REM  ProofLink One-Click Git Push Script
REM  Executes: add → commit → push
REM -------------------------------------------

cd /d "C:\Users\Sam Lyons\Documents\dev\src"

echo ======================================================
echo   ProofLink Git Push - One Click Deployment
echo ======================================================
echo.

REM Detect current branch
for /f "tokens=*" %%i in ('git branch --show-current') do set BRANCH=%%i

echo Current branch: %BRANCH%
echo.

REM Stage all changes
git add .

REM Auto-generate commit message with timestamp
set datetime=%date% %time%
git commit -m "Auto-commit: %datetime%"

REM Push
git push origin %BRANCH%

echo.
echo ======================================================
echo   Push complete. Review above logs.
echo ======================================================
pause

