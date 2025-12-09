@echo off
REM === ProofLink UI Launcher (Windows) ===

REM %~dp0 resolves to the directory of this .bat file, with trailing backslash.
set "PROJECT_ROOT=%~dp0"

REM Normalize to remove any quotes and change to that directory.
cd /d "%PROJECT_ROOT%"

REM Move into src where streamlit_app.py lives.
cd src

REM Run Streamlit using the Python inside your venv.
REM We call `python -m streamlit` so we don't depend on streamlit.exe being on PATH.
"%PROJECT_ROOT%src\.venv\Scripts\python.exe" -m streamlit run streamlit_app.py

REM Keep window open after exit so you can see errors if something blows up.
echo.
echo =====================================================================
echo  Streamlit process exited. If there was an error, review the log above.
echo =====================================================================
echo.
pause
