# Build Sample Evidence Pack Demo Script
# Generates a sample Evidence Pack using the src.manifest module

Write-Host "=== ProofLink Evidence Pack Demo ===" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
$venvPath = ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvPath)) {
    Write-Host "Warning: Virtual environment not found at $venvPath" -ForegroundColor Yellow
    Write-Host "Attempting to run without activation..." -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "Activating virtual environment..." -ForegroundColor Gray
    & $venvPath
    Write-Host ""
}

# Set PYTHONPATH to current directory
$env:PYTHONPATH = (Get-Location).Path
Write-Host "Running Evidence Pack generation..." -ForegroundColor Gray
Write-Host ""

# Run the manifest module
python -m src.manifest

Write-Host ""
Write-Host "=== Demo Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Sample Evidence Pack ZIP generated in: tmp_run_outputs\" -ForegroundColor Green
Write-Host ""
Write-Host "The ZIP file is named: evidence_pack_<RUN_ID>.zip" -ForegroundColor Gray
Write-Host "Check tmp_run_outputs\ for the generated file." -ForegroundColor Gray
Write-Host ""

