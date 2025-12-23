# Find the most recent run directory in streamlit_runs
$streamlitRunsPath = "streamlit_runs"

if (-not (Test-Path $streamlitRunsPath)) {
    Write-Host "Error: streamlit_runs directory does not exist." -ForegroundColor Red
    Write-Host "Please run a reconciliation first to generate outputs." -ForegroundColor Yellow
    exit 1
}

# Get all subdirectories and find the one with the most recent output directory
$runDirs = Get-ChildItem -Path $streamlitRunsPath -Directory

if ($runDirs.Count -eq 0) {
    Write-Host "Error: No run directories found in streamlit_runs." -ForegroundColor Red
    Write-Host "Please run a reconciliation first to generate outputs." -ForegroundColor Yellow
    exit 1
}

# Find the most recent run directory that has an output subdirectory
$latestRunDir = $null
$latestOutputTime = [DateTime]::MinValue

foreach ($dir in $runDirs) {
    $outputDir = Join-Path $dir.FullName "output"
    if (Test-Path $outputDir -PathType Container) {
        $lastWriteTime = (Get-Item $outputDir).LastWriteTime
        if ($lastWriteTime -gt $latestOutputTime) {
            $latestOutputTime = $lastWriteTime
            $latestRunDir = $dir
        }
    }
}

if ($null -eq $latestRunDir) {
    Write-Host "Error: No run directories with output subdirectories found." -ForegroundColor Red
    Write-Host "Please run a reconciliation first to generate outputs." -ForegroundColor Yellow
    exit 1
}

$runId = $latestRunDir.Name
$outputPath = Join-Path $latestRunDir.FullName "output"

Write-Host "Most recent run ID: $runId" -ForegroundColor Green
Write-Host "Output directory: $outputPath" -ForegroundColor Cyan
Write-Host ""

# List files in the output directory
Write-Host "Files in output directory:" -ForegroundColor Yellow
Get-ChildItem -Path $outputPath -File | ForEach-Object {
    Write-Host "  - $($_.Name)" -ForegroundColor White
}

Write-Host ""
Write-Host "Opening output folder in File Explorer..." -ForegroundColor Cyan
Start-Process explorer.exe -ArgumentList $outputPath

