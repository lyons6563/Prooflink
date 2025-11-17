# backup_to_github.ps1
param(
    [string]$RepoPath = "C:\Users\Sam Lyons\Documents\dev"
)

Set-Location $RepoPath

# Optional: activate venv if you want, not required for git
# & ".\.venv\Scripts\Activate.ps1"

# Ensure .git exists
if (-not (Test-Path ".git")) {
    Write-Host "This is not a git repository. Aborting." -ForegroundColor Red
    exit 1
}

# Check for any changes (staged or unstaged)
$changes = git status --porcelain

if ([string]::IsNullOrWhiteSpace($changes)) {
    Write-Host "No changes to back up. Skipping commit."
    exit 0
}

Write-Host "Changes detected, creating backup commit..."

# Stage everything allowed by .gitignore
git add -A

# Timestamped commit message
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$commitMessage = "Auto backup - $timestamp"

git commit -m $commitMessage

# Push to origin/main (adjust branch name if needed)
git push origin main

Write-Host "Backup complete and pushed to GitHub."
