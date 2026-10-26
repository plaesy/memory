# Reload: Build debug binary and install to local Programs\memory

$ErrorActionPreference = "Stop"

# Get LOCALAPPDATA
$localAppData = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::LocalApplicationData)
$destDir = Join-Path $localAppData 'Programs\memory'
$destBinary = Join-Path $destDir 'memory.exe'
$sourceBinary = 'target\debug\memory.exe'

# Verify debug binary exists
if (-not (Test-Path $sourceBinary)) {
    Write-Error "Binary not found: $sourceBinary"
    exit 1
}

# Create destination directory if needed
New-Item -ItemType Directory -Force $destDir | Out-Null

# Copy binary
Copy-Item -Force $sourceBinary $destBinary
Write-Host "Binary updated: $destBinary"
Write-Host "Restart Claude Code and Kilo sessions to pick up changes"
