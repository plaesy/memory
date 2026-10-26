# Install the Plaesy Memory CLI (`memory`) for the current Windows user. Needs only Windows PowerShell.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1                 # from a source checkout (after cargo build --release)
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -BinaryPath C:\Downloads\memory.exe
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Schedule -ScheduleEvery 60   # also register the hourly task
#
# Steps: copy memory.exe to %LOCALAPPDATA%\Programs\memory, add that folder to the user PATH,
# then run `memory onboard` (asks for the notes folder, downloads the model, indexes).
# Re-running is safe: existing files and settings are kept.

[CmdletBinding()]
param(
    [string]$BinaryPath,
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Programs\memory'),
    [switch]$SkipOnboard,
    [switch]$NoPathUpdate,
    # Register a Windows scheduled task that runs `memory schedule once` every -ScheduleEvery minutes.
    [switch]$Schedule,
    [ValidateRange(1, 59940)]
    [int]$ScheduleEvery = 60
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
if (-not $BinaryPath) {
    $BinaryPath = Join-Path $repo 'target\release\memory.exe'
}
if (-not (Test-Path -LiteralPath $BinaryPath -PathType Leaf)) {
    throw "memory.exe not found at '$BinaryPath'. Build it first (cargo build --release) or pass -BinaryPath."
}

# 1. Binary
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$exe = Join-Path $InstallDir 'memory.exe'
Copy-Item -LiteralPath $BinaryPath -Destination $exe -Force
Write-Host "Installed memory.exe to $InstallDir"

# 2. PATH (user scope, so no admin rights are needed)
if (-not $NoPathUpdate) {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $entries = @($userPath -split ';' | Where-Object { $_ })
    if ($entries -notcontains $InstallDir) {
        [Environment]::SetEnvironmentVariable('Path', (($entries + $InstallDir) -join ';'), 'User')
        Write-Host "Added $InstallDir to your user PATH (open a new terminal to use it)."
    } else {
        Write-Host "$InstallDir is already on your user PATH."
    }
}

# 3. Scheduled task (optional). Fixed name, so re-running replaces it instead of adding another.
#    Runs as the current user and only while that user is logged on: no admin rights needed.
if ($Schedule) {
    $taskName = 'PlaesyMemorySchedule'
    $tr = "`"$exe`" schedule once"
    if ($ScheduleEvery -le 999) {
        $timing = @('/SC', 'MINUTE', '/MO', "$ScheduleEvery")
    } else {
        $timing = @('/SC', 'HOURLY', '/MO', "$([math]::Ceiling($ScheduleEvery / 60))")
    }
    & schtasks.exe /Create /F /TN $taskName /TR $tr @timing | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "schtasks could not register '$taskName' (exit code $LASTEXITCODE)."
    }
    Write-Host "Registered scheduled task ${taskName}: memory schedule once every $ScheduleEvery minute(s)."
}

# 4. Setup: interactive, so the user answers the questions in this window
if ($SkipOnboard) {
    Write-Host 'Skipped setup (-SkipOnboard). Run: memory onboard'
} else {
    & $exe onboard
    if ($LASTEXITCODE -ne 0) {
        throw "memory onboard failed (exit code $LASTEXITCODE). Run 'memory onboard' again after fixing the problem."
    }
}
