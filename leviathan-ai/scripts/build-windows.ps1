# Leviathan AI - Windows Build Script
# Builds native Windows executable with installer

param(
    [switch]$Release,
    [switch]$Installer,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           LEVIATHAN AI - Windows Build System              ║" -ForegroundColor Cyan
Write-Host "║        Enterprise AI with Full DAG Auditability            ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Clean if requested
if ($Clean) {
    Write-Host "[CLEAN] Removing build artifacts..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "$ProjectRoot\target" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$ProjectRoot\dist" -ErrorAction SilentlyContinue
}

# Create output directories
$DistDir = "$ProjectRoot\dist\windows"
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

# Determine build profile
$Profile = if ($Release) { "release" } else { "debug" }
$CargoFlags = if ($Release) { "--release" } else { "" }

Write-Host "[BUILD] Building Leviathan CLI ($Profile)..." -ForegroundColor Green

# Build the CLI
Push-Location $ProjectRoot
try {
    cargo build -p leviathan-cli $CargoFlags
    if ($LASTEXITCODE -ne 0) {
        throw "Cargo build failed"
    }
} finally {
    Pop-Location
}

# Copy binary
$BinaryName = "leviathan.exe"
$SourceBinary = "$ProjectRoot\target\$Profile\leviathan-cli.exe"
$DestBinary = "$DistDir\$BinaryName"

if (Test-Path $SourceBinary) {
    Copy-Item $SourceBinary $DestBinary -Force
    Write-Host "[COPY] Binary copied to $DestBinary" -ForegroundColor Green
} else {
    # Try alternate name
    $SourceBinary = "$ProjectRoot\target\$Profile\leviathan.exe"
    if (Test-Path $SourceBinary) {
        Copy-Item $SourceBinary $DestBinary -Force
        Write-Host "[COPY] Binary copied to $DestBinary" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Binary not found" -ForegroundColor Red
        exit 1
    }
}

# Copy supporting files
$ConfigDir = "$DistDir\config"
New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null

# Create default config
$DefaultConfig = @"
# Leviathan AI Configuration
# Enterprise AI Orchestration System

[system]
name = "Leviathan AI"
organization = "Your Organization"
environment = "production"
audit_enabled = true
max_agents = 100
retention_days = 2555  # 7 years for financial compliance

[compliance]
frameworks = ["BCBS239", "FFIEC", "SR117"]
auto_validate = true
report_format = "json"

[swarm]
topology = "hierarchical"
max_parallel = 8
timeout_seconds = 300

[audit]
enabled = true
export_path = "./audit"
hash_algorithm = "blake3"
sign_reports = true

[ui]
theme = "cyberpunk"  # win95 | cyberpunk | corporate
refresh_rate_ms = 250
"@

$DefaultConfig | Out-File -FilePath "$ConfigDir\leviathan.toml" -Encoding UTF8

Write-Host "[CONFIG] Default configuration created" -ForegroundColor Green

# Create installer if requested
if ($Installer) {
    Write-Host "[INSTALLER] Creating Windows installer..." -ForegroundColor Yellow

    # Create Inno Setup script
    $InnoScript = @"
[Setup]
AppName=Leviathan AI
AppVersion=0.1.0
AppPublisher=Leviathan AI Inc.
AppPublisherURL=https://leviathan.ai
DefaultDirName={autopf}\LeviathanAI
DefaultGroupName=Leviathan AI
OutputDir=$DistDir
OutputBaseFilename=LeviathanAI-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern

[Files]
Source: "$DestBinary"; DestDir: "{app}"; Flags: ignoreversion
Source: "$ConfigDir\*"; DestDir: "{app}\config"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\Leviathan AI"; Filename: "{app}\leviathan.exe"; Parameters: "ui"
Name: "{group}\Leviathan Terminal"; Filename: "{app}\leviathan.exe"
Name: "{group}\Uninstall Leviathan AI"; Filename: "{uninstallexe}"

[Registry]
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; \
    ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; \
    Check: NeedsAddPath('{app}')

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE,
    'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
    'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;
"@

    $InnoScriptPath = "$DistDir\installer.iss"
    $InnoScript | Out-File -FilePath $InnoScriptPath -Encoding UTF8

    # Check if Inno Setup is installed
    $InnoCompiler = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    if (Test-Path $InnoCompiler) {
        & $InnoCompiler $InnoScriptPath
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[INSTALLER] Installer created: $DistDir\LeviathanAI-Setup.exe" -ForegroundColor Green
        }
    } else {
        Write-Host "[WARN] Inno Setup not found. Installer script created at $InnoScriptPath" -ForegroundColor Yellow
        Write-Host "       Install Inno Setup 6 to build the installer" -ForegroundColor Yellow
    }
}

# Create portable ZIP
Write-Host "[PACKAGE] Creating portable ZIP..." -ForegroundColor Yellow
$ZipPath = "$DistDir\leviathan-ai-windows-x64.zip"
Compress-Archive -Path "$DistDir\leviathan.exe", "$ConfigDir" -DestinationPath $ZipPath -Force
Write-Host "[PACKAGE] Portable ZIP created: $ZipPath" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    BUILD COMPLETE                          ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Output directory: $DistDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "Files created:" -ForegroundColor Cyan
Get-ChildItem $DistDir -Recurse | ForEach-Object {
    $RelPath = $_.FullName.Replace($DistDir, "").TrimStart("\")
    $Size = if ($_.PSIsContainer) { "[DIR]" } else { "{0:N0} KB" -f ($_.Length / 1KB) }
    Write-Host "  $RelPath - $Size"
}

Write-Host ""
Write-Host "To run: $DestBinary --help" -ForegroundColor Yellow
