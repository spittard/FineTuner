<#
.SYNOPSIS
  Send bridge/TO_CLAUDE.md to local Claude Code (`claude -p`) and capture output.

.DESCRIPTION
  Implements the file-based Cursor <-> Claude collaboration described in bridge/README.md.
  Requires `claude` on PATH (Claude Code CLI).

.PARAMETER InFile
  Markdown file whose body is passed as the print-mode prompt (default: bridge/TO_CLAUDE.md).

.PARAMETER OutFile
  Where combined stdout/stderr is written (default: bridge/FROM_CLAUDE.last.txt).
#>
param(
    [string]$InFile = "",
    [string]$OutFile = ""
)

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not $InFile) { $InFile = Join-Path $Root "bridge\TO_CLAUDE.md" }
if (-not $OutFile) { $OutFile = Join-Path $Root "bridge\FROM_CLAUDE.last.txt" }

if (-not (Test-Path -LiteralPath $InFile)) {
    Write-Error "Missing handoff file: $InFile`nCopy bridge\TO_CLAUDE.example.md -> bridge\TO_CLAUDE.md and edit."
}

New-Item -ItemType Directory -Force -Path (Split-Path $OutFile) | Out-Null

$prompt = Get-Content -LiteralPath $InFile -Raw -Encoding UTF8
if ([string]::IsNullOrWhiteSpace($prompt)) {
    Write-Error "Handoff file is empty: $InFile"
}

$header = @"
================================================================================
bridge_run_claude.ps1 @ $(Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz")
in:  $InFile
out: $OutFile
================================================================================

"@

Write-Host "Invoking: claude -p <TO_CLAUDE.md> (print mode) ..."
Write-Host "Output -> $OutFile"

Set-Content -LiteralPath $OutFile -Value $header -Encoding UTF8

# Capture all streams; preserve claude exit code (Tee-Object in a pipeline can mask it).
# Claude prints a harmless stdin warning to stderr; Stop would turn it into a terminating error.
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $output = & claude -p $prompt --permission-mode acceptEdits 2>&1
    $code = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $prevEap
}
$text = if ($null -eq $output) { "" } elseif ($output -is [string]) { $output } else { ($output | ForEach-Object { "$_" }) -join "`n" }
Add-Content -LiteralPath $OutFile -Value $text -Encoding UTF8
Write-Host "Done. Read: $OutFile (exit $code)"
exit $code
