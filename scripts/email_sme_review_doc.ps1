# Send SME review .docx via SMTP (no Outlook). Requires env vars — see scripts/send_smtp_email.py docstring.
param(
    [string] $Recipient = "scott.pittard@gmail.com"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
python scripts/send_smtp_email.py --to $Recipient
