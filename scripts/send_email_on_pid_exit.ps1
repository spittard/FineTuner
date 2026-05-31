$ErrorActionPreference = "Stop"

param(
    [Parameter(Mandatory = $true)]
    [int]$TargetPid,

    [Parameter(Mandatory = $true)]
    [string]$Recipient
)

$start = Get-Date
while (Get-Process -Id $TargetPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}
$end = Get-Date

$subject = "FineTuner rebuild process completed"
$body = @"
The watched process has finished.

PID: $TargetPid
Started watching: $start
Completed: $end
Host: $env:COMPUTERNAME
"@

try {
    $outlook = New-Object -ComObject Outlook.Application
    $mail = $outlook.CreateItem(0)
    $mail.To = $Recipient
    $mail.Subject = $subject
    $mail.Body = $body
    $mail.Send()
} catch {
    $log = "e:\projects\FineTuner\FineTuner\email_notify_fallback.log"
    $line = "[{0}] Email send failed: {1}" -f (Get-Date -Format s), $_.Exception.Message
    Add-Content -Path $log -Value $line
}
