#!/usr/bin/env bash
set -euo pipefail

# Start both SSH tunnels as hidden background processes in Windows PowerShell.
# Authentication must be available non-interactively (SSH key or ssh-agent).
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command '
$ErrorActionPreference = "Stop"

$tunnels = @(
    @{ LocalPort = 18001; RemotePort = 8001 },
    @{ LocalPort = 18000; RemotePort = 8000 }
)

foreach ($tunnel in $tunnels) {
    $localPort = $tunnel.LocalPort
    $remotePort = $tunnel.RemotePort

    $listener = Get-NetTCPConnection `
        -LocalPort $localPort `
        -State Listen `
        -ErrorAction SilentlyContinue

    if ($listener) {
        Write-Host "Local port $localPort is already listening; skipping."
        continue
    }

    $sshArgs = @(
        "-N",
        "-L", "${localPort}:localhost:${remotePort}",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "a2-via-sg"
    )

    $process = Start-Process `
        -FilePath "ssh.exe" `
        -ArgumentList $sshArgs `
        -WindowStyle Hidden `
        -PassThru

    Start-Sleep -Milliseconds 500
    if ($process.HasExited) {
        throw "Failed to start tunnel localhost:${localPort} -> a2-via-sg:localhost:${remotePort}."
    }

    Write-Host "Started tunnel localhost:${localPort} -> a2-via-sg:localhost:${remotePort} (PID $($process.Id))."
}
'
