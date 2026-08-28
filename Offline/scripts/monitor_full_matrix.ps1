param(
    [int]$IntervalSeconds = 600
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$LogRoot = Join-Path $ProjectRoot 'logs\full_matrix'
$StatusPath = Join-Path $LogRoot 'status.json'
$MonitorPath = Join-Path $LogRoot 'monitor_status.json'
$MonitorLog = Join-Path $LogRoot 'monitor.log'
$InferencePorts = @(28001, 28002, 28003)
$AllPorts = @(28001, 28002, 28003, 8001)
$ExpectedModels = @{
    28001 = 'Qwen/Qwen3-VL-4B-Instruct'
    28002 = 'Qwen/Qwen3-VL-4B-Instruct'
    28003 = 'Qwen/Qwen3-VL-4B-Instruct'
    8001 = 'Qwen/Qwen3-VL-Embedding-2B'
}
$Completed = [System.Collections.Generic.HashSet[string]]::new()

function Write-MonitorLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $MonitorLog -Value $line -Encoding UTF8
    Write-Output $line
}

function Test-ModelPort([int]$Port) {
    try {
        $payload = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 20
        $models = @($payload.data | ForEach-Object { [string]$_.id })
        return [pscustomobject]@{
            Port = $Port
            Healthy = $models -contains $ExpectedModels[$Port]
            Models = $models
            Error = ''
        }
    }
    catch {
        return [pscustomobject]@{
            Port = $Port
            Healthy = $false
            Models = @()
            Error = $_.Exception.Message
        }
    }
}

function Restore-InferenceTunnel {
    Write-MonitorLog 'Inference tunnel unhealthy; starting scoped SSH recovery.'
    $markers = @(
        'haozhen@8.209.211.218',
        '28001:127.0.0.1:8013',
        '28002:127.0.0.1:8014',
        '28003:127.0.0.1:8015'
    )
    $existing = Get-CimInstance Win32_Process -Filter "Name = 'ssh.exe'" |
        Where-Object {
            $command = [string]$_.CommandLine
            ($markers | Where-Object { $command -notlike "*$_*" }).Count -eq 0
        }
    foreach ($process in $existing) {
        Write-MonitorLog "Stopping stale scoped SSH process pid=$($process.ProcessId)."
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }
    $arguments = @(
        '-N',
        '-o', 'ServerAliveInterval=30',
        '-o', 'ServerAliveCountMax=6',
        '-o', 'TCPKeepAlive=yes',
        '-o', 'ExitOnForwardFailure=yes',
        '-o', 'ConnectTimeout=15',
        '-L', '28001:127.0.0.1:8013',
        '-L', '28002:127.0.0.1:8014',
        '-L', '28003:127.0.0.1:8015',
        'haozhen@8.209.211.218'
    )
    $process = Start-Process -FilePath 'ssh.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru
    Write-MonitorLog "Started SSH recovery process pid=$($process.Id)."
    Start-Sleep -Seconds 10
}

if ($IntervalSeconds -lt 60) {
    throw 'IntervalSeconds must be at least 60.'
}
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
Write-MonitorLog "Monitor started; interval=${IntervalSeconds}s pid=$PID."

while ($true) {
    $portRows = @($AllPorts | ForEach-Object { Test-ModelPort $_ })
    $downInference = @($portRows | Where-Object { -not $_.Healthy -and $_.Port -in $InferencePorts })
    if ($downInference.Count -gt 0) {
        Write-MonitorLog "Down inference ports: $($downInference.Port -join ',')."
        Restore-InferenceTunnel
        $portRows = @($AllPorts | ForEach-Object { Test-ModelPort $_ })
    }

    $jobCounts = @{}
    $newCompleted = @()
    $runnerPid = 0
    $runnerAlive = $false
    if (Test-Path -LiteralPath $StatusPath) {
        $status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
        $runnerPid = [int]$status.runner_pid
        $jobs = @($status.jobs.PSObject.Properties)
        foreach ($group in ($jobs | Group-Object { [string]$_.Value.status })) {
            $jobCounts[$group.Name] = $group.Count
        }
        foreach ($job in ($jobs | Where-Object { $_.Value.status -eq 'completed' })) {
            if ($Completed.Add($job.Name)) {
                $newCompleted += $job.Name
            }
        }
        if ($runnerPid -gt 0) {
            & wsl.exe -e sh -c "kill -0 $runnerPid 2>/dev/null"
            $runnerAlive = $LASTEXITCODE -eq 0
        }
    }

    $report = [ordered]@{
        checked_at = (Get-Date -Format o)
        monitor_pid = $PID
        runner_pid = $runnerPid
        runner_alive = $runnerAlive
        ports = @($portRows)
        job_counts = $jobCounts
        newly_completed = $newCompleted
    }
    $temporary = "$MonitorPath.tmp"
    $report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $MonitorPath -Force
    $health = ($portRows | ForEach-Object { "$($_.Port)=$($_.Healthy)" }) -join ','
    Write-MonitorLog "runner_alive=$runnerAlive jobs=$($jobCounts | ConvertTo-Json -Compress) ports=$health"
    foreach ($name in $newCompleted) {
        Write-MonitorLog "NEW_COMPLETED $name"
    }
    Start-Sleep -Seconds $IntervalSeconds
}
