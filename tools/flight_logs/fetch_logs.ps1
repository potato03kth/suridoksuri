# fetch_logs.ps1 — RPi의 logs/ 를 개발컴(Windows)으로 증분 회수 (scp over SSH)
#
# 신규 플라이트 폴더만 복사한다 (이미 로컬에 있는 폴더는 건너뜀).
#
# 사용 예:
#   .\fetch_logs.ps1                                   # 기본값 (RPI_REMOTE 환경변수 또는 pi@raspberrypi.local)
#   .\fetch_logs.ps1 -Remote drone@192.168.0.42
#   .\fetch_logs.ps1 -Remote pi@10.42.0.1 -RemotePath /drone_ws/src/suridoksuri/logs
#
# 주의: RemotePath는 RPi 배포 검증 때 확정할 것 — Docker 컨테이너 `fc` 안 경로가
#       호스트에 볼륨 마운트돼 있는지에 따라 호스트 쪽 실제 경로가 달라진다.

param(
    [string]$Remote = $(if ($env:RPI_REMOTE) { $env:RPI_REMOTE } else { "pi@raspberrypi.local" }),
    [string]$RemotePath = "~/drone_ws/src/suridoksuri/logs",
    [string]$LocalPath = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) "logs")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $LocalPath)) {
    New-Item -ItemType Directory -Force $LocalPath | Out-Null
}

Write-Host "[fetch_logs] 원격: ${Remote}:${RemotePath}"
Write-Host "[fetch_logs] 로컬: $LocalPath"

# 원격 플라이트 폴더 목록 (stderr는 원격 셸에서 버림 — PS 5.1의 2>&1 래핑 회피)
$remoteDirs = ssh $Remote "ls -1 $RemotePath 2>/dev/null"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[fetch_logs] 실패: SSH 접속 불가 또는 원격에 $RemotePath 없음" -ForegroundColor Red
    exit 1
}
$remoteDirs = @($remoteDirs | Where-Object { $_ -match '^\d{4}-\d{2}-\d{2}_flight\d+' })
if ($remoteDirs.Count -eq 0) {
    Write-Host "[fetch_logs] 원격에 플라이트 폴더가 없다."
    exit 0
}

$localDirs = @(Get-ChildItem -Path $LocalPath -Directory | ForEach-Object { $_.Name })
$newDirs = @($remoteDirs | Where-Object { $localDirs -notcontains $_ })

if ($newDirs.Count -eq 0) {
    Write-Host "[fetch_logs] 신규 폴더 없음 (원격 $($remoteDirs.Count)개 전부 로컬에 있음)."
    exit 0
}

Write-Host "[fetch_logs] 신규 $($newDirs.Count)개 복사: $($newDirs -join ', ')"
$failed = 0
foreach ($d in $newDirs) {
    scp -r "${Remote}:${RemotePath}/$d" $LocalPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[fetch_logs] 복사 실패: $d" -ForegroundColor Red
        $failed++
    } else {
        Write-Host "[fetch_logs] OK: $d"
    }
}

if ($failed -gt 0) {
    Write-Host "[fetch_logs] $failed 개 실패 — 재실행하면 실패분만 다시 시도한다." -ForegroundColor Yellow
    exit 1
}
Write-Host "[fetch_logs] 완료. 분석: Claude에게 로그 폴더 경로를 주거나 pyulog 사용 (README.md 참조)"
