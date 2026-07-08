param(
    [string]$ModelRoot = "models\azure-embedded",
    [string]$GrpcUrl = "127.0.0.1:8793",
    [string]$ExecutablePath = "native\azure_embedded_speech_grpc\build\tts\Release\azure_embedded_tts_grpc.exe",
    [string]$OutLog = "azure_embedded_tts.out.log",
    [string]$ErrLog = "azure_embedded_tts.err.log",
    [switch]$NoSmoke
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

$resolvedExe = Resolve-Path $ExecutablePath
Get-NetTCPConnection -LocalPort 8793 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

$env:VOICE_AGENT_AZURE_EMBEDDED_MODEL_ROOT = $ModelRoot
$env:AZURE_EMBEDDED_TTS_GRPC_URL = $GrpcUrl
$env:VOICE_AGENT_AZURE_EMBEDDED_TTS_PRELOAD = "0"
$env:VOICE_AGENT_AZURE_EMBEDDED_TTS_WARMUP = "0"
Remove-Item Env:VOICE_AGENT_AZURE_EMBEDDED_TTS_MODEL_KEY -ErrorAction SilentlyContinue

$envLine = Get-Content .env -ErrorAction SilentlyContinue | Where-Object { $_ -match '^PASCO_MODEL_KEY=' } | Select-Object -First 1
if ($envLine) {
    $env:PASCO_MODEL_KEY = $envLine.Substring('PASCO_MODEL_KEY='.Length).Trim().Trim('"')
}

Remove-Item $OutLog,$ErrLog -ErrorAction SilentlyContinue
$process = Start-Process -FilePath $resolvedExe -WorkingDirectory $root -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru
Write-Host "Azure Embedded TTS sidecar PID: $($process.Id)"

if (-not $NoSmoke) {
    $env:PYTHONPATH = "src"
    @'
from pathlib import Path
import grpc
from voice_agent.config import Settings
from voice_agent.providers import azure_embedded_pb2 as pb
from voice_agent.providers import azure_embedded_pb2_grpc as pb_grpc

settings = Settings.from_env({}, base_dir=Path.cwd())
channel = grpc.insecure_channel(settings.audio.azure_embedded_tts_grpc_url)
grpc.channel_ready_future(channel).result(timeout=10)
stub = pb_grpc.AzureEmbeddedSpeechStub(channel)
health = stub.Health(pb.HealthRequest(), timeout=10)
print("HEALTH_STATUS", health.status)
response = stub.Synthesize(pb.TtsRequest(voice="azure-embedded-zh-CN-XiaoxiaoNeuralV6", locale="zh-CN", text="你好，本地高清语音测试。", sample_rate_hz=24000), timeout=60)
print("SYNTH_BYTES", len(response.audio), response.audio[:12])
'@ | py -3.11 -
}

Get-Process -Id $process.Id | Select-Object Id,ProcessName,Responding,HasExited