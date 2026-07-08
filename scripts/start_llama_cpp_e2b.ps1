param(
    [string]$ModelPath = "models\llm\gemma-3n-e2b-it\gemma-3n-E2B-it-Q4_K_M.gguf",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8080,
    [string]$ReleaseTag = "b9860",
    [string]$Backend = "cpu",
    [int]$ContextSize = 4096,
    [int]$Threads = [Math]::Max(1, [Environment]::ProcessorCount - 1)
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$resolvedModelPath = Join-Path $root $ModelPath
if (-not (Test-Path $resolvedModelPath)) {
    throw "Model file not found: $resolvedModelPath. Download it with: hf download unsloth/gemma-3n-E2B-it-GGUF gemma-3n-E2B-it-Q4_K_M.gguf --local-dir models/llm/gemma-3n-e2b-it"
}

$assetName = switch ($Backend.ToLowerInvariant()) {
    "vulkan" { "llama-$ReleaseTag-bin-win-vulkan-x64.zip" }
    default { "llama-$ReleaseTag-bin-win-cpu-x64.zip" }
}

$installDir = Join-Path $root "models\llama-cpp\$ReleaseTag-$Backend"
$serverPath = Join-Path $installDir "llama-server.exe"
if (-not (Test-Path $serverPath)) {
    New-Item -ItemType Directory -Force -Path $installDir | Out-Null
    $zipPath = Join-Path $installDir $assetName
    $url = "https://github.com/ggml-org/llama.cpp/releases/download/$ReleaseTag/$assetName"
    Write-Host "Downloading llama.cpp $Backend server from $url"
    Invoke-WebRequest -Uri $url -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $installDir -Force
    Remove-Item $zipPath -Force
}

Write-Host "Starting llama.cpp server on http://${HostName}:$Port"
Write-Host "Model: $resolvedModelPath"
& $serverPath --host $HostName --port $Port --model $resolvedModelPath --ctx-size $ContextSize --threads $Threads