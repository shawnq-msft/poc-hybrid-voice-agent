# Hybrid Voice Agent POC

This repository is a Windows-first implementation scaffold for a full-duplex local/cloud hybrid voice assistant. The target runtime is a Python Pipecat backend with a browser WebRTC client, local-first speech providers, Foundry Local for ASR and LLM where available, Windows speech synthesis for TTS, and a guarded tool layer for local Copilot/VS Code automation.

## Architecture

```text
Browser microphone/speaker
  -> WebRTC signaling and media transport
  -> Pipecat pipeline boundary
  -> local VAD
  -> local ASR, preferably Foundry Local ASR
  -> Foundry Local LLM
  -> allowlisted Copilot tool router
  -> Windows TTS
  -> browser audio output
```

The current implementation establishes the project structure, configuration model, provider adapter boundaries, health checks, a static browser client, and tests for the highest-risk routing and tool policy decisions. The media pipeline is intentionally mock-friendly so it can be validated before downloading large models.

## Model choices

- VAD: Silero VAD first, WebRTC VAD as an ultra-light fallback.
- ASR: Foundry Local streaming speech models first. The current options are `nemotron-3.5-asr-streaming-0.6b` for multilingual/auto-detect and `nemotron-speech-streaming-en-0.6b` for English. `faster-whisper` remains available only as a separate local fallback implementation.
- Azure Embedded Speech: local ASR/TTS model assets live under `models/azure-embedded`, which is ignored by Git. The model key belongs in `.env` as `PASCO_MODEL_KEY`; do not commit it.
- LLM: Foundry Local. This machine currently has `qwen2.5-0.5b-instruct-cuda-gpu:4` available; use Gemma only if it appears in your Foundry Local catalog.
- TTS: Azure Embedded HD voices can run through the native gRPC sidecar; Edge neural TTS and Windows SAPI remain selectable fallbacks.

## Setup

```powershell
cd d:\dev\poc-hybrid-voice-agent
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev,local-audio]"
Copy-Item .env.example .env
```

If PowerShell blocks activation scripts, use the Python interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,local-audio]"
```

## Foundry Local checks

Before wiring real audio, confirm the local endpoint and model IDs on your machine. The defaults in `.env.example` are placeholders for the desired shape, not guaranteed catalog names.

Expected checks during implementation:

```powershell
foundry model list
foundry model download <gemma-model-id>
foundry model download <asr-model-id>
```

Then update these values in `.env`:

```text
VOICE_AGENT_FOUNDRY_ENDPOINT=<leave empty to auto-discover, or set http://127.0.0.1:<port>/v1>
VOICE_AGENT_FOUNDRY_LLM_MODEL=<exact-chat-model-id, for example qwen2.5-0.5b-instruct-cuda-gpu:4>
VOICE_AGENT_FOUNDRY_ASR_MODEL=nemotron-3.5-asr-streaming-0.6b
VOICE_AGENT_FOUNDRY_TIMEOUT_SECONDS=180
```

## Azure Embedded Speech assets

The local model assets are ignored by Git under `models/`. Keep encrypted ASR assets and extracted TTS voice folders in this layout:

```text
models/azure-embedded/asr/zh-CN/encrypted/35M/
models/azure-embedded/asr/en-GB/encrypted/v6/35M/
models/azure-embedded/tts/zh-CN/XiaoxiaoNeuralHD/
models/azure-embedded/tts/en-US/AvaNeuralHDv2/
```

The TTS zip files are external inputs and should not be committed. To populate the local ignored model library:

```powershell
New-Item -ItemType Directory -Force models\azure-embedded\tts\zh-CN,models\azure-embedded\tts\en-US
tar -xf C:\Users\shawnq\Downloads\XiaoxiaoNeuralHD.zip -C models\azure-embedded\tts\zh-CN
tar -xf C:\Users\shawnq\Downloads\AvaNeuralHDv2.zip -C models\azure-embedded\tts\en-US
```

Set these values in `.env`:

```text
PASCO_MODEL_KEY=<local model key>
VOICE_AGENT_AZURE_EMBEDDED_GRPC_URL=127.0.0.1:8792
VOICE_AGENT_AZURE_EMBEDDED_ASR_LOCALE=zh-CN
VOICE_AGENT_AZURE_EMBEDDED_ASR_ZH_CN_MODEL_DIR=models/azure-embedded/asr/zh-CN/encrypted/35M
VOICE_AGENT_AZURE_EMBEDDED_ASR_EN_GB_MODEL_DIR=models/azure-embedded/asr/en-GB/encrypted/v6/35M
VOICE_AGENT_AZURE_EMBEDDED_TTS_VOICE=azure-embedded-zh-CN-XiaoxiaoNeuralHD
VOICE_AGENT_AZURE_EMBEDDED_TTS_ZH_CN_MODEL_DIR=models/azure-embedded/tts/zh-CN/XiaoxiaoNeuralHD
VOICE_AGENT_AZURE_EMBEDDED_TTS_EN_US_MODEL_DIR=models/azure-embedded/tts/en-US/AvaNeuralHDv2
```

Current status: the Python provider boundary now targets a native gRPC sidecar on `127.0.0.1:8792` using `protos/azure_embedded_speech.proto`. The C++ scaffold lives in `native/azure_embedded_speech_grpc` and targets Azure Speech SDK `1.47`. This machine still needs a C++ toolchain, gRPC C++/protobuf, and Speech SDK C++ headers/libs before that sidecar can be compiled. The current ASR model assets are incompatible with the 1.47 recognition runtime, so the legacy .NET sidecar remains in `AzureEmbeddedSpeech` as a module-runnable compatibility path pinned to Speech SDK `1.24.2`. Python ASR tries native gRPC first and falls back to the legacy WebSocket sidecar while the native 1.47 path is unavailable.

Generate Python gRPC stubs after installing dev dependencies:

```powershell
python scripts\generate_grpc_stubs.py
```

Build the native sidecar once CMake, vcpkg/gRPC, protobuf, and the Speech SDK C++ package are available:

```powershell
cmake -S native\azure_embedded_speech_grpc -B native\azure_embedded_speech_grpc\build -DSPEECHSDK_ROOT=<speech-sdk-cpp-root>
cmake --build native\azure_embedded_speech_grpc\build --config Release
```

The old decrypted ASR folders are not used and should remain absent:

```text
models/azure-embedded-asr/*/decrypted/  # removed
```

Run the sidecar:

```powershell
$env:PASCO_MODEL_KEY = (Get-Content .env | Select-String '^PASCO_MODEL_KEY=').ToString().Split('=',2)[1].Trim('"')
$env:VOICE_AGENT_AZURE_EMBEDDED_MODEL_ROOT = 'models/azure-embedded'
dotnet run --project AzureEmbeddedSpeech\AzureEmbeddedSpeech.csproj  # legacy migration reference
```

Validate it:

```powershell
Invoke-RestMethod http://127.0.0.1:8791/health | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8791/models/load -Method Post | ConvertTo-Json -Depth 5
```

## Streaming transport

Whisper-style ASR models use `/api/session/turn-ws`, a WebSocket protocol that sends the complete browser recording after VAD end:

```text
browser -> JSON config
browser -> binary recorded audio
browser -> JSON end
server  -> JSON progress events
server  -> binary TTS audio chunks
server  -> JSON result/done
```

Azure Embedded ASR uses a streaming split path. On browser VAD start, the frontend opens the sidecar WebSocket and streams 16 kHz PCM frames directly to `AzureEmbeddedSpeech`. Sidecar partial results update the `You` transcript immediately. On browser VAD end, the frontend sends `end`, waits for the final ASR result, then sends that text to `/api/session/text-turn-ws` so the backend runs only LLM and TTS streaming. This leaves the Foundry Local Nemotron ASR blob path unchanged.

The frontend queues binary TTS chunks and starts playback as soon as each chunk arrives. `/api/session/turn-events` remains as an NDJSON fallback for the classic audio-turn path.

Model warmup runs in the background when the FastAPI server starts, and can still be triggered manually with `Load Models`.

## Run

```powershell
python -m voice_agent.server
```

Open `http://127.0.0.1:8787`, click `Load Models`, then click `Start` and grant microphone permission. The browser shows live volume and VAD state. When VAD detects speech followed by trailing silence, Azure Embedded ASR final text is sent to `/api/session/text-turn-ws`; Foundry Local Nemotron ASR sends the captured utterance to `/api/session/turn-ws`. The backend then calls Foundry Local chat completions and Windows/Edge TTS. If TTS synthesis fails, the browser uses `speechSynthesis` as a playback fallback.

The UI shows current and average latency for ASR, LLM, TTS, backend total time, and speech-to-speech end-to-end time. Model warmup latency is shown as the current model status but is not included in turn averages.

If the browser blocks microphone permission, click `Backend Check`. It calls `/api/session/backend-check`, which generates a local WAV and runs the same real ASR -> Foundry Local LLM -> Edge TTS backend chain without using mock providers.

If ASR returns empty text, the UI now restarts the recorder automatically. Speak again for at least one second and click `Send Turn` again.

Check readiness before trying a real turn:

```powershell
Invoke-RestMethod http://127.0.0.1:8787/api/ready | ConvertTo-Json -Depth 8
```

If Foundry Local is not running or the model IDs are wrong, `/api/session/turn` returns `503` with a setup hint instead of falling back to mock behavior.

## Tests

The core tests use only the standard library so they can run before optional audio dependencies are installed:

```powershell
python -m unittest discover -s tests
```

Run the mock end-to-end smoke turn:

```powershell
$env:PYTHONPATH='src'
python -m voice_agent.smoke
```

With the server running, the same smoke path is available at:

```powershell
Invoke-RestMethod http://127.0.0.1:8787/api/smoke | ConvertTo-Json -Depth 8
```

Run the direct real-chain validation with a synthesized WAV input:

```powershell
$env:PYTHONPATH='src'
py -3.11 scripts\real_chain_check.py
```

Run the HTTP-level real turn validation against the server:

```powershell
$env:PYTHONPATH='src'
py -3.11 scripts\http_turn_check.py
```

After installing the package in editable mode, `pytest` also works:

```powershell
python -m pytest
```

## Copilot automation safety

Automatic local Copilot/VS Code control is implemented behind an allowlist policy. By default, requests run in dry-run mode and are written to `.voice-agent/audit.jsonl`. Keep `VOICE_AGENT_COPILOT_TOOLS_DRY_RUN=true` until the allowlist and audit log are reviewed.

The tool layer rejects unknown actions, unexpected arguments, paths outside the workspace, and arbitrary shell command execution.
