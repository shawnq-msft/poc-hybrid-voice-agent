# AzureEmbeddedSpeech

Long-lived WebSocket sidecar for Azure Embedded Speech ASR, with the project/module name widened for future Embedded TTS support.

Current status: protocol, model registry, health, model load, asset validation, and streaming ASR recognition are implemented. `/asr` starts `EmbeddedSpeechConfig` continuous recognition on `start`, writes incoming PCM frames into `PushAudioInputStream`, emits partial recognition events, and returns final text after `end`. Speech SDK embedded packages use the latest project-wide Speech SDK version.

Run from the repository root:

```powershell
$env:PASCO_MODEL_KEY = (Get-Content .env | Select-String '^PASCO_MODEL_KEY=').ToString().Split('=',2)[1].Trim('"')
$env:VOICE_AGENT_AZURE_EMBEDDED_MODEL_ROOT = 'models/azure-embedded'
dotnet run --project AzureEmbeddedSpeech/AzureEmbeddedSpeech.csproj
```

Protocol:

```text
client -> { "type": "start", "locale": "zh-CN" }
client -> binary PCM 16 kHz 16-bit mono frames
client -> { "type": "end" }

server -> { "type": "ready" }
server -> { "type": "partial", "text": "..." }
server -> { "type": "final", "text": "...", "asrMs": 123 }
```
