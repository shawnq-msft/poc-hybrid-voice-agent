from __future__ import annotations

import re
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


GEMMA_4_E2B_ASR_PROVIDER = "gemma-4-e2b"
GEMMA_4_E2B_HF_MODEL_ID = "google/gemma-4-E2B"
GEMMA_4_E2B_FULL_INSTRUCT_MODEL_ID = "google/gemma-4-E2B-it"
GEMMA_4_E2B_INSTRUCT_MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
GEMMA_4_E2B_MAX_AUDIO_SECONDS = 30
LFM2_AUDIO_PROVIDER = "lfm2-audio"
LFM2_AUDIO_MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"
GEMMA_4_E2B_MODEL_ALIASES = frozenset(
    {
        "gemma-4-e2b",
        "gemma-4-e2b-it",
        GEMMA_4_E2B_HF_MODEL_ID,
        GEMMA_4_E2B_FULL_INSTRUCT_MODEL_ID,
        GEMMA_4_E2B_INSTRUCT_MODEL_ID,
    }
)


def is_gemma_4_e2b_model(model: str | None) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized in {alias.lower() for alias in GEMMA_4_E2B_MODEL_ALIASES}


LOCAL_ASR_PROVIDERS = {"foundry-local", "faster-whisper", "azure-embedded", GEMMA_4_E2B_ASR_PROVIDER, LFM2_AUDIO_PROVIDER, "whisper-cpp", "vosk"}
ASR_PROVIDER_CAPABILITIES = {
    "azure-embedded": {"inferenceMode": "streaming", "transportMode": "streaming", "vadRole": "start"},
    "foundry-local": {"inferenceMode": "streaming", "transportMode": "streaming", "vadRole": "start-end"},
    "faster-whisper": {"inferenceMode": "batch", "transportMode": "streaming", "vadRole": "start-end"},
    GEMMA_4_E2B_ASR_PROVIDER: {
        "inferenceMode": "batch",
        "transportMode": "streaming",
        "vadRole": "start-end",
        "maxAudioSeconds": GEMMA_4_E2B_MAX_AUDIO_SECONDS,
        "requiresLlmProvider": GEMMA_4_E2B_ASR_PROVIDER,
        "requiresLlmModel": GEMMA_4_E2B_INSTRUCT_MODEL_ID,
    },
    LFM2_AUDIO_PROVIDER: {
        "inferenceMode": "audio-to-audio",
        "transportMode": "streaming",
        "vadRole": "start-end",
        "requiresLlmProvider": LFM2_AUDIO_PROVIDER,
        "requiresLlmModel": LFM2_AUDIO_MODEL_ID,
        "requiresTtsProvider": LFM2_AUDIO_PROVIDER,
        "requiresTtsModel": LFM2_AUDIO_MODEL_ID,
    },
    "whisper-cpp": {"inferenceMode": "batch", "transportMode": "streaming", "vadRole": "start-end"},
    "vosk": {"inferenceMode": "batch", "transportMode": "streaming", "vadRole": "start-end"},
}
FOUNDRY_STREAMING_ASR_MODELS = {
    "nemotron-3.5-asr-streaming-0.6b": {
        "language": "auto",
        "label": "Nemotron ASR multilingual 0.6B",
        "modelId": "nemotron-3.5-asr-streaming-0.6b-generic-cpu:3",
    },
    "nemotron-speech-streaming-en-0.6b": {
        "language": "en",
        "label": "Nemotron Speech English 0.6B",
        "modelId": "nemotron-speech-streaming-en-0.6b-generic-cpu:3",
    },
}
LOCAL_TTS_PROVIDERS = {"windows-winrt", "windows-sapi", "edge-tts", "azure-speech", "azure-embedded", LFM2_AUDIO_PROVIDER}
LOCAL_VAD_PROVIDERS = {"silero", "webrtcvad", "energy"}
LOCAL_LLM_PROVIDERS = {"foundry-local", "llama-cpp", GEMMA_4_E2B_ASR_PROVIDER, LFM2_AUDIO_PROVIDER}


def _env(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name)
    if value is None or value == "":
        return default
    return value


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = _env(env, name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    value = _env(env, name, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _default_foundry_endpoint(env: Mapping[str, str]) -> str:
    configured = env.get("VOICE_AGENT_FOUNDRY_ENDPOINT")
    if configured:
        return configured
    discovered = discover_foundry_endpoint()
    return discovered or "http://127.0.0.1:5273/v1"


def discover_foundry_endpoint() -> str | None:
    try:
        completed = subprocess.run(
            ["foundry", "service", "status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = f"{completed.stdout}\n{completed.stderr}"
    match = re.search(r"(http://127\.0\.0\.1:\d+)(?:/openai/status)?", output)
    if not match:
        return None
    return f"{match.group(1)}/v1"


@dataclass(frozen=True)
class ProviderSettings:
    vad: str = "silero"
    asr: str = "foundry-local"
    tts: str = "azure-embedded"
    llm: str = "foundry-local"
    cloud_fallback_enabled: bool = False

    def validate(self) -> None:
        if self.vad not in LOCAL_VAD_PROVIDERS:
            raise ValueError(f"Unsupported VAD provider: {self.vad}")
        if self.asr not in LOCAL_ASR_PROVIDERS:
            raise ValueError(f"Unsupported ASR provider: {self.asr}")
        if self.tts not in LOCAL_TTS_PROVIDERS:
            raise ValueError(f"Unsupported TTS provider: {self.tts}")
        if self.llm not in LOCAL_LLM_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {self.llm}")


@dataclass(frozen=True)
class FoundrySettings:
    endpoint: str = "http://127.0.0.1:5273/v1"
    llm_model: str = "qwen2.5-0.5b-instruct-cuda-gpu:4"
    asr_model: str = "nemotron-3.5-asr-streaming-0.6b"
    timeout_seconds: float = 180.0


@dataclass(frozen=True)
class LlamaCppSettings:
    endpoint: str = "http://127.0.0.1:8080"
    model: str = "gemma-3n-e2b-it"
    model_path: Path = Path("models/llm/gemma-3n-e2b-it/gemma-3n-E2B-it-Q4_K_M.gguf")
    slot_id: int = 0
    timeout_seconds: float = 180.0


@dataclass(frozen=True)
class Gemma4E2BSettings:
    model_id: str = GEMMA_4_E2B_INSTRUCT_MODEL_ID
    model_dir: Path = Path("models/llm/gemma-4-e2b")
    allow_download: bool = False
    max_new_tokens: int = 128
    audio_turn_max_new_tokens: int = 32
    torch_threads: int = 0
    torch_interop_threads: int = 2


@dataclass(frozen=True)
class LFM2AudioSettings:
    model_id: str = LFM2_AUDIO_MODEL_ID
    model_dir: Path = Path("models/lfm2-audio/LFM2.5-Audio-1.5B")
    allow_download: bool = False
    max_new_tokens: int = 512
    audio_temperature: float = 1.0
    audio_top_k: int = 4
    torch_threads: int = 0


@dataclass(frozen=True)
class AudioSettings:
    asr_language: str = "auto"
    faster_whisper_model: str = "tiny"
    pasco_model_key: str | None = None
    azure_embedded_asr_locale: str = "zh-CN"
    azure_embedded_grpc_url: str = "127.0.0.1:8792"
    azure_embedded_tts_grpc_url: str = "127.0.0.1:8793"
    azure_embedded_asr_sidecar_url: str = "/api/azure-embedded/asr-ws"
    azure_embedded_asr_zh_cn_model_dir: Path = Path("models/azure-embedded/asr/zh-CN/decrypted/35M")
    azure_embedded_asr_en_gb_model_dir: Path = Path("models/azure-embedded/asr/en-GB/decrypted/v6/35M")
    azure_embedded_tts_voice: str = "azure-embedded-zh-CN-XiaoxiaoNeuralV6"
    azure_embedded_tts_zh_cn_model_dir: Path = Path("models/azure-embedded/tts/zh-CN/XiaoxiaoNeuralV6")
    azure_embedded_tts_en_us_model_dir: Path = Path("models/azure-embedded/tts/en-US/AvaNeuralHDv2")
    whisper_cpp_model_path: Path = Path("models/whisper/base-q5_1.gguf")
    vosk_model_path: Path = Path("models/vosk")
    windows_tts_voice: str | None = None
    edge_tts_voice: str = "zh-CN-XiaoxiaoNeural"
    tts_sample_rate: int = 24000


@dataclass(frozen=True)
class CopilotToolSettings:
    enabled: bool = True
    dry_run: bool = True
    policy_path: Path = Path(".voice-agent/copilot-tools.json")
    audit_log_path: Path = Path(".voice-agent/audit.jsonl")
    workspace_root: Path = Path.cwd()


@dataclass(frozen=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8787
    web_dir: Path = Path("web")


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    providers: ProviderSettings
    foundry: FoundrySettings
    llama_cpp: LlamaCppSettings
    gemma_4_e2b: Gemma4E2BSettings
    lfm2_audio: LFM2AudioSettings
    audio: AudioSettings
    copilot_tools: CopilotToolSettings
    server: ServerSettings

    @classmethod
    def from_env(cls, env: Mapping[str, str], base_dir: Path | None = None) -> "Settings":
        root = (base_dir or Path.cwd()).resolve()
        providers = ProviderSettings(
            vad=_env(env, "VOICE_AGENT_VAD_PROVIDER", "silero"),
            asr=_env(env, "VOICE_AGENT_ASR_PROVIDER", "foundry-local"),
            tts=_env(env, "VOICE_AGENT_TTS_PROVIDER", "azure-embedded"),
            llm=_env(env, "VOICE_AGENT_LLM_PROVIDER", "foundry-local"),
            cloud_fallback_enabled=_env_bool(env, "VOICE_AGENT_CLOUD_FALLBACK_ENABLED", False),
        )
        providers.validate()

        audio = AudioSettings(
            asr_language=_env(env, "VOICE_AGENT_ASR_LANGUAGE", "auto"),
            faster_whisper_model=_env(env, "VOICE_AGENT_FASTER_WHISPER_MODEL", "tiny"),
            pasco_model_key=env.get("PASCO_MODEL_KEY") or None,
            azure_embedded_asr_locale=_env(env, "VOICE_AGENT_AZURE_EMBEDDED_ASR_LOCALE", "zh-CN"),
            azure_embedded_grpc_url=_env(env, "VOICE_AGENT_AZURE_EMBEDDED_GRPC_URL", "127.0.0.1:8792"),
            azure_embedded_tts_grpc_url=_env(
                env,
                "VOICE_AGENT_AZURE_EMBEDDED_TTS_GRPC_URL",
                _env(env, "VOICE_AGENT_AZURE_EMBEDDED_GRPC_URL", "127.0.0.1:8793"),
            ),
            azure_embedded_asr_sidecar_url=_env(
                env,
                "VOICE_AGENT_AZURE_EMBEDDED_ASR_SIDECAR_URL",
                "/api/azure-embedded/asr-ws",
            ),
            azure_embedded_asr_zh_cn_model_dir=_path(
                root,
                _env(
                    env,
                    "VOICE_AGENT_AZURE_EMBEDDED_ASR_ZH_CN_MODEL_DIR",
                    "models/azure-embedded/asr/zh-CN/decrypted/35M",
                ),
            ),
            azure_embedded_asr_en_gb_model_dir=_path(
                root,
                _env(
                    env,
                    "VOICE_AGENT_AZURE_EMBEDDED_ASR_EN_GB_MODEL_DIR",
                    "models/azure-embedded/asr/en-GB/decrypted/v6/35M",
                ),
            ),
            azure_embedded_tts_voice=_env(
                env,
                "VOICE_AGENT_AZURE_EMBEDDED_TTS_VOICE",
                "azure-embedded-zh-CN-XiaoxiaoNeuralV6",
            ),
            azure_embedded_tts_zh_cn_model_dir=_path(
                root,
                _env(
                    env,
                    "VOICE_AGENT_AZURE_EMBEDDED_TTS_ZH_CN_MODEL_DIR",
                    "models/azure-embedded/tts/zh-CN/XiaoxiaoNeuralV6",
                ),
            ),
            azure_embedded_tts_en_us_model_dir=_path(
                root,
                _env(
                    env,
                    "VOICE_AGENT_AZURE_EMBEDDED_TTS_EN_US_MODEL_DIR",
                    "models/azure-embedded/tts/en-US/AvaNeuralHDv2",
                ),
            ),
            whisper_cpp_model_path=_path(
                root,
                _env(env, "VOICE_AGENT_WHISPER_CPP_MODEL_PATH", "models/whisper/base-q5_1.gguf"),
            ),
            vosk_model_path=_path(root, _env(env, "VOICE_AGENT_VOSK_MODEL_PATH", "models/vosk")),
            windows_tts_voice=env.get("VOICE_AGENT_WINDOWS_TTS_VOICE") or None,
            edge_tts_voice=_env(env, "VOICE_AGENT_EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural"),
            tts_sample_rate=_env_int(env, "VOICE_AGENT_TTS_SAMPLE_RATE", 24000),
        )

        return cls(
            base_dir=root,
            providers=providers,
            foundry=FoundrySettings(
                endpoint=_default_foundry_endpoint(env),
                llm_model=_env(env, "VOICE_AGENT_FOUNDRY_LLM_MODEL", "qwen2.5-0.5b-instruct-cuda-gpu:4"),
                asr_model=_env(env, "VOICE_AGENT_FOUNDRY_ASR_MODEL", "nemotron-3.5-asr-streaming-0.6b"),
                timeout_seconds=_env_float(env, "VOICE_AGENT_FOUNDRY_TIMEOUT_SECONDS", 180.0),
            ),
            llama_cpp=LlamaCppSettings(
                endpoint=_env(env, "VOICE_AGENT_LLAMA_CPP_ENDPOINT", "http://127.0.0.1:8080"),
                model=_env(env, "VOICE_AGENT_LLAMA_CPP_MODEL", "gemma-3n-e2b-it"),
                model_path=_path(root, _env(env, "VOICE_AGENT_LLAMA_CPP_MODEL_PATH", "models/llm/gemma-3n-e2b-it/gemma-3n-E2B-it-Q4_K_M.gguf")),
                slot_id=_env_int(env, "VOICE_AGENT_LLAMA_CPP_SLOT_ID", 0),
                timeout_seconds=_env_float(env, "VOICE_AGENT_LLAMA_CPP_TIMEOUT_SECONDS", 180.0),
            ),
            gemma_4_e2b=Gemma4E2BSettings(
                model_id=_env(env, "VOICE_AGENT_GEMMA_4_E2B_MODEL", GEMMA_4_E2B_INSTRUCT_MODEL_ID),
                model_dir=_path(root, _env(env, "VOICE_AGENT_GEMMA_4_E2B_MODEL_DIR", "models/llm/gemma-4-e2b")),
                allow_download=_env_bool(env, "VOICE_AGENT_GEMMA_4_E2B_ALLOW_DOWNLOAD", False),
                max_new_tokens=_env_int(env, "VOICE_AGENT_GEMMA_4_E2B_MAX_NEW_TOKENS", 128),
                audio_turn_max_new_tokens=_env_int(env, "VOICE_AGENT_GEMMA_4_E2B_AUDIO_TURN_MAX_NEW_TOKENS", 32),
                torch_threads=_env_int(env, "VOICE_AGENT_GEMMA_4_E2B_TORCH_THREADS", os.cpu_count() or 1),
                torch_interop_threads=_env_int(env, "VOICE_AGENT_GEMMA_4_E2B_TORCH_INTEROP_THREADS", 2),
            ),
            lfm2_audio=LFM2AudioSettings(
                model_id=_env(env, "VOICE_AGENT_LFM2_AUDIO_MODEL", LFM2_AUDIO_MODEL_ID),
                model_dir=_path(root, _env(env, "VOICE_AGENT_LFM2_AUDIO_MODEL_DIR", "models/lfm2-audio/LFM2.5-Audio-1.5B")),
                allow_download=_env_bool(env, "VOICE_AGENT_LFM2_AUDIO_ALLOW_DOWNLOAD", False),
                max_new_tokens=_env_int(env, "VOICE_AGENT_LFM2_AUDIO_MAX_NEW_TOKENS", 512),
                audio_temperature=_env_float(env, "VOICE_AGENT_LFM2_AUDIO_TEMPERATURE", 1.0),
                audio_top_k=_env_int(env, "VOICE_AGENT_LFM2_AUDIO_TOP_K", 4),
                torch_threads=_env_int(env, "VOICE_AGENT_LFM2_AUDIO_TORCH_THREADS", os.cpu_count() or 1),
            ),
            audio=audio,
            copilot_tools=CopilotToolSettings(
                enabled=_env_bool(env, "VOICE_AGENT_COPILOT_TOOLS_ENABLED", True),
                dry_run=_env_bool(env, "VOICE_AGENT_COPILOT_TOOLS_DRY_RUN", True),
                policy_path=_path(
                    root,
                    _env(env, "VOICE_AGENT_COPILOT_TOOL_POLICY", ".voice-agent/copilot-tools.json"),
                ),
                audit_log_path=_path(
                    root,
                    _env(env, "VOICE_AGENT_COPILOT_AUDIT_LOG", ".voice-agent/audit.jsonl"),
                ),
                workspace_root=root,
            ),
            server=ServerSettings(
                host=_env(env, "VOICE_AGENT_HOST", "127.0.0.1"),
                port=_env_int(env, "VOICE_AGENT_PORT", 8787),
                web_dir=_path(root, _env(env, "VOICE_AGENT_WEB_DIR", "web")),
            ),
        )

    def public_summary(self) -> dict[str, object]:
        gemma_llm_model = self.gemma_4_e2b.model_id
        lfm2_audio_model = self.lfm2_audio.model_id
        lfm2_audio_option = {
            "provider": LFM2_AUDIO_PROVIDER,
            "model": lfm2_audio_model,
            "hfModel": LFM2_AUDIO_MODEL_ID,
            "label": "LiquidAI LFM2.5 Audio 1.5B",
            "runtime": "liquid-audio",
            "modalities": ["audio-in", "text", "audio-out"],
            "requiresAsrProvider": LFM2_AUDIO_PROVIDER,
            "requiresAsrModel": lfm2_audio_model,
            "requiresLlmProvider": LFM2_AUDIO_PROVIDER,
            "requiresLlmModel": lfm2_audio_model,
            "requiresTtsProvider": LFM2_AUDIO_PROVIDER,
            "requiresTtsModel": lfm2_audio_model,
        }
        return {
            "providers": {
                "vad": self.providers.vad,
                "asr": self.providers.asr,
                "tts": self.providers.tts,
                "llm": self.providers.llm,
                "cloudFallbackEnabled": self.providers.cloud_fallback_enabled,
                "asrCapabilities": ASR_PROVIDER_CAPABILITIES,
                "modelOptions": {
                    "asr": [
                        {
                            "provider": GEMMA_4_E2B_ASR_PROVIDER,
                            "model": gemma_llm_model,
                            "hfModel": GEMMA_4_E2B_HF_MODEL_ID,
                            "fullPrecisionModel": GEMMA_4_E2B_FULL_INSTRUCT_MODEL_ID,
                            "label": "Gemma 4 E2B QAT multimodal ASR",
                            "runtime": "hf-transformers",
                            "quantization": "QAT mobile 8-bit",
                            "modalities": ["text", "image", "audio"],
                            "maxAudioSeconds": GEMMA_4_E2B_MAX_AUDIO_SECONDS,
                            "requiresLlmProvider": GEMMA_4_E2B_ASR_PROVIDER,
                            "requiresLlmModel": gemma_llm_model,
                        },
                        lfm2_audio_option,
                    ],
                    "llm": [
                        {
                            "provider": GEMMA_4_E2B_ASR_PROVIDER,
                            "model": gemma_llm_model,
                            "hfModel": GEMMA_4_E2B_HF_MODEL_ID,
                            "fullPrecisionModel": GEMMA_4_E2B_FULL_INSTRUCT_MODEL_ID,
                            "label": "Gemma 4 E2B IT QAT",
                            "runtime": "hf-transformers",
                            "quantization": "QAT mobile 8-bit",
                            "modalities": ["text", "image", "audio"],
                        },
                        lfm2_audio_option,
                    ],
                    "tts": [lfm2_audio_option],
                },
            },
            "lfm2Audio": {
                "provider": LFM2_AUDIO_PROVIDER,
                "model": lfm2_audio_model,
                "hfModel": LFM2_AUDIO_MODEL_ID,
                "preferredRuntime": "liquid-audio",
                "modelDir": str(self.lfm2_audio.model_dir),
                "allowDownload": self.lfm2_audio.allow_download,
                "maxNewTokens": self.lfm2_audio.max_new_tokens,
                "audioTemperature": self.lfm2_audio.audio_temperature,
                "audioTopK": self.lfm2_audio.audio_top_k,
                "torchThreads": self.lfm2_audio.torch_threads,
                "modalities": ["audio-in", "text", "audio-out"],
            },
            "gemma4E2B": {
                "provider": GEMMA_4_E2B_ASR_PROVIDER,
                "hfModel": GEMMA_4_E2B_HF_MODEL_ID,
                "fullPrecisionModel": GEMMA_4_E2B_FULL_INSTRUCT_MODEL_ID,
                "instructModel": gemma_llm_model,
                "preferredRuntime": "hf-transformers",
                "quantization": "QAT mobile 8-bit",
                "modelDir": str(self.gemma_4_e2b.model_dir),
                "allowDownload": self.gemma_4_e2b.allow_download,
                "maxAudioSeconds": GEMMA_4_E2B_MAX_AUDIO_SECONDS,
                "maxNewTokens": self.gemma_4_e2b.max_new_tokens,
                "audioTurnMaxNewTokens": self.gemma_4_e2b.audio_turn_max_new_tokens,
                "torchThreads": self.gemma_4_e2b.torch_threads,
                "torchInteropThreads": self.gemma_4_e2b.torch_interop_threads,
                "modalities": ["text", "image", "audio"],
            },
            "foundry": {
                "endpoint": self.foundry.endpoint,
                "llmModel": self.foundry.llm_model,
                "asrModel": self.foundry.asr_model,
                "streamingAsrModels": FOUNDRY_STREAMING_ASR_MODELS,
            },
            "llamaCpp": {
                "endpoint": self.llama_cpp.endpoint,
                "model": self.llama_cpp.model,
                "modelPath": str(self.llama_cpp.model_path),
                "slotId": self.llama_cpp.slot_id,
            },
            "audio": {
                "asrLanguage": self.audio.asr_language,
                "ttsSampleRate": self.audio.tts_sample_rate,
                "windowsTtsVoice": self.audio.windows_tts_voice,
                "edgeTtsVoice": self.audio.edge_tts_voice,
                "azureEmbeddedGrpcUrl": self.audio.azure_embedded_grpc_url,
                "azureEmbeddedAsr": {
                    "locale": self.audio.azure_embedded_asr_locale,
                    "sidecarUrl": self.audio.azure_embedded_asr_sidecar_url,
                    "zhCnModelDir": str(self.audio.azure_embedded_asr_zh_cn_model_dir),
                    "enGbModelDir": str(self.audio.azure_embedded_asr_en_gb_model_dir),
                    "keyConfigured": self.audio.pasco_model_key is not None,
                },
                "azureEmbeddedTts": {
                    "voice": self.audio.azure_embedded_tts_voice,
                    "zhCnModelDir": str(self.audio.azure_embedded_tts_zh_cn_model_dir),
                    "enUsModelDir": str(self.audio.azure_embedded_tts_en_us_model_dir),
                    "keyConfigured": self.audio.pasco_model_key is not None,
                },
            },
            "copilotTools": {
                "enabled": self.copilot_tools.enabled,
                "dryRun": self.copilot_tools.dry_run,
            },
        }
