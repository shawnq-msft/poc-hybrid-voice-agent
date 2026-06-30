from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from voice_agent.config import AudioSettings


@dataclass(frozen=True)
class WindowsVoice:
    id: str
    display_name: str
    language: str | None = None


def enumerate_windows_voices() -> list[WindowsVoice]:
    voices = _enumerate_winrt_voices()
    if voices:
        return voices
    return _enumerate_sapi_voices()


def _enumerate_winrt_voices() -> list[WindowsVoice]:
    try:
        from winrt.windows.media.speechsynthesis import SpeechSynthesizer

        return [
            WindowsVoice(id=voice.id, display_name=voice.display_name, language=voice.language)
            for voice in SpeechSynthesizer.all_voices
        ]
    except Exception:
        return []


def _enumerate_sapi_voices() -> list[WindowsVoice]:
    try:
        import pyttsx3

        engine = pyttsx3.init()
        return [
            WindowsVoice(
                id=getattr(voice, "id", ""),
                display_name=getattr(voice, "name", getattr(voice, "id", "SAPI voice")),
                language=_first_language(getattr(voice, "languages", None)),
            )
            for voice in engine.getProperty("voices")
        ]
    except Exception:
        return []


def _first_language(languages: object) -> str | None:
    if isinstance(languages, (list, tuple)) and languages:
        first = languages[0]
        if isinstance(first, bytes):
            return first.decode(errors="ignore")
        return str(first)
    return None


@dataclass(frozen=True)
class WindowsTTSPlan:
    provider: str
    requested_voice: str | None
    sample_rate: int
    fallback_provider: str | None = "windows-sapi"


def plan_windows_tts(provider: str, requested_voice: str | None, sample_rate: int) -> WindowsTTSPlan:
    fallback = "windows-sapi" if provider == "windows-winrt" else None
    return WindowsTTSPlan(
        provider=provider,
        requested_voice=requested_voice,
        sample_rate=sample_rate,
        fallback_provider=fallback,
    )


def synthesize_sapi_wav(text: str, voice: str | None = None) -> bytes:
    if not text.strip():
        raise RuntimeError("Cannot synthesize empty text")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        text_path = temp_path / "text.txt"
        wav_path = temp_path / "speech.wav"
        text_path.write_text(text, encoding="utf-8")

        voice_block = ""
        if voice:
            voice_block = f"$synth.SelectVoice('{_escape_powershell_single_quoted(voice)}');"

        command = (
            "Add-Type -AssemblyName System.Speech;"
            f"$text = Get-Content -Raw -LiteralPath '{_escape_powershell_single_quoted(str(text_path))}';"
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            f"{voice_block}"
            f"$synth.SetOutputToWaveFile('{_escape_powershell_single_quoted(str(wav_path))}');"
            "$synth.Speak($text);"
            "$synth.Dispose();"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0 or not wav_path.exists():
            detail = completed.stderr.strip() or completed.stdout.strip() or "PowerShell TTS failed"
            raise RuntimeError(detail)
        return wav_path.read_bytes()


def _escape_powershell_single_quoted(value: str) -> str:
    return value.replace("'", "''")


async def synthesize_edge_mp3(text: str, voice: str = "zh-CN-XiaoxiaoNeural") -> bytes:
    if not text.strip():
        raise RuntimeError("Cannot synthesize empty text")
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("Install edge-tts to use Edge neural TTS") from exc

    with tempfile.TemporaryDirectory() as temp_dir:
        mp3_path = Path(temp_dir) / "speech.mp3"
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(str(mp3_path))
        if not mp3_path.exists():
            raise RuntimeError("Edge TTS did not produce audio")
        return mp3_path.read_bytes()


def synthesize_azure_embedded_wav(text: str, audio_settings: AudioSettings) -> bytes:
    if not text.strip():
        raise RuntimeError("Cannot synthesize empty text")
    try:
        import grpc
        from voice_agent.providers import azure_embedded_pb2 as pb
        from voice_agent.providers import azure_embedded_pb2_grpc as pb_grpc
    except ImportError as exc:
        raise RuntimeError("Install grpcio and generate Azure Embedded gRPC stubs to use azure-embedded TTS") from exc

    voice = audio_settings.azure_embedded_tts_voice
    locale = "en-US" if "en-US" in voice else "zh-CN"
    channel = grpc.insecure_channel(audio_settings.azure_embedded_grpc_url)
    stub = pb_grpc.AzureEmbeddedSpeechStub(channel)
    try:
        response = stub.Synthesize(
            pb.TtsRequest(
                voice=voice,
                locale=locale,
                text=text,
                sample_rate_hz=audio_settings.tts_sample_rate,
            ),
            timeout=30,
        )
    finally:
        channel.close()
    if not response.audio:
        raise RuntimeError("Azure Embedded TTS returned empty audio")
    return bytes(response.audio)
