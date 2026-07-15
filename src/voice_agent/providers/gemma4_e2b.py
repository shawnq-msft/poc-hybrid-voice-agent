from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from voice_agent.config import GEMMA_4_E2B_MAX_AUDIO_SECONDS, Gemma4E2BSettings
from voice_agent.providers.llm_foundry import ChatMessage


@dataclass
class PreparedGemma4E2BTurn:
    llm: "Gemma4E2BLLM"
    messages: list[ChatMessage]

    def with_user_text(self, user_text: str) -> "PreparedGemma4E2BTurn":
        return PreparedGemma4E2BTurn(self.llm, [*self.messages, ChatMessage("user", user_text)])

    async def complete(self) -> str:
        return await self.llm.complete(self.messages)


@dataclass(frozen=True)
class Gemma4E2BAudioTurnResult:
    transcript_text: str
    assistant_text: str


class Gemma4E2BFusedTurnIncomplete(RuntimeError):
    def __init__(self, transcript_text: str, raw_response: str):
        self.transcript_text = transcript_text
        self.raw_response = raw_response
        super().__init__(f"Gemma 4 E2B fused turn returned transcription only. Raw response: {raw_response[:500]}")


@dataclass(frozen=True)
class Gemma4E2BLLM:
    settings: Gemma4E2BSettings

    async def prepare_turn(self, messages: Iterable[ChatMessage]) -> PreparedGemma4E2BTurn:
        return PreparedGemma4E2BTurn(self, list(messages))

    async def complete(self, messages: Iterable[ChatMessage]) -> str:
        return await asyncio.to_thread(_complete_messages, self.settings, list(messages))

    async def complete_audio_turn(
        self,
        audio_bytes: bytes,
        filename: str,
        language: str,
        messages: Iterable[ChatMessage],
    ) -> Gemma4E2BAudioTurnResult:
        return await asyncio.to_thread(_complete_audio_turn, self.settings, audio_bytes, filename, language, list(messages))


@dataclass(frozen=True)
class Gemma4E2BASR:
    settings: Gemma4E2BSettings

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        media_type: str,
        language: str = "auto",
    ):
        from voice_agent.providers.asr import ASRTranscript

        if not audio_bytes:
            raise RuntimeError("Cannot transcribe empty audio")
        text = await asyncio.to_thread(_transcribe_audio, self.settings, audio_bytes, filename, language)
        return ASRTranscript(text=text, language=language, provider="gemma-4-e2b")


def check_gemma_4_e2b_ready(settings: Gemma4E2BSettings) -> dict[str, object]:
    dependency = _check_transformers_dependency(settings.model_id)
    torch_runtime = _check_torch_runtime(settings)
    local_snapshot = _has_local_snapshot(settings.model_dir)
    ready = dependency.get("ready") is True and torch_runtime.get("ready") is True and (local_snapshot or settings.allow_download)
    dependency_details = {key: value for key, value in dependency.items() if key not in {"ready", "error"}}
    runtime_details = {key: value for key, value in torch_runtime.items() if key not in {"ready", "error"}}
    return {
        "ready": ready,
        "provider": "gemma-4-e2b",
        "runtime": "hf-transformers",
        "model": settings.model_id,
        "modelDir": str(settings.model_dir),
        "localSnapshot": local_snapshot,
        "allowDownload": settings.allow_download,
        "maxAudioSeconds": GEMMA_4_E2B_MAX_AUDIO_SECONDS,
        **dependency_details,
        **runtime_details,
        **({} if ready else {"error": _gemma_ready_error(settings, dependency, torch_runtime, local_snapshot)}),
    }


def warm_gemma_4_e2b(settings: Gemma4E2BSettings) -> dict[str, object]:
    ready = check_gemma_4_e2b_ready(settings)
    if not ready.get("ready"):
        raise RuntimeError(str(ready.get("error") or "Gemma 4 E2B is not ready"))
    _load_components(settings)
    return {key: value for key, value in ready.items() if key != "error"}


def _gemma_ready_error(settings: Gemma4E2BSettings, dependency: dict[str, object], torch_runtime: dict[str, object], local_snapshot: bool) -> str:
    if not dependency.get("ready"):
        return str(dependency.get("error") or "Transformers cannot load Gemma 4 E2B")
    if not torch_runtime.get("ready"):
        return str(torch_runtime.get("error") or "PyTorch cannot load Gemma 4 E2B")
    if not local_snapshot and not settings.allow_download:
        return (
            f"Gemma 4 E2B snapshot is not present at {settings.model_dir}. "
            "Run `py -3.11 scripts\\gemma4_e2b_smoke.py --download` or set VOICE_AGENT_GEMMA_4_E2B_ALLOW_DOWNLOAD=1."
        )
    return "Gemma 4 E2B is not ready"


def _check_transformers_dependency(model_id: str) -> dict[str, object]:
    try:
        import transformers
        from transformers import AutoConfig
    except ImportError as exc:
        return {"ready": False, "error": f"Install transformers to use Gemma 4 E2B: {exc}"}
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        return {"ready": False, "transformersVersion": transformers.__version__, "error": f"Transformers cannot load {model_id}: {exc}"}
    return {
        "ready": True,
        "transformersVersion": transformers.__version__,
        "modelType": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
    }


def _check_torch_runtime(settings: Gemma4E2BSettings) -> dict[str, object]:
    try:
        torch, configured = _configure_torch_runtime(settings)
    except ImportError as exc:
        return {"ready": False, "error": f"Install torch to use Gemma 4 E2B: {exc}"}
    cuda_available = bool(torch.cuda.is_available())
    details: dict[str, object] = {
        "ready": True,
        "torchVersion": torch.__version__,
        "torchCudaAvailable": cuda_available,
        "torchCudaVersion": getattr(torch.version, "cuda", None),
        "device": "cuda" if cuda_available else "cpu",
        "torchThreads": torch.get_num_threads(),
        "torchInteropThreads": torch.get_num_interop_threads(),
        "torchThreadConfigApplied": configured,
    }
    if cuda_available:
        details["torchCudaDeviceCount"] = torch.cuda.device_count()
        try:
            details["torchCudaDeviceName"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    else:
        details["performanceWarning"] = "Gemma 4 E2B is running on CPU-only PyTorch; expect high TTFT and non-real-time audio turns."
    return details


def _has_local_snapshot(model_dir: Path) -> bool:
    return (model_dir / "config.json").exists() and (model_dir / "model.safetensors").exists()


def _model_source(settings: Gemma4E2BSettings) -> str:
    if _has_local_snapshot(settings.model_dir):
        return str(settings.model_dir)
    if settings.allow_download:
        return settings.model_id
    raise RuntimeError(_gemma_ready_error(settings, _check_transformers_dependency(settings.model_id), _check_torch_runtime(settings), False))


def _configure_torch_runtime(settings: Gemma4E2BSettings):
    threads = max(1, int(settings.torch_threads or os.cpu_count() or 1))
    interop_threads = max(1, int(settings.torch_interop_threads or 1))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(threads))
    import torch

    configured = True
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        configured = False
    return torch, configured


@lru_cache(maxsize=1)
def _load_components_cached(model_source: str, max_new_tokens: int, torch_threads: int, torch_interop_threads: int):
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_source)
    model_kwargs: dict[str, object] = {"dtype": "auto"}
    if importlib.util.find_spec("accelerate") is not None:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    model.eval()
    return processor, model


def _load_components(settings: Gemma4E2BSettings):
    _configure_torch_runtime(settings)
    return _load_components_cached(_model_source(settings), settings.max_new_tokens, settings.torch_threads, settings.torch_interop_threads)


def _complete_messages(settings: Gemma4E2BSettings, messages: list[ChatMessage]) -> str:
    processor, model = _load_components(settings)
    prompt = processor.apply_chat_template(
        [{"role": message.role, "content": message.content} for message in messages],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=prompt, return_tensors="pt")
    inputs = _move_inputs_to_model(inputs, model)
    input_len = inputs["input_ids"].shape[-1]
    outputs = _generate(model, inputs, settings.max_new_tokens)
    response = _decode_generated_response(processor, outputs, input_len)
    return _clean_text_response(response)


def _transcribe_audio(settings: Gemma4E2BSettings, audio_bytes: bytes, filename: str, language: str) -> str:
    processor, model = _load_components(settings)
    audio = _decode_audio_to_float32(audio_bytes, filename, 16000)
    prompt_language = "the detected language" if language in {"", "auto"} else language
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _asr_prompt(prompt_language)},
                {"type": "audio"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=prompt, audio=audio, sampling_rate=16000, return_tensors="pt")
    inputs = _move_inputs_to_model(inputs, model)
    input_len = inputs["input_ids"].shape[-1]
    outputs = _generate(model, inputs, settings.audio_turn_max_new_tokens)
    response = _decode_generated_response(processor, outputs, input_len)
    text = _clean_text_response(response)
    if not text:
        raise RuntimeError("Gemma 4 E2B ASR returned empty text")
    return text


def _complete_audio_turn(settings: Gemma4E2BSettings, audio_bytes: bytes, filename: str, language: str, messages: list[ChatMessage]) -> Gemma4E2BAudioTurnResult:
    processor, model = _load_components(settings)
    audio = _decode_audio_to_float32(audio_bytes, filename, 16000)
    prompt_language = "the detected language" if language in {"", "auto"} else language
    system_messages = [{"role": message.role, "content": message.content} for message in messages]
    turn_instruction = (
        f"The audio is the user's spoken request in {prompt_language}. Do not only transcribe. "
        "Output exactly two short lines:\n"
        "User: transcription\n"
        "Assistant: brief answer"
    )
    prompt_messages = [
        *system_messages,
        {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": turn_instruction}]},
    ]
    prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=prompt, audio=audio, sampling_rate=16000, return_tensors="pt")
    inputs = _move_inputs_to_model(inputs, model)
    input_len = inputs["input_ids"].shape[-1]
    outputs = _generate(model, inputs, settings.audio_turn_max_new_tokens)
    response = _decode_generated_response(processor, outputs, input_len)
    transcript, assistant = _parse_audio_turn_response(response)
    cleaned_response = _clean_text_response(response)
    if not transcript and cleaned_response:
        raise Gemma4E2BFusedTurnIncomplete(cleaned_response, cleaned_response)
    if transcript and not assistant:
        raise Gemma4E2BFusedTurnIncomplete(transcript, cleaned_response)
    if not transcript:
        raise RuntimeError(f"Gemma 4 E2B fused turn returned empty transcript. Raw response: {cleaned_response[:500]}")
    if not assistant:
        raise RuntimeError(f"Gemma 4 E2B fused turn returned empty assistant response. Raw response: {cleaned_response[:500]}")
    return Gemma4E2BAudioTurnResult(transcript_text=transcript, assistant_text=assistant)


def _decode_generated_response(processor, outputs, input_len: int) -> str:
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    if not str(response or "").strip():
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    if hasattr(processor, "parse_response"):
        parsed = processor.parse_response(response)
        if isinstance(parsed, str):
            return parsed
        if isinstance(parsed, dict):
            return str(parsed.get("content") or parsed.get("response") or parsed.get("text") or response)
    return response


def _generate(model, inputs, max_new_tokens: int):
    import torch

    with torch.inference_mode():
        return model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)


def _move_inputs_to_model(inputs, model):
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            return inputs
    return inputs.to(device)


def _decode_audio_to_float32(audio_bytes: bytes, filename: str, sampling_rate: int):
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install PyAV and numpy to decode audio for Gemma 4 E2B ASR") from exc

    suffix = Path(filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(audio_bytes)
        temp_path = Path(handle.name)
    try:
        chunks = []
        with av.open(str(temp_path)) as container:
            resampler = av.AudioResampler(format="flt", layout="mono", rate=sampling_rate)
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1).astype("float32"))
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().reshape(-1).astype("float32"))
    finally:
        temp_path.unlink(missing_ok=True)
    if not chunks:
        raise RuntimeError("No audio frames decoded for Gemma 4 E2B ASR")
    audio = np.concatenate(chunks)
    max_samples = sampling_rate * GEMMA_4_E2B_MAX_AUDIO_SECONDS
    if audio.shape[0] > max_samples:
        raise RuntimeError(f"Gemma 4 E2B audio input must be at most {GEMMA_4_E2B_MAX_AUDIO_SECONDS} seconds")
    return audio


def _asr_prompt(language: str) -> str:
    return (
        f"Transcribe the following speech segment in {language} into {language} text.\n\n"
        "Follow these specific instructions for formatting the answer:\n"
        "* Only output the transcription, with no newlines.\n"
        "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."
    )


def _clean_text_response(response: object) -> str:
    text = str(response or "").strip()
    markers = ["<|channel>final\n", "<channel|>", "<start_of_turn>model", "<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>", "<turn|>"]
    for marker in markers:
        text = text.replace(marker, "")
    return " ".join(part.strip() for part in text.splitlines() if part.strip()).strip()


def _parse_audio_turn_response(response: object) -> tuple[str, str]:
    text = _clean_text_response(response)
    conversation_match = re.search(r"(?:^|\s)User:\s*(.*?)(?:\s+Assistant:\s*(.*))$", text, flags=re.IGNORECASE | re.DOTALL)
    if conversation_match:
        return _clean_text_response(conversation_match.group(1)), _clean_text_response(conversation_match.group(2))
    line_match = re.search(r"(?:^|\s)T:\s*(.*?)(?:\s+A:\s*(.*))$", text, flags=re.IGNORECASE | re.DOTALL)
    if line_match:
        return _clean_text_response(line_match.group(1)), _clean_text_response(line_match.group(2))
    transcript_match = re.search(r"<transcript>(.*?)</transcript>", text, flags=re.IGNORECASE | re.DOTALL)
    response_match = re.search(r"<response>(.*?)</response>", text, flags=re.IGNORECASE | re.DOTALL)
    if transcript_match and response_match:
        return _clean_text_response(transcript_match.group(1)), _clean_text_response(response_match.group(1))
    if "<response>" in text.lower():
        before, _, after = re.split(r"<response>", text, maxsplit=1, flags=re.IGNORECASE)
        return _clean_text_response(before.replace("<transcript>", "")), _clean_text_response(after.replace("</response>", ""))
    return "", text
