from __future__ import annotations

try:
    from voice_agent.providers._generated.azure_embedded_speech_pb2_grpc import *  # type: ignore[F403]
except ImportError as exc:  # pragma: no cover - exercised only when generated stubs are missing.
    raise RuntimeError(
        "Azure Embedded gRPC Python stubs are not generated. Install grpcio-tools and run "
        "`python -m grpc_tools.protoc -I protos --python_out=src/voice_agent/providers/_generated "
        "--grpc_python_out=src/voice_agent/providers/_generated protos/azure_embedded_speech.proto`."
    ) from exc
