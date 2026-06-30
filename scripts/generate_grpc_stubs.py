from __future__ import annotations

from pathlib import Path


def main() -> None:
    try:
        from grpc_tools import protoc
    except ImportError as exc:
        raise RuntimeError("Install dev dependencies first: python -m pip install -e .[dev]") from exc

    root = Path(__file__).resolve().parents[1]
    output_dir = root / "src" / "voice_agent" / "providers" / "_generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "__init__.py").touch()
    proto_path = root / "protos" / "azure_embedded_speech.proto"
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{root / 'protos'}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(proto_path),
        ]
    )
    if result != 0:
        raise SystemExit(result)
    grpc_path = output_dir / "azure_embedded_speech_pb2_grpc.py"
    grpc_path.write_text(
        grpc_path.read_text(encoding="utf-8").replace(
            "import azure_embedded_speech_pb2 as azure__embedded__speech__pb2",
            "from . import azure_embedded_speech_pb2 as azure__embedded__speech__pb2",
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
