from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPath:
    kind: str
    name: str
    path: Path
    required: bool = True


def expected_paths(root: Path) -> list[ModelPath]:
    azure_root = root / "azure-embedded"
    return [
        ModelPath("asr", "zh-CN 35M decrypted", azure_root / "asr" / "zh-CN" / "decrypted" / "35M"),
        ModelPath("asr", "en-GB 35M decrypted", azure_root / "asr" / "en-GB" / "decrypted" / "v6" / "35M"),
        ModelPath("tts", "zh-CN XiaoxiaoNeuralV6", azure_root / "tts" / "zh-CN" / "XiaoxiaoNeuralV6"),
        ModelPath("tts", "en-US AvaNeuralHDv2", azure_root / "tts" / "en-US" / "AvaNeuralHDv2"),
    ]


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def format_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def find_obsolete_asset_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    candidates = [
        root / "asr" / "zh-CN" / "encrypted",
        root / "asr" / "en-GB" / "encrypted",
        root / "tts" / "zh-CN" / "XiaoxiaoNeuralHD",
    ]
    return sorted(path for path in candidates if path.is_dir())


def remove_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Azure Embedded model library layout without copying model assets.")
    parser.add_argument("--root", type=Path, default=Path("models"), help="Local ignored model root. Defaults to ./models.")
    parser.add_argument("--delete-obsolete-assets", action="store_true", help="Delete old SDK/model asset directories after reporting them.")
    args = parser.parse_args()

    root = args.root
    print(f"Model root: {root.resolve()}")
    print("Expected Azure Embedded layout:")
    missing_required = False
    for model_path in expected_paths(root):
        exists = model_path.path.exists()
        missing_required = missing_required or (model_path.required and not exists)
        size = format_size(directory_size(model_path.path)) if exists else "missing"
        status = "present" if exists else "missing"
        print(f"- {model_path.kind}: {model_path.name}: {status} ({size}) -> {model_path.path}")

    obsolete_dirs = find_obsolete_asset_dirs(root / "azure-embedded")
    if obsolete_dirs:
        print("Obsolete asset directories:")
        for path in obsolete_dirs:
            print(f"- {path} ({format_size(directory_size(path))})")
        if args.delete_obsolete_assets:
            for path in obsolete_dirs:
                remove_tree(path)
                print(f"deleted: {path}")
        else:
            print("Run again with --delete-obsolete-assets to remove these directories.")
    else:
        print("Obsolete asset directories: none")

    if missing_required:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
