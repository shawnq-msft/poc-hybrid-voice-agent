#!/usr/bin/env python3
"""
Decrypt Pasco / embedded SR model files encrypted with VernamEncryptionEngine
and EncryptedFileWriter.

File layout:
    int32_t  headerLength
    bytes    encryptedHeader[headerLength]
    bytes    encryptedContent

Usage:
    decrypt_model.py --key <KEY> <input> <output>
    decrypt_model.py --key <KEY> --in-dir <dir> --out-dir <dir> [--ext .onnx]
    decrypt_model.py --key-prefixed "Key:<KEY>" ...
    decrypt_model.py --key-env PASCO_MODEL_KEY ...
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path


ENCRYPTION_HEADER = b"##Encrypted##"
MIN_KEY_LEN = 60
MAX_KEY_LEN = 181


def xor_bytes(data: bytes, key: bytes) -> bytes:
    key_len = len(key)
    return bytes(byte ^ key[index % key_len] for index, byte in enumerate(data))


def expected_header(key: bytes) -> bytes:
    header = ENCRYPTION_HEADER
    while len(header) < len(key):
        header += header
    return header[: len(key)]


def decrypt_file(in_path: Path, out_path: Path, key: bytes, *, verify: bool = True) -> None:
    raw = in_path.read_bytes()
    if len(raw) < 4:
        raise ValueError(f"{in_path}: too small to be encrypted")

    (header_len,) = struct.unpack("<i", raw[:4])
    if header_len <= 0 or header_len > MAX_KEY_LEN or 4 + header_len > len(raw):
        raise ValueError(f"{in_path}: bad header length {header_len}")
    if header_len != len(key):
        raise ValueError(
            f"{in_path}: header length {header_len} != key length {len(key)} "
            "(wrong key?)"
        )

    enc_header = raw[4 : 4 + header_len]
    body = raw[4 + header_len :]

    if verify:
        decoded = xor_bytes(enc_header, key)
        wanted = expected_header(key)
        if decoded != wanted:
            raise ValueError(
                f"{in_path}: header mismatch; wrong key. "
                f"got {decoded[:16]!r}..., want {wanted[:16]!r}..."
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(xor_bytes(body, key))


def normalize_key(raw_key: str) -> bytes:
    if raw_key.startswith("Key:"):
        raw_key = raw_key[len("Key:") :]
    if not (MIN_KEY_LEN <= len(raw_key) <= MAX_KEY_LEN):
        raise SystemExit(f"key length {len(raw_key)} not in [{MIN_KEY_LEN}, {MAX_KEY_LEN}]")
    return raw_key.encode("ascii")


def looks_encrypted(path: Path, key_len: int) -> bool:
    try:
        with path.open("rb") as file:
            head = file.read(4)
    except OSError:
        return False
    if len(head) < 4:
        return False
    (header_len,) = struct.unpack("<i", head)
    return header_len == key_len


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    key_group = parser.add_mutually_exclusive_group(required=True)
    key_group.add_argument("--key", help="raw key without Key: prefix")
    key_group.add_argument("--key-prefixed", help="key with Key: prefix")
    key_group.add_argument("--key-file", help="file whose stripped contents are the key")
    key_group.add_argument("--key-env", help="environment variable containing the key")

    parser.add_argument("inputs", nargs="*", help="input file(s); paired 1:1 with --output, or one file with positional <out>")
    parser.add_argument("-o", "--output", help="output file, only when decrypting one input")
    parser.add_argument("--in-dir", help="directory of encrypted files")
    parser.add_argument("--out-dir", help="directory for decrypted output")
    parser.add_argument(
        "--ext",
        action="append",
        default=None,
        help="file extension to process in --in-dir. Default: .onnx .config .ini .table .list .txt",
    )
    parser.add_argument("--no-verify", action="store_true", help="skip encrypted header check")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()

    raw_key = args.key or args.key_prefixed
    if args.key_file:
        raw_key = Path(args.key_file).read_text(encoding="utf-8").strip()
    if args.key_env:
        raw_key = os.environ.get(args.key_env)
        if not raw_key:
            raise SystemExit(f"environment variable {args.key_env} is not set")
    key = normalize_key(raw_key or "")

    jobs: list[tuple[Path, Path]] = []
    if args.in_dir:
        if not args.out_dir:
            parser.error("--in-dir requires --out-dir")
        in_root = Path(args.in_dir)
        out_root = Path(args.out_dir)
        exts = {ext.lower() for ext in (args.ext or [".onnx", ".config", ".ini", ".table", ".list", ".txt"])}
        for path in sorted(in_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            if not looks_encrypted(path, len(key)):
                print(f"  skip (not encrypted with this key): {path.relative_to(in_root)}")
                continue
            jobs.append((path, out_root / path.relative_to(in_root)))
    elif args.output and len(args.inputs) == 1:
        jobs.append((Path(args.inputs[0]), Path(args.output)))
    elif len(args.inputs) == 2 and not args.output:
        jobs.append((Path(args.inputs[0]), Path(args.inputs[1])))
    else:
        parser.error("provide either '<in> <out>', '--output <out> <in>', or '--in-dir + --out-dir'")

    print(f"Decrypting {len(jobs)} file(s) with key length {len(key)}...")
    failed = 0
    for src, dst in jobs:
        if dst.exists() and not args.force:
            print(f"  skip (exists, use --force): {dst}")
            continue
        try:
            decrypt_file(src, dst, key, verify=not args.no_verify)
            print(f"  ok: {src} -> {dst} ({dst.stat().st_size} bytes)")
        except Exception as exc:
            failed += 1
            print(f"  FAIL: {src}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())