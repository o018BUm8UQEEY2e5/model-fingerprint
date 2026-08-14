#!/usr/bin/env python3
"""Fingerprint GGUF and safetensors model files.

A model "fingerprint" is a stable, content-derived identifier for the
weights. It is robust to renames, re-packaging and metadata edits (chat
template, file name, etc.) but changes if the actual tensor data changes.

Approach
--------
1. Detect the file format and parse its header:
   - GGUF: header magic, version, metadata KV store, tensor info table.
   - safetensors: u64 header length + JSON header (dtype/shape/data_offsets).
2. For every tensor we hash (dimensions, dtype, raw bytes) and feed the
   per-tensor digests into a final Merkle-style hash. That final digest is the
   model fingerprint. Digests are sorted first so the result is independent of
   tensor ordering within the file. Tensor names are ignored by default so that
   differently-named but otherwise identical weights (e.g. Hugging Face vs
   GGUF naming) hash the same; pass --with-names for a stricter identity.
3. A secondary "metadata fingerprint" is computed over the parsed metadata KV
   store, useful for detecting config/tokenizer changes without re-hashing
   gigabytes of weights.

Tensor byte ranges for GGUF are derived from llama.cpp's per-type block
layouts, so hashing captures exactly the model weights (not file padding) and
is independent of tensor ordering within the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__version__ = "0.1.0"

# --------------------------------------------------------------------------
# GGUF constants
# --------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian (bytes b"GGUF")

# Metadata value types
MVT_UINT8 = 0
MVT_INT8 = 1
MVT_UINT16 = 2
MVT_INT16 = 3
MVT_UINT32 = 4
MVT_INT32 = 5
MVT_FLOAT32 = 6
MVT_BOOL = 7
MVT_STRING = 8
MVT_ARRAY = 9

_MVT_SCALAR = {
    MVT_UINT8: ("B", 1),
    MVT_INT8: ("b", 1),
    MVT_UINT16: ("H", 2),
    MVT_INT16: ("h", 2),
    MVT_UINT32: ("I", 4),
    MVT_INT32: ("i", 4),
    MVT_FLOAT32: ("f", 4),
    MVT_BOOL: ("B", 1),
}

# GGML type index -> human readable name. Only the commonly used ones are
# named; unknown indices fall back to "type_<n>".
GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 31: "Q4_K_S", 32: "Q2_K_S", 34: "TQ1_0", 35: "TQ2_0",
}

GGUF_FILE_TYPES: dict[int, str] = {
    0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1",
    7: "MOSTLY_Q8_0", 8: "MOSTLY_Q5_0", 9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K",
    11: "MOSTLY_Q3_K_S", 12: "MOSTLY_Q3_K_M", 13: "MOSTLY_Q3_K_L",
    14: "MOSTLY_Q4_K_S", 15: "MOSTLY_Q4_K_M", 16: "MOSTLY_Q5_K_S",
    17: "MOSTLY_Q5_K_M", 18: "MOSTLY_Q6_K", 19: "MOSTLY_IQ2_XXS",
    20: "MOSTLY_IQ2_XS", 21: "MOSTLY_Q2_K_S", 22: "MOSTLY_IQ3_XS",
    23: "MOSTLY_IQ3_XXS", 24: "MOSTLY_IQ1_S", 25: "MOSTLY_IQ4_NL",
    26: "MOSTLY_IQ3_S", 27: "MOSTLY_IQ3_M", 28: "MOSTLY_IQ2_S",
    29: "MOSTLY_IQ2_M", 30: "MOSTLY_IQ4_XS", 31: "MOSTLY_IQ1_M",
    32: "MOSTLY_BF16",
}


def ggml_type_name(idx: int) -> str:
    return GGML_TYPE_NAMES.get(idx, f"type_{idx}")


# ggml type index -> (block_size, bytes_per_block). Used to compute the exact
# unpadded byte size of a tensor (nblocks * bytes_per_block), so fingerprinting
# ignores inter-tensor alignment padding and is independent of block ordering
# and repacking. Values follow llama.cpp ggml (GGML_TYPE_SIZE/BLCK_SIZE).
# Types not listed (rare/new: IQ1_M, TQ1_0, TQ2_0, ...) fall back to the
# offset-derived size, which may still include alignment padding.
GGML_TYPE_LAYOUTS: dict[int, tuple[int, int]] = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
    6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
    10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
    14: (256, 210), 15: (256, 258),
    16: (256, 66), 17: (256, 74), 18: (256, 74), 19: (256, 50),
    20: (32, 18), 21: (256, 90), 22: (256, 82), 23: (256, 138),
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    30: (1, 2), 31: (256, 144), 32: (256, 84),
}


def ggml_nbytes(ggml_type: int, dims: list[int]) -> int | None:
    """Exact unpadded byte size, or None for unknown type layouts."""
    layout = GGML_TYPE_LAYOUTS.get(ggml_type)
    if layout is None:
        return None
    blck, per = layout
    ne = 1
    for d in dims:
        ne *= d
    nblocks = (ne + blck - 1) // blck
    return nblocks * per


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    dtype: str
    offset: int  # relative to start of tensor data section
    nbytes: int = 0
    ggml_type: int | None = None  # set only for GGUF; used for exact sizing


@dataclass
class Model:
    format: str  # "gguf" or "safetensors"
    tensors: list[TensorInfo]
    metadata: dict[str, Any]
    data_start: int
    file_size: int
    path: Path | None = None
    version: int | None = None
    alignment: int = 0


class ModelError(Exception):
    pass


# --------------------------------------------------------------------------
# File reading helpers
# --------------------------------------------------------------------------

class FileReader:
    """Common low-level file reading with offset tracking."""

    def __init__(self, path: Path):
        self.path = path
        self.file_size = path.stat().st_size
        self.f = open(path, "rb")
        self.pos = 0

    def close(self) -> None:
        self.f.close()

    def __enter__(self) -> "FileReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _read(self, n: int) -> bytes:
        data = self.f.read(n)
        if len(data) != n:
            raise ModelError(f"unexpected EOF at offset {self.pos} (wanted {n} bytes)")
        self.pos += n
        return data

    def _read_u32(self) -> int:
        return struct.unpack("<I", self._read(4))[0]

    def _read_u64(self) -> int:
        return struct.unpack("<Q", self._read(8))[0]

    def _read_string(self, version: int) -> str:
        n = self._read_u32() if version == 1 else self._read_u64()
        raw = self._read(n)
        return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# GGUF parsing
# --------------------------------------------------------------------------

def _read_value(r: FileReader, version: int, vtype: int) -> Any:
    if vtype in _MVT_SCALAR:
        fmt, size = _MVT_SCALAR[vtype]
        return struct.unpack("<" + fmt, r._read(size))[0]
    if vtype == MVT_STRING:
        return r._read_string(version)
    if vtype == MVT_ARRAY:
        sub = r._read_u32()
        count = r._read_u64()
        return _read_array(r, version, sub, count)
    raise ModelError(f"unknown metadata value type {vtype}")


def _read_array(r: FileReader, version: int, sub: int, count: int) -> Any:
    if count == 0:
        return []
    if sub in _MVT_SCALAR:
        fmt, size = _MVT_SCALAR[sub]
        raw = r._read(size * count)
        return list(struct.unpack(f"<{count}{fmt}", raw))
    if sub == MVT_STRING:
        return [r._read_string(version) for _ in range(count)]
    if sub == MVT_ARRAY:
        raise ModelError("nested arrays are not supported")
    raise ModelError(f"unknown array element type {sub}")


def parse_gguf(r: FileReader, path: Path) -> Model:
    magic = r._read_u32()
    if magic != GGUF_MAGIC:
        raise ModelError("not a GGUF file (bad magic)")
    version = r._read_u32()
    if version not in (1, 2, 3):
        raise ModelError(f"unsupported GGUF version {version}")
    tensor_count = r._read_u64()
    kv_count = r._read_u64()

    metadata: dict[str, Any] = {}
    alignment = 32

    for _ in range(kv_count):
        key = r._read_string(version)
        vtype = r._read_u32()
        metadata[key] = _read_value(r, version, vtype)
        if key == "general.alignment":
            try:
                alignment = int(metadata[key])
            except (TypeError, ValueError):
                pass

    # Header offset width changed between V1 (u32) and V2/V3 (u64).
    offset_fmt = "<I" if version == 1 else "<Q"

    tensors: list[TensorInfo] = []
    for _ in range(tensor_count):
        name = r._read_string(version)
        n_dims = r._read_u32()
        dims = [r._read_u64() for _ in range(n_dims)]
        ggml_type = r._read_u32()
        offset = struct.unpack(offset_fmt, r._read(struct.calcsize(offset_fmt)))[0]
        tensors.append(TensorInfo(name=name, dims=dims,
                                  dtype=ggml_type_name(ggml_type), offset=offset,
                                  ggml_type=ggml_type))

    # Tensor data starts at the aligned header end.
    header_end = r.pos
    if alignment <= 0:
        raise ModelError("invalid general.alignment value")
    data_start = (header_end + alignment - 1) // alignment * alignment

    # Derive each tensor's byte length. Prefer the exact size computed from the
    # type layout (excluding inter-tensor alignment padding) so the fingerprint
    # is independent of block order and repacking; fall back to the distance to
    # the next tensor's offset when the type layout is unknown.
    tensors.sort(key=lambda t: t.offset)
    data_end = r.file_size - data_start
    for i, t in enumerate(tensors):
        nxt = tensors[i + 1].offset if i + 1 < len(tensors) else data_end
        if nxt < t.offset:
            raise ModelError("tensor offsets are not contiguous/sorted")
        avail = nxt - t.offset
        exact = ggml_nbytes(t.ggml_type, t.dims) if t.ggml_type is not None else None
        if exact is not None and 0 < exact <= avail and avail - exact < alignment:
            t.nbytes = exact
        else:
            t.nbytes = avail

    return Model(format="gguf", tensors=tensors, metadata=metadata,
                 data_start=data_start, file_size=r.file_size, path=path,
                 version=version, alignment=alignment)


# --------------------------------------------------------------------------
# safetensors parsing
# --------------------------------------------------------------------------

def parse_safetensors(r: FileReader, path: Path) -> Model:
    if r.file_size < 8:
        raise ModelError("file too small to be safetensors")
    n = r._read_u64()
    if n > r.file_size - 8:
        raise ModelError("safetensors header length exceeds file size")
    raw = r._read(n)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelError(f"invalid safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise ModelError("safetensors header is not a JSON object")

    metadata = header.get("__metadata__") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    data_start = r.pos  # right after 8-byte length + header
    tensors: list[TensorInfo] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise ModelError(f"invalid tensor info for {name}")
        try:
            dtype = str(info["dtype"])
            shape = [int(d) for d in info["shape"]]
            offsets = info["data_offsets"]
            begin, end = int(offsets[0]), int(offsets[1])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ModelError(f"invalid tensor info for {name}: {exc}") from exc
        if begin < 0 or end < begin:
            raise ModelError(f"invalid data_offsets for tensor {name}")
        if data_start + end > r.file_size:
            raise ModelError(f"tensor {name} extends past end of file")
        tensors.append(TensorInfo(name=name, dims=shape, dtype=dtype,
                                  offset=begin, nbytes=end - begin))

    tensors.sort(key=lambda t: t.name)
    return Model(format="safetensors", tensors=tensors, metadata=metadata,
                 data_start=data_start, file_size=r.file_size, path=path)


def parse_detect(r: FileReader, path: Path) -> Model:
    """Peek the first bytes and dispatch to the right parser."""
    head = r.f.read(4)
    r.f.seek(0)
    r.pos = 0
    if head == b"GGUF":
        return parse_gguf(r, path)
    return parse_safetensors(r, path)


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------

def _tensor_digest(t: TensorInfo, reader: FileReader, data_start: int,
                   chunk: int, include_names: bool) -> bytes:
    h = hashlib.sha256()
    if include_names:
        h.update(t.name.encode("utf-8"))
    h.update(struct.pack("<" + "Q" * len(t.dims), *t.dims))
    h.update(t.dtype.encode("utf-8"))
    reader.f.seek(data_start + t.offset)
    remaining = t.nbytes
    while remaining > 0:
        n_read = min(chunk, remaining)
        block = reader.f.read(n_read)
        if not block:
            raise ModelError(
                f"unexpected EOF reading tensor {t.name!r} "
                f"({remaining} bytes short)")
        h.update(block)
        remaining -= len(block)
    return h.digest()


def fingerprint(model: Model, reader: FileReader, *,
                include_names: bool = False, chunk: int = 1 << 20) -> str:
    """Content fingerprint: Merkle hash over every tensor's raw bytes."""
    per_tensor = [_tensor_digest(t, reader, model.data_start, chunk,
                                 include_names)
                  for t in model.tensors]
    per_tensor.sort()
    final = hashlib.sha256()
    for d in per_tensor:
        final.update(d)
    return final.hexdigest()


def metadata_fingerprint(model: Model) -> str:
    """Fingerprint over the parsed metadata KV store only."""
    h = hashlib.sha256()
    for key in sorted(model.metadata):
        h.update(key.encode("utf-8"))
        payload = json.dumps(model.metadata[key], sort_keys=True,
                             separators=(",", ":"), default=str).encode("utf-8")
        h.update(struct.pack("<Q", len(payload)))
        h.update(payload)
    return h.hexdigest()


def total_parameters(model: Model) -> int:
    total = 0
    for t in model.tensors:
        n = 1
        for d in t.dims:
            n *= d
        total += n
    return total


def dtype_profile(model: Model) -> dict[str, int]:
    profile: dict[str, int] = {}
    for t in model.tensors:
        profile[t.dtype] = profile.get(t.dtype, 0) + 1
    return profile


def resolve_vocab_size(md: dict[str, Any]) -> int | None:
    arch = md.get("general.architecture")
    if arch and f"{arch}.vocab_size" in md:
        try:
            return int(md[f"{arch}.vocab_size"])
        except (TypeError, ValueError):
            pass
    tokens = md.get("tokenizer.ggml.tokens")
    if isinstance(tokens, list):
        return len(tokens)
    return None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_summary(model: Model, content_fp: str,
                  meta_fp: str | None = None) -> dict[str, Any]:
    md = model.metadata
    if model.format == "gguf":
        ft_enum = md.get("general.file_type")
        quant_label = (GGUF_FILE_TYPES.get(ft_enum, f"type_{ft_enum}")
                       if isinstance(ft_enum, int) else None)
        vocab_size = resolve_vocab_size(md)
    else:
        quant_label = None
        vocab_size = None

    summary = {
        "format": model.format,
        "file": str(model.path) if model.path else "",
        "gguf_version": model.version,
        "file_size_bytes": model.file_size,
        "tensor_count": len(model.tensors),
        "total_parameters": total_parameters(model),
        "architecture": md.get("general.architecture"),
        "name": md.get("general.name") or md.get("_name_or_path"),
        "quantization": quant_label,
        "dtype_profile": dtype_profile(model),
        "vocab_size": vocab_size,
        "content_fingerprint": content_fp,
    }
    if meta_fp is not None:
        summary["metadata_fingerprint"] = meta_fp
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fingerprint a GGUF or safetensors model file.")
    ap.add_argument("model", type=Path, help="path to .gguf or .safetensors file")
    ap.add_argument("--metadata-only", action="store_true",
                    help="skip tensor hashing; fingerprint metadata only")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--with-names", action="store_true",
                    help="include tensor names in the content fingerprint")
    args = ap.parse_args(argv)

    if not args.model.is_file():
        print(f"error: file not found: {args.model}", file=sys.stderr)
        return 2

    try:
        with FileReader(args.model) as reader:
            model = parse_detect(reader, args.model)
            if args.metadata_only:
                content_fp = "(skipped)"
            else:
                content_fp = fingerprint(model, reader,
                                         include_names=args.with_names)
            meta_fp = metadata_fingerprint(model)
    except ModelError as err:
        print(f"error: failed to parse {args.model}: {err}", file=sys.stderr)
        return 1
    except OSError as err:
        print(f"error: cannot read {args.model}: {err}", file=sys.stderr)
        return 1

    summary = build_summary(model, content_fp, meta_fp)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Format              : {summary['format']}")
        print(f"Content fingerprint : {content_fp}")
        print(f"Metadata fingerprint: {meta_fp}")
        print(f"Architecture        : {summary['architecture']}")
        print(f"Model name          : {summary['name']}")
        print(f"Tensors             : {summary['tensor_count']}")
        print(f"Parameters          : {summary['total_parameters']:,}")
        print(f"Quantization        : {summary['quantization']}")
        print(f"Dtype profile       : {summary['dtype_profile']}")
        print(f"File size           : {summary['file_size_bytes']:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
