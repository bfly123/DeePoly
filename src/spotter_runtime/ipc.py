from __future__ import annotations

import enum
import json
import struct
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


_HEADER_STRUCT = struct.Struct("!Q")


def send_message(stream: BinaryIO, payload: Any) -> None:
    encoded = _encode(payload)
    data = json.dumps(encoded, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    stream.write(_HEADER_STRUCT.pack(len(data)))
    stream.write(data)
    stream.flush()


def recv_message(stream: BinaryIO) -> Any:
    header = _read_exact(stream, _HEADER_STRUCT.size)
    if not header:
        raise EOFError("EOF while reading IPC header")
    (size,) = _HEADER_STRUCT.unpack(header)
    body = _read_exact(stream, size)
    if body is None:
        raise EOFError("EOF while reading IPC payload")
    payload = json.loads(body.decode("utf-8"))
    return _decode(payload)


def _read_exact(stream: BinaryIO, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            return None if not chunks else bytes(chunks)
        chunks.extend(chunk)
    return bytes(chunks)


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return _encode(value.value)
    if is_dataclass(value):
        return _encode(asdict(value))
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": True, "items": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if hasattr(value, "__dict__"):
        return {str(key): _encode(item) for key, item in vars(value).items()}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__ndarray__"):
            array = np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
            return array.reshape(value["shape"])
        if value.get("__tuple__"):
            return tuple(_decode(item) for item in value["items"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value
