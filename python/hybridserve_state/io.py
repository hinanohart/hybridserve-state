"""NumPy <-> ``.hss`` I/O on top of the Rust container core.

The on-disk container is canonically little-endian. ``save`` normalizes every
multi-byte array to little-endian before writing and records ``byte_order`` in
the metadata, so files are portable and serialization is byte-deterministic.
"""

from __future__ import annotations

import sys
from typing import Mapping

import numpy as np

from ._core import __hss_version__, write as _write, read as _read, inspect as _inspect

# Mapping between NumPy dtype names and the container's canonical dtype strings.
_NP_TO_HSS: dict[str, str] = {
    "float64": "F64",
    "float32": "F32",
    "float16": "F16",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint8": "U8",
    "bool": "BOOL",
}
_HSS_TO_NP: dict[str, str] = {v: k for k, v in _NP_TO_HSS.items()}

__all__ = ["save", "load", "inspect_file", "supported_dtypes", "__hss_version__"]


def supported_dtypes() -> tuple[str, ...]:
    """NumPy dtype names this layer can serialize.

    The container format (``hss-spec`` §3) also defines ``BF16`` as a valid
    2-byte opaque dtype, but NumPy has no native ``bfloat16``, so this NumPy I/O
    layer neither produces nor consumes it; a ``bfloat16`` array must be carried
    as raw bytes through the Rust core directly. An unsupported dtype passed to
    :func:`save` raises ``ValueError`` rather than being silently mislabeled.
    """
    return tuple(_NP_TO_HSS)


def _to_little_endian(arr: np.ndarray) -> np.ndarray:
    """Return a little-endian view of ``arr``, preserving its shape (including
    0-d). Contiguity is handled by ``tobytes(order="C")`` at the call site, so we
    avoid ``np.ascontiguousarray`` here, which would force a minimum of 1 dim."""
    if arr.dtype.itemsize > 1:
        bo = arr.dtype.byteorder
        is_big = bo == ">" or (bo == "=" and sys.byteorder == "big")
        if is_big:
            arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))
    return arr


def save(
    path: str,
    state: Mapping[str, np.ndarray],
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Serialize a ``name -> ndarray`` mapping plus string metadata to ``path``.

    Raises ``ValueError`` for unsupported dtypes (see :func:`supported_dtypes`).
    """
    meta: dict[str, str] = {str(k): str(v) for k, v in (metadata or {}).items()}
    meta.setdefault("hss_version", __hss_version__)
    meta["byte_order"] = "little"

    names: list[str] = []
    dtypes: list[str] = []
    shapes: list[list[int]] = []
    buffers: list[bytes] = []

    for name, arr in state.items():
        arr = np.asarray(arr)
        np_name = arr.dtype.name
        if np_name not in _NP_TO_HSS:
            raise ValueError(
                f"tensor {name!r}: unsupported dtype {np_name!r}; "
                f"supported: {', '.join(_NP_TO_HSS)}"
            )
        arr = _to_little_endian(arr)
        names.append(str(name))
        dtypes.append(_NP_TO_HSS[np_name])
        shapes.append([int(d) for d in arr.shape])
        buffers.append(arr.tobytes(order="C"))

    _write(str(path), names, dtypes, shapes, buffers, meta)


def load(path: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load ``path`` into a ``(state, metadata)`` pair.

    Returned arrays are owned, writable, C-contiguous, and in native byte order.
    """
    metadata, tensors = _read(str(path))
    state: dict[str, np.ndarray] = {}
    for name, dtype, shape, data in tensors:
        np_dtype = np.dtype(_HSS_TO_NP[dtype]).newbyteorder("<")
        flat = np.frombuffer(data, dtype=np_dtype)
        arr = flat.reshape([int(d) for d in shape])
        # Copy off the read-only buffer into a writable, C-contiguous array in
        # native byte order. ``astype`` (unlike ``ascontiguousarray``) preserves
        # 0-d shape rather than forcing a minimum of one dimension.
        state[name] = arr.astype(arr.dtype.newbyteorder("="), order="C", copy=True)
    return state, metadata


def inspect_file(
    path: str,
) -> tuple[dict[str, str], list[tuple[str, str, tuple[int, ...], int]]]:
    """Return ``(metadata, [(name, dtype, shape, nbytes), ...])`` without copying
    tensor data into Python."""
    metadata, tensors = _inspect(str(path))
    descriptors = [
        (n, d, tuple(int(x) for x in s), int(nb)) for (n, d, s, nb) in tensors
    ]
    return metadata, descriptors
