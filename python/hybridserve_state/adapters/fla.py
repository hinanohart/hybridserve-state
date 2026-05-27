"""flash-linear-attention ``Cache`` <-> ``.hss`` structural mapping.

**Scope (CLAIM).** This adapter maps the *structure* of the
`flash-linear-attention <https://github.com/fla-org/flash-linear-attention>`_
``Cache`` — its documented per-layer anchors ``recurrent_state`` /
``conv_state`` / ``attn_state`` / ``ffn_state`` and the per-layer ``offset``
(seen tokens) — to and from the ``.hss`` container. It is unit-tested against
that schema using *synthetic* Cache objects, so it neither imports nor requires
`torch`/`fla`.

**Out of scope (NON-CLAIM).** End-to-end resume of a real ``fla`` model is not
part of v0.1.0a1. Tensor conversion for exotic dtypes (e.g. ``bfloat16``, which
NumPy cannot represent) is delegated to a caller-supplied ``tensor_to_numpy``
hook rather than guessed at.

A FLA cache is treated as an object exposing ``.states``: a list with one entry
per layer, each a mapping that may contain any of::

    {
        "recurrent_state": <tensor>,             # SSM / linear-attn fold
        "conv_state":      <tensor>,             # depthwise conv ring buffer
        "attn_state":      (<k tensor>, <v tensor>),  # softmax-attn KV
        "ffn_state":       <tensor>,             # optional aux state
        "offset":          <int>,                # tokens folded into this layer
    }
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .. import io as hss_io
from .. import roles

TensorToNumpy = Callable[[Any], np.ndarray]

# Per-layer anchor keys understood by this adapter.
_RECURRENT = "recurrent_state"
_CONV = "conv_state"
_ATTN = "attn_state"
_FFN = "ffn_state"
_OFFSET = "offset"

__all__ = [
    "cache_to_state",
    "state_to_layers",
    "save_cache",
    "load_layers",
    "default_tensor_to_numpy",
]


def default_tensor_to_numpy(t: Any) -> np.ndarray:
    """Best-effort conversion of a framework tensor to a NumPy array.

    Handles NumPy arrays and CPU-movable tensors that expose ``.detach`` /
    ``.cpu`` / ``.numpy``. Raises ``NotImplementedError`` for tensors NumPy
    cannot represent (e.g. ``bfloat16``); pass a custom ``tensor_to_numpy`` hook
    in that case.
    """
    if isinstance(t, np.ndarray):
        return t
    obj = t
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if hasattr(obj, "numpy"):
        try:
            return np.asarray(obj.numpy())
        except (TypeError, ValueError) as exc:  # e.g. bfloat16
            raise NotImplementedError(
                f"cannot convert tensor of dtype {getattr(t, 'dtype', '?')} to NumPy; "
                "pass a custom tensor_to_numpy hook"
            ) from exc
    return np.asarray(obj)


def _iter_layer_states(cache: Any) -> list[Any]:
    states = getattr(cache, "states", None)
    if states is None:
        states = cache  # already a list of per-layer mappings
    return list(states)


def _seen_tokens(cache: Any, layer_states: list[Any]) -> int:
    for attr in ("_seen_tokens", "seen_tokens"):
        v = getattr(cache, attr, None)
        if isinstance(v, int):
            return v
    get_len = getattr(cache, "get_seq_length", None)
    if callable(get_len):
        try:
            return int(get_len())
        except Exception:  # noqa: BLE001 - tolerate unusual cache APIs
            pass
    offsets = [int(s[_OFFSET]) for s in layer_states if _has(s, _OFFSET)]
    return max(offsets) if offsets else 0


def _has(state: Any, key: str) -> bool:
    try:
        return key in state and state[key] is not None
    except TypeError:
        return False


def cache_to_state(
    cache: Any,
    *,
    tensor_to_numpy: TensorToNumpy = default_tensor_to_numpy,
    extra_metadata: dict[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Flatten a FLA cache into a ``(state, metadata)`` pair ready for
    :func:`hybridserve_state.io.save`."""
    layer_states = _iter_layer_states(cache)
    state: dict[str, np.ndarray] = {}
    meta: dict[str, str] = {
        roles.KEY_ENGINE: "flash-linear-attention",
        roles.KEY_N_LAYERS: str(len(layer_states)),
        roles.KEY_SEEN_TOKENS: str(_seen_tokens(cache, layer_states)),
    }

    for i, ls in enumerate(layer_states):
        has_attn = _has(ls, _ATTN)
        has_recurrent = _has(ls, _RECURRENT)

        # Coarse ROLE label = the layer's primary mixer type.
        if has_attn:
            meta[roles.layer_role_key(i)] = roles.ATTN_KV
        elif has_recurrent:
            meta[roles.layer_role_key(i)] = roles.RECURRENT_STATE

        if has_recurrent:
            state[f"layers.{i}.{_RECURRENT}"] = tensor_to_numpy(ls[_RECURRENT])
        if _has(ls, _CONV):
            state[f"layers.{i}.{_CONV}"] = tensor_to_numpy(ls[_CONV])
            # FLA stores the conv buffer already aligned; record phase 0 unless
            # the cache provides one explicitly.
            phase = ls.get("conv_phase", 0) if hasattr(ls, "get") else 0
            meta[roles.layer_conv_phase_key(i)] = str(int(phase))
        if has_attn:
            k, v = ls[_ATTN]
            state[f"layers.{i}.{_ATTN}.k"] = tensor_to_numpy(k)
            state[f"layers.{i}.{_ATTN}.v"] = tensor_to_numpy(v)
        if _has(ls, _FFN):
            state[f"layers.{i}.{_FFN}"] = tensor_to_numpy(ls[_FFN])
        if _has(ls, _OFFSET):
            meta[f"layers.{i}.offset"] = str(int(ls[_OFFSET]))

    if extra_metadata:
        meta.update({str(k): str(v) for k, v in extra_metadata.items()})
    roles.validate_metadata(meta)
    return state, meta


def state_to_layers(
    state: dict[str, np.ndarray], metadata: dict[str, str]
) -> list[dict[str, Any]]:
    """Inverse of :func:`cache_to_state`: regroup a flat state into a list of
    per-layer mappings (NumPy tensors), one entry per layer index."""
    n_layers = int(metadata.get(roles.KEY_N_LAYERS, "0"))
    # Discover layer count from names too, in case metadata is absent.
    for name in state:
        if name.startswith("layers."):
            try:
                idx = int(name.split(".", 2)[1])
            except (IndexError, ValueError):
                continue
            n_layers = max(n_layers, idx + 1)

    layers: list[dict[str, Any]] = [dict() for _ in range(n_layers)]
    attn_parts: dict[int, dict[str, np.ndarray]] = {}

    for name, arr in state.items():
        if not name.startswith("layers."):
            continue
        parts = name.split(".")
        i = int(parts[1])
        leaf = ".".join(parts[2:])
        if leaf == _RECURRENT:
            layers[i][_RECURRENT] = arr
        elif leaf == _CONV:
            layers[i][_CONV] = arr
        elif leaf == _FFN:
            layers[i][_FFN] = arr
        elif leaf == f"{_ATTN}.k":
            attn_parts.setdefault(i, {})["k"] = arr
        elif leaf == f"{_ATTN}.v":
            attn_parts.setdefault(i, {})["v"] = arr

    for i, kv in attn_parts.items():
        if "k" in kv and "v" in kv:
            layers[i][_ATTN] = (kv["k"], kv["v"])

    for i in range(n_layers):
        off_key = f"layers.{i}.offset"
        if off_key in metadata:
            layers[i][_OFFSET] = int(metadata[off_key])

    return layers


def save_cache(
    cache: Any,
    path: str,
    *,
    tensor_to_numpy: TensorToNumpy = default_tensor_to_numpy,
    extra_metadata: dict[str, str] | None = None,
) -> None:
    """Serialize a FLA cache directly to a ``.hss`` file."""
    state, meta = cache_to_state(
        cache, tensor_to_numpy=tensor_to_numpy, extra_metadata=extra_metadata
    )
    hss_io.save(path, state, meta)


def load_layers(path: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load a ``.hss`` file written by this adapter back into per-layer
    mappings of NumPy tensors plus the metadata map."""
    state, meta = hss_io.load(path)
    return state_to_layers(state, meta), meta
