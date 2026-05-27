"""Per-layer ROLE vocabulary and metadata-key conventions for hybrid state.

A ``.hss`` container is engine-agnostic: the Rust core only guarantees the bytes
are well-formed. The *semantics* of a hybrid SSM+Attention state live entirely in
the ``__metadata__`` string map, using the conventions defined here and in
``hss-spec/hss-spec.md``. Keeping them in one module lets adapters, the
equivalence harness, and the CLI agree on a single source of truth.
"""

from __future__ import annotations

# Per-layer ROLE vocabulary. A hybrid model mixes these across its layer stack.
RECURRENT_STATE = "recurrent_state"  # SSM / linear-attention fold of seen tokens
CONV_STATE = "conv_state"  # short causal-conv ring buffer (depthwise)
ATTN_KV = "attn_kv"  # softmax-attention key/value cache
FFN_STATE = "ffn_state"  # any feed-forward / MoE auxiliary carried state

ROLES: frozenset[str] = frozenset({RECURRENT_STATE, CONV_STATE, ATTN_KV, FFN_STATE})

# Reserved metadata keys (string values only, per container format).
KEY_HSS_VERSION = "hss_version"
KEY_BYTE_ORDER = "byte_order"
KEY_ENGINE = "engine"
KEY_MODEL_ARCH = "model_arch"
KEY_N_LAYERS = "n_layers"
KEY_SEEN_TOKENS = "seen_tokens"


def layer_role_key(layer: int) -> str:
    """Metadata key holding a layer's ROLE, e.g. ``layers.3.role``."""
    return f"layers.{layer}.role"


def layer_conv_phase_key(layer: int) -> str:
    """Metadata key holding a conv layer's ring-buffer write position."""
    return f"layers.{layer}.conv_phase"


def layer_chunk_boundary_key(layer: int) -> str:
    """Metadata key holding an SSM layer's last chunk-boundary token index."""
    return f"layers.{layer}.chunk_boundary"


class SemanticError(ValueError):
    """Raised when ``__metadata__`` violates the hybrid-state invariants."""


def validate_metadata(metadata: dict[str, str]) -> None:
    """Check the hybrid-state semantic invariants on a metadata map.

    Enforces (when the relevant keys are present):

    * every ``layers.{i}.role`` value is a known ROLE,
    * ``seen_tokens`` and ``n_layers`` parse as non-negative integers,
    * each ``conv_phase`` / ``chunk_boundary`` parses as a non-negative integer.

    Absent keys are tolerated (a container may describe a partial state); this
    validates *consistency*, not *completeness*.
    """
    for key, value in metadata.items():
        if key.endswith(".role"):
            if value not in ROLES:
                raise SemanticError(
                    f"{key}={value!r} is not a known ROLE ({', '.join(sorted(ROLES))})"
                )
        elif key.endswith(".conv_phase") or key.endswith(".chunk_boundary"):
            _require_nonneg_int(key, value)

    if KEY_SEEN_TOKENS in metadata:
        _require_nonneg_int(KEY_SEEN_TOKENS, metadata[KEY_SEEN_TOKENS])
    if KEY_N_LAYERS in metadata:
        _require_nonneg_int(KEY_N_LAYERS, metadata[KEY_N_LAYERS])


def _require_nonneg_int(key: str, value: str) -> None:
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise SemanticError(f"{key}={value!r} must be an integer") from exc
    if n < 0:
        raise SemanticError(f"{key}={value!r} must be non-negative")
