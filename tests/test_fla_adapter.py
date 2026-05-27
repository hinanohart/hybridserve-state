"""Structural round-trip tests for the flash-linear-attention Cache adapter.

These use *synthetic* Cache objects that mirror the documented FLA ``Cache``
schema, so they require neither ``torch`` nor ``flash-linear-attention``.
"""

from __future__ import annotations

import numpy as np

from hybridserve_state import roles
from hybridserve_state.adapters import fla


class SyntheticCache:
    """Mimics ``fla.models.utils.Cache``: a ``.states`` list of per-layer dicts
    plus a ``_seen_tokens`` counter."""

    def __init__(self, states, seen_tokens):
        self.states = states
        self._seen_tokens = seen_tokens


def _hybrid_cache(seed=0):
    rng = np.random.default_rng(seed)
    states = [
        {  # layer 0: linear-attention (recurrent + conv)
            "recurrent_state": rng.standard_normal((4, 8, 8)).astype(np.float32),
            "conv_state": rng.standard_normal((4, 16, 3)).astype(np.float32),
            "offset": 12,
        },
        {  # layer 1: softmax attention (KV cache)
            "attn_state": (
                rng.standard_normal((4, 16, 12, 8)).astype(np.float32),
                rng.standard_normal((4, 16, 12, 8)).astype(np.float32),
            ),
            "offset": 12,
        },
        {  # layer 2: linear-attention + ffn aux state
            "recurrent_state": rng.standard_normal((4, 8, 8)).astype(np.float32),
            "conv_state": rng.standard_normal((4, 16, 3)).astype(np.float32),
            "ffn_state": rng.standard_normal((4, 32)).astype(np.float32),
            "offset": 12,
        },
    ]
    return SyntheticCache(states, seen_tokens=12)


def test_cache_to_state_metadata_and_roles():
    cache = _hybrid_cache()
    state, meta = fla.cache_to_state(cache)

    assert meta[roles.KEY_ENGINE] == "flash-linear-attention"
    assert meta[roles.KEY_N_LAYERS] == "3"
    assert meta[roles.KEY_SEEN_TOKENS] == "12"
    assert meta[roles.layer_role_key(0)] == roles.RECURRENT_STATE
    assert meta[roles.layer_role_key(1)] == roles.ATTN_KV
    assert meta[roles.layer_role_key(2)] == roles.RECURRENT_STATE
    assert meta[roles.layer_conv_phase_key(0)] == "0"

    assert "layers.0.recurrent_state" in state
    assert "layers.0.conv_state" in state
    assert "layers.1.attn_state.k" in state
    assert "layers.1.attn_state.v" in state
    assert "layers.2.ffn_state" in state
    # metadata invariants hold
    roles.validate_metadata(meta)


def test_cache_round_trip_through_hss(tmp_path):
    cache = _hybrid_cache(seed=7)
    path = tmp_path / "cache.hss"
    fla.save_cache(cache, str(path))
    layers, meta = fla.load_layers(str(path))

    assert len(layers) == 3
    orig = cache.states

    # layer 0: recurrent + conv preserved bitwise
    assert np.array_equal(layers[0]["recurrent_state"], orig[0]["recurrent_state"])
    assert np.array_equal(layers[0]["conv_state"], orig[0]["conv_state"])
    assert layers[0]["offset"] == 12

    # layer 1: attn_state tuple reconstructed
    k, v = layers[1]["attn_state"]
    assert np.array_equal(k, orig[1]["attn_state"][0])
    assert np.array_equal(v, orig[1]["attn_state"][1])

    # layer 2: recurrent + conv + ffn
    assert np.array_equal(layers[2]["ffn_state"], orig[2]["ffn_state"])
    assert meta[roles.KEY_SEEN_TOKENS] == "12"


def test_state_to_layers_without_metadata_infers_layer_count():
    cache = _hybrid_cache()
    state, _meta = fla.cache_to_state(cache)
    # Drop metadata entirely; layer count must be recovered from tensor names.
    layers = fla.state_to_layers(state, {})
    assert len(layers) == 3
    assert "recurrent_state" in layers[0]
    assert "attn_state" in layers[1]


def test_default_tensor_to_numpy_passes_through_ndarray():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert fla.default_tensor_to_numpy(a) is a


def test_states_as_plain_list_supported():
    # A cache may be passed as a bare list of per-layer dicts.
    states = [{"recurrent_state": np.zeros((2, 2), np.float32), "offset": 3}]
    state, meta = fla.cache_to_state(states)
    assert meta[roles.KEY_N_LAYERS] == "1"
    assert meta[roles.KEY_SEEN_TOKENS] == "3"
