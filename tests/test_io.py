"""Round-trip and container-level tests for the NumPy <-> .hss I/O layer."""

from __future__ import annotations

import numpy as np
import pytest

import hybridserve_state as hss
from hybridserve_state import roles


def _sample_state():
    rng = np.random.default_rng(1234)
    return {
        "layers.0.recurrent_state": rng.standard_normal((8, 16)).astype(np.float32),
        "layers.0.conv_state": rng.standard_normal((8, 3)).astype(np.float32),
        "layers.1.attn_kv.k": rng.standard_normal((4, 12, 8)).astype(np.float32),
        "layers.1.attn_kv.v": rng.standard_normal((4, 12, 8)).astype(np.float32),
        "seen_tokens_vec": np.arange(12, dtype=np.int64),
        "flags": np.array([True, False, True], dtype=bool),
    }


def test_round_trip_preserves_values_dtypes_shapes(tmp_path):
    state = _sample_state()
    meta = {
        "n_layers": "2",
        "seen_tokens": "12",
        "layers.0.role": roles.RECURRENT_STATE,
        "layers.1.role": roles.ATTN_KV,
        "layers.0.conv_phase": "2",
    }
    path = tmp_path / "state.hss"
    hss.save(str(path), state, meta)
    loaded, meta2 = hss.load(str(path))

    assert set(loaded) == set(state)
    for k in state:
        assert loaded[k].dtype == state[k].dtype, k
        assert loaded[k].shape == state[k].shape, k
        assert np.array_equal(loaded[k], state[k]), k

    # metadata preserved (plus injected hss_version / byte_order)
    for k, v in meta.items():
        assert meta2[k] == v
    assert meta2["byte_order"] == "little"
    assert "hss_version" in meta2


def test_serialization_is_byte_deterministic(tmp_path):
    state = _sample_state()
    a = tmp_path / "a.hss"
    b = tmp_path / "b.hss"
    hss.save(str(a), state, {"engine": "ref"})
    hss.save(str(b), state, {"engine": "ref"})
    assert a.read_bytes() == b.read_bytes()


def test_inspect_reports_structure_without_loading(tmp_path):
    state = _sample_state()
    path = tmp_path / "s.hss"
    hss.save(str(path), state, {"n_layers": "2"})
    metadata, tensors = hss.inspect_file(str(path))
    names = {t[0] for t in tensors}
    assert names == set(state)
    by_name = {t[0]: t for t in tensors}
    name, dtype, shape, nbytes = by_name["layers.1.attn_kv.k"]
    assert dtype == "F32"
    assert shape == (4, 12, 8)
    assert nbytes == 4 * 12 * 8 * 4


def test_unsupported_dtype_raises(tmp_path):
    with pytest.raises(ValueError):
        hss.save(
            str(tmp_path / "x.hss"), {"c": np.array([1 + 2j], dtype=np.complex128)}
        )


def test_empty_state_round_trips(tmp_path):
    path = tmp_path / "empty.hss"
    hss.save(str(path), {})
    loaded, meta = hss.load(str(path))
    assert loaded == {}
    assert meta["byte_order"] == "little"


def test_scalar_and_zero_sized(tmp_path):
    state = {
        "scalar": np.array(3.5, dtype=np.float64),
        "empty": np.zeros((0,), dtype=np.float32),
    }
    path = tmp_path / "edge.hss"
    hss.save(str(path), state)
    loaded, _ = hss.load(str(path))
    assert loaded["scalar"].shape == ()
    assert np.array_equal(loaded["scalar"], state["scalar"])
    assert loaded["empty"].shape == (0,)


def test_metadata_validation_rejects_unknown_role():
    with pytest.raises(roles.SemanticError):
        roles.validate_metadata({"layers.0.role": "not_a_role"})


def test_metadata_validation_accepts_known(tmp_path):
    roles.validate_metadata(
        {
            "layers.0.role": roles.RECURRENT_STATE,
            "layers.1.role": roles.ATTN_KV,
            "seen_tokens": "100",
            "layers.0.conv_phase": "3",
        }
    )
