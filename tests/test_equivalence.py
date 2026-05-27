"""Rehydration-equivalence contract: positive (bitwise, cross-process) and
adversarial negative (non-vacuity) tests.

These are the load-bearing tests of the project. The positive cases prove a
serialized-then-rehydrated state reproduces the continuation bit-for-bit across
a fresh process; the negative cases prove a corrupted state does *not*, so the
guarantee is not vacuously true.
"""

from __future__ import annotations

import pytest

from hybridserve_state import verify
from hybridserve_state.reference import ModelConfig

# Two architectures, including one hybrid (recurrent + attention interleaved).
HYBRID = ("recurrent", "attn", "recurrent")
PURE_RECURRENT = ("recurrent", "recurrent")


@pytest.mark.parametrize(
    "layer_types", [HYBRID, PURE_RECURRENT], ids=["hybrid", "pure_recurrent"]
)
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_positive_equivalence_is_bitwise_cross_process(tmp_path, layer_types, seed):
    cfg = ModelConfig(seed=seed, layer_types=layer_types)
    result = verify.run_equivalence(
        cfg, prompt=[1, 2, 3], total=20, checkpoint_at=8, tmpdir=str(tmp_path)
    )
    assert result.tokens_match, result.render()
    assert result.logits_bitwise_match, result.render()
    assert result.ok
    assert result.n_compared == 12


@pytest.mark.parametrize(
    "tag,corruption",
    [
        ("conv_phase", verify.corrupt_conv_phase),
        ("recurrent_fold", verify.corrupt_recurrent_fold),
        ("attn_drop_row", verify.corrupt_attn_drop_row),
    ],
)
def test_negative_corruption_is_detected(tmp_path, tag, corruption):
    cfg = ModelConfig(seed=0, layer_types=HYBRID)
    detected, msg = verify.run_negative(
        cfg,
        prompt=[1, 2, 3],
        total=20,
        checkpoint_at=8,
        corruption=corruption,
        tmpdir=str(tmp_path),
        tag=tag,
    )
    assert detected, msg


def test_selfcheck_all_pass():
    report = verify.run_selfcheck(seed=0)
    assert report.ok, report.render()
    # exactly one positive + three negative lines
    assert len(report.lines) == 4
    assert all("PASS" in line for line in report.lines), report.render()


def test_epsilon_is_committed_zero():
    # The bitwise tolerance is a committed constant and must never be widened.
    assert verify.EPSILON == 0.0


def test_checkpoint_bounds_validated(tmp_path):
    cfg = ModelConfig(seed=0)
    with pytest.raises(ValueError):
        verify.run_equivalence(
            cfg, [1, 2, 3], total=10, checkpoint_at=10, tmpdir=str(tmp_path)
        )
