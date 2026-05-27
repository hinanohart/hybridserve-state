"""The rehydration-equivalence harness: the machine-checked core of the project.

Positive contract
    Run a deterministic decode for ``total`` tokens. Separately, run it for
    ``checkpoint_at`` tokens, serialize the full hybrid state to ``.hss``,
    rehydrate it **in a fresh process** (:mod:`._resume_worker`), resume, and
    assert the continuation is **bitwise identical** (ε = 0) to the
    uninterrupted run — both the integer tokens and the ``float64`` logits.

Non-vacuity (adversarial negative)
    For each boundary field the format carries (conv-phase, recurrent fold,
    attention KV length), corrupt it and assert the resumed continuation
    **diverges** (or the rehydration is rejected outright). This proves the
    guarantee is not vacuously true: the bytes genuinely determine the resume.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .reference import ModelConfig, ReferenceModel

# Committed equivalence tolerance. This is bitwise: it is never widened.
EPSILON = 0.0

Corruption = Callable[[dict, dict], None]


def _bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _resume_in_subprocess(
    hss_path: str, n: int, tmpdir: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Resume in a fresh process. Returns ``(tokens, logits)`` or ``None`` if the
    worker rejected the (possibly corrupted) state."""
    out = os.path.join(tmpdir, f"resume_{os.path.basename(hss_path)}.npz")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "hybridserve_state._resume_worker",
            hss_path,
            str(n),
            out,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not os.path.exists(out):
        return None
    data = np.load(out)
    return data["tokens"], data["logits"]


@dataclass
class EquivalenceResult:
    ok: bool
    n_compared: int
    tokens_match: bool
    logits_bitwise_match: bool

    def render(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        return (
            f"[{status}] positive rehydration-equivalence over {self.n_compared} tokens "
            f"(cross-process): tokens_match={self.tokens_match} "
            f"logits_bitwise_match={self.logits_bitwise_match} (epsilon={EPSILON})"
        )


def run_equivalence(
    cfg: ModelConfig,
    prompt: list[int],
    total: int,
    checkpoint_at: int,
    tmpdir: str,
) -> EquivalenceResult:
    if not 0 < checkpoint_at < total:
        raise ValueError("require 0 < checkpoint_at < total")

    full = ReferenceModel(cfg)
    full.warm_prompt(prompt)
    full_pairs = full.gen_steps(total)
    full_tokens = np.array([t for t, _ in full_pairs], dtype=np.int64)
    full_logits = np.stack([lg for _, lg in full_pairs])
    tail_tokens = full_tokens[checkpoint_at:]
    tail_logits = full_logits[checkpoint_at:]

    ck = ReferenceModel(cfg)
    ck.warm_prompt(prompt)
    ck.gen_steps(checkpoint_at)
    path = os.path.join(tmpdir, "ckpt.hss")
    ck.save_state(path)

    remaining = total - checkpoint_at
    resumed = _resume_in_subprocess(path, remaining, tmpdir)
    if resumed is None:
        return EquivalenceResult(False, remaining, False, False)
    rt, rl = resumed
    tokens_match = _bitwise_equal(tail_tokens, rt)
    logits_match = _bitwise_equal(tail_logits, rl)
    return EquivalenceResult(
        tokens_match and logits_match, remaining, tokens_match, logits_match
    )


# --- adversarial corruptions (each must cause divergence / rejection) ---------


def corrupt_conv_phase(state: dict, meta: dict) -> None:
    """Rotate the first conv layer's ring-buffer phase by one — a 'valid-looking'
    but wrong phase that reorders the causal-conv window."""
    cfg = ModelConfig.from_metadata(meta)
    for key in list(meta):
        if key.endswith(".conv_phase"):
            meta[key] = str((int(meta[key]) + 1) % cfg.conv_kernel)
            return


def corrupt_recurrent_fold(state: dict, meta: dict) -> None:
    """Zero a recurrent_state, as if a fold chunk were dropped."""
    for name in state:
        if name.endswith(".recurrent_state"):
            state[name] = np.zeros_like(state[name])
            return


def corrupt_attn_drop_row(state: dict, meta: dict) -> None:
    """Drop the most recent cached KV position from the attention layer."""
    for name in list(state):
        if name.endswith(".attn_kv.k") or name.endswith(".attn_kv.v"):
            arr = state[name]
            if arr.shape[0] > 1:
                state[name] = arr[:-1].copy()


def run_negative(
    cfg: ModelConfig,
    prompt: list[int],
    total: int,
    checkpoint_at: int,
    corruption: Corruption,
    tmpdir: str,
    tag: str,
) -> tuple[bool, str]:
    """Returns ``(detected, message)`` where ``detected`` is True iff the
    corrupted rehydration either diverged or was rejected."""
    full = ReferenceModel(cfg)
    full.warm_prompt(prompt)
    full_pairs = full.gen_steps(total)
    tail_tokens = np.array([t for t, _ in full_pairs][checkpoint_at:], dtype=np.int64)
    tail_logits = np.stack([lg for _, lg in full_pairs])[checkpoint_at:]

    ck = ReferenceModel(cfg)
    ck.warm_prompt(prompt)
    ck.gen_steps(checkpoint_at)
    state, meta = ck.capture_state()
    corruption(state, meta)

    path = os.path.join(tmpdir, f"corrupt_{tag}.hss")
    from . import io as hss_io

    hss_io.save(path, state, meta)

    remaining = total - checkpoint_at
    resumed = _resume_in_subprocess(path, remaining, tmpdir)
    if resumed is None:
        return True, f"[PASS] negative/{tag}: corrupted state was rejected"
    rt, rl = resumed
    # Non-vacuity is measured at the same granularity as the positive claim: a
    # corrupted state must fail to reproduce the bitwise-identical continuation
    # (either the tokens or the float64 logits differ).
    reproduces = _bitwise_equal(tail_tokens, rt) and _bitwise_equal(tail_logits, rl)
    diverged = not reproduces
    status = "PASS" if diverged else "FAIL"
    return diverged, f"[{status}] negative/{tag}: diverged={diverged} (must be True)"


@dataclass
class SelfCheckReport:
    ok: bool
    lines: list[str]

    def render(self) -> str:
        header = "rehydration-equivalence self-check: " + (
            "ALL PASS" if self.ok else "FAILED"
        )
        return "\n".join([header, *self.lines])


def run_selfcheck(
    seed: int = 0,
    prompt: list[int] | None = None,
    total: int = 20,
    checkpoint_at: int = 8,
) -> SelfCheckReport:
    """Run the positive equivalence test plus all adversarial negatives on the
    default hybrid reference model. Used by ``hss selfcheck`` and the test suite."""
    cfg = ModelConfig(seed=seed)
    prompt = prompt or [1, 2, 3]
    lines: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        pos = run_equivalence(cfg, prompt, total, checkpoint_at, tmp)
        lines.append(pos.render())
        negatives = [
            ("conv_phase", corrupt_conv_phase),
            ("recurrent_fold", corrupt_recurrent_fold),
            ("attn_drop_row", corrupt_attn_drop_row),
        ]
        all_neg_ok = True
        for tag, fn in negatives:
            detected, msg = run_negative(
                cfg, prompt, total, checkpoint_at, fn, tmp, tag
            )
            lines.append(msg)
            all_neg_ok = all_neg_ok and detected
    return SelfCheckReport(ok=pos.ok and all_neg_ok, lines=lines)
