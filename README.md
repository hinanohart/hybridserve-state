# hybridserve-state

**Engine-agnostic on-disk interchange for hybrid SSM + Attention LLM inference
state, with a machine-checked rehydration-equivalence contract.**

> Status: **v0.1.0a1 — pre-alpha.** The container format and the
> rehydration-equivalence contract are real and tested on CPU; the surface is
> small and the API may change. Read the [CLAIM / NON-CLAIM](#claim--non-claim)
> section before depending on this.

Modern open LLMs are increasingly **hybrid**: they interleave
attention layers (which carry a growing key/value cache) with
state-space / linear-attention layers (which carry a fixed-size *recurrent*
state plus a short causal-convolution buffer). Their *inference state* is
therefore heterogeneous — part KV cache, part recurrent fold, part conv ring
buffer — and today it lives only in a specific engine's memory, in that engine's
private layout.

`hybridserve-state` defines `.hss`: a small, safetensors-shaped container that
writes that heterogeneous state to disk in an **engine-neutral** way, together
with the semantics needed to put it back (`hss-spec/hss-spec.md`). On top of the
container it ships a **rehydration-equivalence contract**: a harness that
interrupts a generation, serializes the state, rehydrates it in a *fresh
process*, resumes, and checks that the continuation is **bitwise identical** to
the run that was never interrupted — and an adversarial test that proves a
*corrupted* rehydration is *rejected*, so the guarantee cannot be vacuous.

Think "safetensors for inference state."

## Why

You cannot today take the mid-generation state of a hybrid model out of one
process and resume it elsewhere with a *checked* guarantee that nothing drifted.
KV-cache offloading exists (see [Related work](#related-work-what-this-is-not)),
but it is in-memory or transformer-KV-only and tied to one engine, and none of
it ships a machine-checked equivalence contract that also covers the recurrent
and conv parts of a hybrid model. That gap is what this project fills.

## Install

```bash
python -m venv .venv && . .venv/bin/activate   # or your preferred env
pip install maturin numpy
maturin develop --release      # builds the Rust core into the Python package
```

Run the build from inside the activated environment (the same flow CI uses); if
`maturin` is not found on `PATH`, invoke it as `python -m maturin develop
--release`. Requires a stable Rust toolchain and Python ≥ 3.10. NumPy is the only
runtime dependency. PyTorch / `flash-linear-attention` are **optional** and only
used by the (CI-skipped) real-engine adapters.

## Quickstart

```python
import numpy as np
import hybridserve_state as hss

state = {
    "layers.0.recurrent_state": np.zeros((8, 16), dtype=np.float32),
    "layers.0.conv_state":      np.zeros((8, 3),  dtype=np.float32),
    "layers.1.attn_kv.k":       np.zeros((4, 12, 8), dtype=np.float32),
    "layers.1.attn_kv.v":       np.zeros((4, 12, 8), dtype=np.float32),
}
meta = {
    "n_layers": "2",
    "seen_tokens": "12",
    "layers.0.role": "recurrent_state",
    "layers.1.role": "attn_kv",
    "layers.0.conv_phase": "2",
}

hss.save("state.hss", state, meta)
loaded, meta2 = hss.load("state.hss")
assert all(np.array_equal(state[k], loaded[k]) for k in state)
```

```bash
hss inspect state.hss        # show structure
hss verify  state.hss        # parse + check metadata invariants
hss diff a.hss b.hss         # byte-level comparison
hss selfcheck                # run the bitwise rehydration-equivalence self-check
```

## How the equivalence contract works

The contract is proven on **deterministic, pure-NumPy reference layers** that
exercise both halves of a hybrid stack:

* a **gated-linear-attention–style** layer with a `recurrent_state` and a
  depthwise `conv_state` ring buffer, and
* a **softmax-attention** layer with a growing `attn_kv` cache.

The self-check (`hss selfcheck`, and `tests/`):

1. runs greedy decoding for *N* steps and records the continuation;
2. re-runs, but at step *k* serializes the full hybrid state to `.hss`,
   **rehydrates it in a separate Python process**, and resumes;
3. asserts the two continuations are **bitwise identical** (ε = 0, a committed
   constant — never widened).

A separate **adversarial negative test** corrupts the rehydrated state — it
rotates a conv-phase, zeroes a recurrent fold, and drops the most recent
attention-KV row — and asserts the resumed continuation **diverges** in each
case, i.e. the format actually carries the information the guarantee depends on,
and the guarantee is not vacuously true. (`seen_tokens` is *identity* metadata
the spec requires a concatenating reader to honor, but the reference recurrence
does not read it, so it is not one of the exercised corruptions — see
`hss-spec/hss-spec.md` §4.3.)

## CLAIM / NON-CLAIM

**CLAIM (tested in this release):**

* `.hss` is an engine-neutral on-disk container for heterogeneous hybrid
  inference state (recurrent + conv + attention-KV + ffn roles), with a
  `unsafe`-free, fuzz-smoke-tested Rust parser at the untrusted-input boundary.
* Save → load is byte-exact for supported NumPy dtypes; serialization is
  deterministic (byte-identical output for identical input).
* **Bitwise** rehydration-equivalence holds across a fresh process on the
  **deterministic CPU reference layers** (≥ 2 layer roles incl. a hybrid mix).
* A corrupted rehydration is **detected** (the adversarial negative test fails
  loudly), so the equivalence guarantee is non-vacuous.

**NON-CLAIM (explicitly out of scope for v0.1.0a1):**

* This is **not** a serving engine and makes **no** performance/throughput
  claims.
* `adapters/fla.py` maps the `flash-linear-attention` `Cache` *structure*
  (`recurrent_state` / `conv_state` / `attn_state` / `offset` / `layer_idx`)
  to/from `.hss` and is unit-tested against that documented schema using
  synthetic Cache objects; **end-to-end resume on a real `fla` model is not part
  of this release.**
* vLLM / SGLang adapters are **deferred to v0.1.1 and are not part of this
  release**. The `python/hybridserve_state/_experimental/` package is reserved
  for them and is excluded from CI; it ships empty in v0.1.0a1.
* Bitwise equivalence is asserted only on **CPU-deterministic** kernels.
  Non-deterministic GPU kernels are out of scope; equivalence there would be a
  tolerance-based claim, which this release does not make.

## Related work (what this is *not*)

KV-cache offloading and state persistence are active areas; `hybridserve-state`
is deliberately narrow and does **not** claim to be the first to put inference
state on disk:

* **vLLM / SGLang** hybrid-state support is **in-memory** and engine-internal.
* **LMCache / llm-d / KVSwap / KV offloading** persist or tier the **transformer
  KV cache**; they do not cover SSM recurrent/conv state and ship no cross-engine
  equivalence contract.
* **safetensors** is the inspiration for the container shape, but stores model
  *weights*, not heterogeneous *inference state* with role/boundary semantics.

What is new here is the combination: **engine-neutral container for the *hybrid*
(recurrent + conv + attention) state, plus a machine-checked rehydration-
equivalence contract with an adversarial non-vacuity test.**

## Project layout

```
core/      Rust: zero-copy, unsafe-free .hss container reader/writer (+ fuzz smoke)
bindings/  Rust: PyO3 bindings -> hybridserve_state._core
python/    Python: io, roles, verify (equivalence harness), cli, adapters
hss-spec/  the container + semantics specification
tests/     round-trip, equivalence, and adversarial negative tests
fuzz/       optional cargo-fuzz target (nightly; CI uses the in-test fuzz smoke)
```

## License

MIT. See [LICENSE](LICENSE).
