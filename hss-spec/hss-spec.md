# The `.hss` container specification (v0.1)

`.hss` ("hybrid serving state") is an engine-neutral on-disk container for the
*inference state* of a hybrid SSM + Attention language model. It is deliberately
shaped like [safetensors](https://github.com/huggingface/safetensors): a length
prefix, a JSON header, then a flat data section. The difference is the payload —
heterogeneous **inference state**, not model weights — and the `__metadata__`
semantics that make a state **rehydratable** in a different process or engine.

This document is normative for v0.1. The Rust core
(`core/src/lib.rs`) enforces the structural rules; the semantic conventions in
[§4](#4-metadata-semantics) are enforced by
`python/hybridserve_state/roles.py` and consumed by the equivalence harness.

## 1. Byte layout

```text
┌──────────────┬───────────────────────────┬──────────────────────────┐
│ 8 bytes      │ N bytes                   │ remainder                │
│ u64 LE = N   │ UTF-8 JSON header         │ tensor data (concatenated)│
└──────────────┴───────────────────────────┴──────────────────────────┘
```

* The first 8 bytes are the JSON header length `N`, little-endian `u64`.
* The next `N` bytes are a UTF-8 JSON object (the header).
* The remaining bytes are the raw tensor data section. All tensor
  `data_offsets` are relative to the **start of this section** (i.e. relative to
  byte `8 + N` of the file).

A reader MUST reject a file where `8 + N` exceeds the file length, where
`N` exceeds an implementation limit (the reference core uses 100 MiB), or where
the header is not valid UTF-8 / JSON.

## 2. Header JSON

The header is a JSON object. Every key is either the reserved key
`"__metadata__"` or a **tensor name**.

A tensor entry is:

```json
"layers.0.recurrent_state": {
  "dtype": "F32",
  "shape": [8, 16],
  "data_offsets": [0, 512]
}
```

* `dtype` — one of the dtype strings in [§3](#3-dtypes).
* `shape` — array of non-negative integers. The empty array `[]` denotes a
  0-dimensional (scalar) tensor.
* `data_offsets` — `[start, end]`, byte offsets into the data section,
  `0 ≤ start ≤ end ≤ data_section_length`.

A reader MUST reject an entry where `start > end`, where `end` exceeds the data
section length, or where `product(shape) * sizeof(dtype) != end - start`
(checked with overflow-safe arithmetic).

`"__metadata__"` is reserved and MUST NOT be used as a tensor name. Writers MUST
emit tensor entries and header keys in a deterministic order (the reference
writer sorts by name and uses sorted JSON keys) so that identical input produces
byte-identical output — a prerequisite for the bitwise equivalence contract.

## 3. Dtypes

Dtypes are **byte-width containers**; the format does not interpret numeric
values, so e.g. `BF16` and `I16` are both opaque 2-byte little-endian elements.

| string | bytes | | string | bytes |
|--------|-------|-|--------|-------|
| `F64`  | 8     | | `I32`  | 4     |
| `F32`  | 4     | | `I16`  | 2     |
| `F16`  | 2     | | `I8`   | 1     |
| `BF16` | 2     | | `U8`   | 1     |
| `I64`  | 8     | | `BOOL` | 1     |

Multi-byte values are stored **little-endian**. A writer on a big-endian host
MUST byte-swap to little-endian and record `byte_order=little` (see §4).

## 4. `__metadata__` semantics

`__metadata__` is a JSON object of **string → string** pairs. All values are
strings (integers are decimal strings). It carries everything an engine needs to
*reconstruct* a hybrid state beyond the raw bytes.

### 4.1 Reserved keys

| key | meaning |
|-----|---------|
| `hss_version` | container spec version, e.g. `"0.1"` |
| `byte_order`  | `"little"` (only value defined in v0.1) |
| `engine`      | free-form producing-engine identifier (optional) |
| `model_arch`  | free-form architecture identifier (optional) |
| `n_layers`    | number of layers, decimal string (optional) |
| `seen_tokens` | total tokens already folded into the state (decimal) |

### 4.2 Per-layer ROLE

Each layer's contribution to the state has a **ROLE**, recorded as
`layers.{i}.role` with one of:

| ROLE | carries |
|------|---------|
| `recurrent_state` | the SSM / linear-attention fixed-size fold of seen tokens |
| `conv_state`      | a short depthwise causal-convolution ring buffer |
| `attn_kv`         | a growing softmax-attention key/value cache |
| `ffn_state`       | any feed-forward / MoE auxiliary carried state |

A hybrid model mixes ROLEs across its layer stack (e.g. linear-attention layers
with `recurrent_state` + `conv_state`, interleaved with `attn_kv` layers). Tensor
names SHOULD encode the layer and role, e.g. `layers.3.recurrent_state`,
`layers.3.conv_state`, `layers.7.attn_kv.k`, `layers.7.attn_kv.v`.

### 4.3 Boundary semantics (the part that makes rehydration *correct*)

These are the fields whose absence or corruption silently breaks a resume, and
which the adversarial negative test deliberately attacks:

* **`seen_tokens`** — A `recurrent_state` is a *lossy fold* of exactly this many
  tokens. Two states folded over different token counts are **not**
  concatenable; a reader that resumes MUST treat `seen_tokens` as part of the
  state's identity.
* **`layers.{i}.conv_phase`** — A `conv_state` is a ring buffer of the last
  `kernel_size - 1` inputs. `conv_phase` is the write position (0-based) within
  that ring. Resuming with the wrong phase rotates the buffer and corrupts the
  next conv output. Equivalently, a writer MAY normalize the ring so that phase
  is always 0; if so it MUST still record `conv_phase=0`.
* **`layers.{i}.chunk_boundary`** — For chunked SSM scans, the token index of the
  last completed chunk boundary. A resume that re-folds a partial chunk or skips
  one diverges. Optional when the engine folds token-by-token.

A reader MUST reject (or refuse to claim equivalence for) a state whose
`layers.{i}.role` is not a known ROLE, or whose `seen_tokens` / `n_layers` /
`conv_phase` / `chunk_boundary` are not non-negative integers.

## 5. The rehydration-equivalence contract

A producer and consumer satisfy the contract for a model and a decoding policy
when, for any interruption point `k`:

> Serializing the full state at step `k` to `.hss`, rehydrating it in a fresh
> process, and resuming the **same deterministic** decoding policy yields a
> continuation that is **bitwise identical** (ε = 0) to the continuation of an
> uninterrupted run.

ε = 0 is a committed constant and is never widened. The contract is only
asserted over **deterministic** kernels (the reference implementation forces a
deterministic CPU path). On non-deterministic kernels the contract is undefined;
this format makes no tolerance-based equivalence claim in v0.1.

The contract is only meaningful if a *broken* rehydration is *detected*. The
reference test suite therefore pairs every positive equivalence test with an
adversarial negative test that corrupts a boundary field (§4.3) and asserts the
resumed continuation **diverges**.

## 6. Versioning

The `hss_version` metadata key carries the spec version. v0.1 is pre-alpha: the
byte layout (§1–§3) is stable, but the metadata vocabulary (§4) may grow.
Readers SHOULD ignore unknown metadata keys and unknown tensor names rather than
fail, except where §4.3 requires a value to be well-formed.
