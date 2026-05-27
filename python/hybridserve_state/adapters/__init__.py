"""Engine adapters mapping a framework's in-memory cache to/from ``.hss``.

* :mod:`hybridserve_state.adapters.fla` — flash-linear-attention ``Cache``
  structure (CLAIM: structural field mapping, unit-tested against the documented
  schema with synthetic Cache objects).

vLLM / SGLang adapters are experimental and live in
``hybridserve_state._experimental`` (excluded from CI, deferred to v0.1.1).
"""
