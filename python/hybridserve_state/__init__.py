"""hybridserve-state: engine-agnostic on-disk interchange for hybrid
SSM+Attention LLM inference state, with a machine-checked rehydration-equivalence
contract.

Public API:
    save / load / inspect_file   -- NumPy <-> .hss container I/O
    roles                        -- per-layer ROLE vocabulary for hybrid state
    verify                       -- rehydration-equivalence harness (CPU, bitwise)
    adapters.fla                 -- flash-linear-attention Cache <-> .hss mapping

See ``hss-spec/hss-spec.md`` for the container specification and the precise
CLAIM / NON-CLAIM boundary.
"""

from __future__ import annotations

from ._core import __hss_version__
from .io import save, load, inspect_file, supported_dtypes

__version__ = "0.1.0a1"

__all__ = [
    "save",
    "load",
    "inspect_file",
    "supported_dtypes",
    "__hss_version__",
    "__version__",
]
