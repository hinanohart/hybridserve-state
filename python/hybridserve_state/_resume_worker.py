"""Subprocess entry point: rehydrate a reference-model state from a ``.hss``
file in a *fresh process* and resume generation.

Invoked as ``python -m hybridserve_state._resume_worker <hss> <n> <out.npz>``.
This is what makes the rehydration-equivalence claim cross-process rather than
merely in-memory: the comparison run shares no Python state with the original.
"""

from __future__ import annotations

import sys

import numpy as np

from .reference import ReferenceModel


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3:
        print("usage: _resume_worker <hss_path> <n_tokens> <out_npz>", file=sys.stderr)
        return 2
    hss_path, n_str, out_path = argv
    n = int(n_str)

    model = ReferenceModel.restore(hss_path)
    result = model.gen_steps(n)

    tokens = np.array([t for t, _ in result], dtype=np.int64)
    if result:
        logits = np.stack([lg for _, lg in result])
    else:
        logits = np.zeros((0, model.cfg.vocab), dtype=np.float64)
    np.savez(out_path, tokens=tokens, logits=logits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
