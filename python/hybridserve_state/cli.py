"""``hss`` command-line tool: inspect, verify, diff, and self-check containers."""

from __future__ import annotations

import argparse
import sys

from . import io as hss_io
from . import roles


def _cmd_inspect(args: argparse.Namespace) -> int:
    metadata, tensors = hss_io.inspect_file(args.file)
    print(f"# {args.file}")
    print(f"## metadata ({len(metadata)} keys)")
    for k in sorted(metadata):
        print(f"  {k} = {metadata[k]}")
    print(f"## tensors ({len(tensors)})")
    for name, dtype, shape, nbytes in tensors:
        shp = "x".join(str(d) for d in shape) if shape else "scalar"
        print(f"  {name:<32} {dtype:<5} [{shp}] {nbytes} bytes")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    metadata, tensors = hss_io.inspect_file(args.file)
    try:
        roles.validate_metadata(metadata)
    except roles.SemanticError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        f"OK: {args.file} parses and metadata invariants hold "
        f"({len(tensors)} tensors, {len(metadata)} metadata keys)"
    )
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    a_state, a_meta = hss_io.load(args.a)
    b_state, b_meta = hss_io.load(args.b)
    differences: list[str] = []

    a_names, b_names = set(a_state), set(b_state)
    for name in sorted(a_names - b_names):
        differences.append(f"- only in {args.a}: {name}")
    for name in sorted(b_names - a_names):
        differences.append(f"- only in {args.b}: {name}")
    for name in sorted(a_names & b_names):
        x, y = a_state[name], b_state[name]
        if x.dtype != y.dtype or x.shape != y.shape:
            differences.append(f"- {name}: {x.dtype}{x.shape} vs {y.dtype}{y.shape}")
        elif x.tobytes() != y.tobytes():
            differences.append(f"- {name}: byte-level mismatch")

    for key in sorted(set(a_meta) | set(b_meta)):
        if a_meta.get(key) != b_meta.get(key):
            differences.append(
                f"- metadata {key}: {a_meta.get(key)!r} vs {b_meta.get(key)!r}"
            )

    if differences:
        print(f"DIFFER ({len(differences)}):")
        for d in differences:
            print(d)
        return 1
    print("IDENTICAL")
    return 0


def _cmd_selfcheck(args: argparse.Namespace) -> int:
    # Lazy import: the equivalence harness pulls in the reference layers.
    from . import verify

    report = verify.run_selfcheck(seed=args.seed)
    print(report.render())
    return 0 if report.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hss", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="show container structure")
    p_inspect.add_argument("file")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_verify = sub.add_parser("verify", help="parse + check metadata invariants")
    p_verify.add_argument("file")
    p_verify.set_defaults(func=_cmd_verify)

    p_diff = sub.add_parser("diff", help="compare two containers")
    p_diff.add_argument("a")
    p_diff.add_argument("b")
    p_diff.set_defaults(func=_cmd_diff)

    p_self = sub.add_parser(
        "selfcheck",
        help="run the bitwise rehydration-equivalence self-check on reference layers",
    )
    p_self.add_argument("--seed", type=int, default=0)
    p_self.set_defaults(func=_cmd_selfcheck)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
