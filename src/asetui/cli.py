from __future__ import annotations

import argparse
from pathlib import Path

from asetui.app import AppState, run_app
from asetui.io import read_atoms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atui",
        description="Interactive terminal viewer for ASE-supported structure files.",
    )
    parser.add_argument("input", type=Path, help="Path to a structure file, such as XYZ or CIF.")
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Start with element symbols instead of single-character atom markers.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        atoms = read_atoms(args.input)
    except Exception as exc:  # pragma: no cover - thin CLI wrapper
        parser.exit(1, f"atui: {exc}\n")

    return run_app(atoms, AppState(show_labels=args.labels))


if __name__ == "__main__":
    raise SystemExit(main())
