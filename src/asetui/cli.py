from __future__ import annotations

import argparse
from pathlib import Path

from asetui.app import AppState, run_app
from asetui.io import read_all_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atui",
        description="Interactive terminal viewer for ASE-supported structure files.",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a structure file (e.g. structure.xyz). "
             "Append @index for ASE slice notation, e.g. traj.xyz@:10 or traj.xyz@-1.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        frames = read_all_frames(args.input)
    except Exception as exc:  # pragma: no cover - thin CLI wrapper
        parser.exit(1, f"atui: {exc}\n")

    return run_app(frames, AppState())


if __name__ == "__main__":
    raise SystemExit(main())
