from asetui.cli import build_parser


def test_parser_accepts_multiple_inputs_with_optional_ase_slice() -> None:
    parser = build_parser()

    parsed = parser.parse_args(["sample.xyz", "traj.json@-1", "frames.xyz@:3"])

    assert parsed.inputs == ["sample.xyz", "traj.json@-1", "frames.xyz@:3"]
