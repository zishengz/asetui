from asetui.cli import build_parser


def test_parser_accepts_input_path_with_optional_ase_slice() -> None:
    parser = build_parser()

    plain = parser.parse_args(["sample.xyz"])
    sliced = parser.parse_args(["traj.xyz@-1"])

    assert plain.input == "sample.xyz"
    assert sliced.input == "traj.xyz@-1"
