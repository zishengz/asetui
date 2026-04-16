from asetui.cli import build_parser


def test_parser_defaults() -> None:
    args = build_parser().parse_args(["sample.xyz"])

    assert str(args.input) == "sample.xyz"
    assert args.labels is False
