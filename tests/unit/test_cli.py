from mody_analyse.cli import build_arg_parser


def test_cli_parses_pilot_and_all_modes():
    parser = build_arg_parser()
    args_pilot = parser.parse_args(["--pilot", "--stage", "kbi"])
    assert args_pilot.pilot is True
    assert args_pilot.stage == "kbi"

    args_all = parser.parse_args(["--all"])
    assert args_all.all is True
    assert args_all.stage == "all"
