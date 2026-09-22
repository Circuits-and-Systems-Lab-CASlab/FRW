from types import SimpleNamespace

from FRW.cli import _build_parser, _select_master_vgss


def test_select_master_vgss_by_name() -> None:
    all_net_vgss = [
        SimpleNamespace(net=SimpleNamespace(name="C1")),
        SimpleNamespace(net=SimpleNamespace(name="C2")),
    ]

    selected = _select_master_vgss(all_net_vgss, "C2")

    assert selected is all_net_vgss[1]


def test_cli_accepts_a_mixed_finite_window() -> None:
    args = _build_parser().parse_args(
        [
            "input.m",
            "--window",
            "-10",
            "30",
            "-10",
            "15",
            "0",
            "3",
            "--neumann-axes",
            "z",
        ]
    )

    assert args.window == [-10.0, 30.0, -10.0, 15.0, 0.0, 3.0]
    assert args.neumann_axes == ["z"]
