from types import SimpleNamespace

from FRW.cli import _select_master_vgss


def test_select_master_vgss_by_name() -> None:
    all_net_vgss = [
        SimpleNamespace(net=SimpleNamespace(name="C1")),
        SimpleNamespace(net=SimpleNamespace(name="C2")),
    ]

    selected = _select_master_vgss(all_net_vgss, "C2")

    assert selected is all_net_vgss[1]
