import pytest

from FRW.metrics import MetricsAccumulator, print_metrics
from bgs.models import Bounds, Conductor
from bgs.topology import find_touching_nets


class _MasterVGSS:
    def __init__(self, net):
        self.net = net


def _conductor(name: str, bounds: Bounds) -> Conductor:
    return Conductor(
        name=name,
        bounds=bounds,
        face_vertices=bounds.vertices(),
    )


def test_metrics_accumulator_uses_compact_hop_counts() -> None:
    conductors = [
        _conductor("C1", Bounds(0, 1, 0, 1, 0, 1)),
        _conductor("C2", Bounds(3, 4, 0, 1, 0, 1)),
    ]
    master_vgss = _MasterVGSS(find_touching_nets(conductors)[0])
    accumulator = MetricsAccumulator(conductors, master_vgss)

    accumulator.update(
        {
            "termination": "hit_conductor",
            "hit_conductor": "C2",
            "hop_count": 7,
        }
    )
    accumulator.update(
        {
            "termination": "escaped",
            "hit_conductor": "Infinity (Ground)",
            "hop_count": 3,
        }
    )
    result = accumulator.finalize()

    assert result["net_hit_counts"] == {"Net2": 1}
    assert result["termination_counts"]["escaped"] == 1
    assert result["hop_statistics"]["mean"] == pytest.approx(5.0)


def test_print_metrics_outputs_summary(capsys) -> None:
    metrics = {
        "termination_counts": {"hit_conductor": 3, "escaped": 1, "max_hops": 2},
        "self_net_hits": 3,
        "other_net_hits": 1,
        "hop_statistics": {"mean": 4.5, "max": 8},
    }

    print_metrics(metrics)

    captured = capsys.readouterr().out
    assert "FRW Metrics" in captured
    assert "termination_counts:" in captured
    assert "hit_conductor: 3" in captured
    assert "self_net_hits:" in captured
    assert "3" in captured
