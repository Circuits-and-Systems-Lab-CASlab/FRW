from FRW.metrics import print_metrics


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
