import pytest

from src.evaluate import (LatencyBreakdown, leakage_prevention_rate, relative_gain,
                          reversibility_accuracy, reversibility_weighted_relative_gain,
                          utility_retention)


def test_core_evaluation_metrics():
    assert leakage_prevention_rate(["Alice", "123"], "Hello [PERSON]") == 1.0
    assert reversibility_accuracy(["Alice", "123"], "Alice was assigned case 123") == 1.0
    assert utility_retention(0.8, 0.6) == pytest.approx(0.75)
    assert relative_gain(0.8, 1.0, 0.1, 1.0) == pytest.approx(0.7)
    assert reversibility_weighted_relative_gain(0.8, 1.0, 0.1, 1.0, 1.0) == pytest.approx(1.2)


def test_latency_breakdown_tracks_required_stages():
    latency = LatencyBreakdown()
    with latency.measure("detection"):
        pass
    report = latency.complete_report()
    assert report["detection"]["seconds"] >= 0
    assert report["routing"]["seconds"] == 0.0
