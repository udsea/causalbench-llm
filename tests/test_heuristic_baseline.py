from __future__ import annotations

from causalbench.eval.heuristic_baseline import heuristic_predict_from_features


def test_heuristic_predict_directional_cases():
    assert (
        heuristic_predict_from_features(delta_a_baseline=-0.2, ci_width=0.05, n_in_band=120)
        == "do_gt_obs"
    )
    assert (
        heuristic_predict_from_features(delta_a_baseline=0.2, ci_width=0.05, n_in_band=120)
        == "obs_gt_do"
    )


def test_heuristic_predict_approx_equal_with_weak_evidence():
    assert (
        heuristic_predict_from_features(delta_a_baseline=0.01, ci_width=0.05, n_in_band=120)
        == "approx_equal"
    )
    assert (
        heuristic_predict_from_features(delta_a_baseline=-0.2, ci_width=0.4, n_in_band=120)
        == "approx_equal"
    )
    assert (
        heuristic_predict_from_features(delta_a_baseline=-0.2, ci_width=0.05, n_in_band=10)
        == "approx_equal"
    )
