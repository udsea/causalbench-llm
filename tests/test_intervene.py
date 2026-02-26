import numpy as np
from causalbench.scm.generate import make_confounding_scm
from causalbench.scm.simulate import sample_observational
from causalbench.scm.intervene import estimate_obs_prob, estimate_do_prob, compare_obs_vs_do


def test_estimate_obs_prob_in_range_or_nan():
    scm = make_confounding_scm(seed=0)
    data = sample_observational(scm, n=5000, seed=1)
    p = estimate_obs_prob(data, x_value=1.0, band=0.2)
    assert (np.isnan(p)) or (0.0 <= p <= 1.0)


def test_estimate_do_prob_in_range():
    scm = make_confounding_scm(seed=0)
    p = estimate_do_prob(scm, do={"X": 1.0}, n_mc=5000, seed=2)
    assert 0.0 <= p <= 1.0


def test_compare_obs_vs_do_has_valid_label():
    scm = make_confounding_scm(seed=0)
    out = compare_obs_vs_do(scm, do={"X": 1.0}, n_obs=8000, n_mc=8000, seed=3)
    assert "obs_prob" in out and "do_prob" in out and "label" in out
    assert out["label"] in {"obs_gt_do", "do_gt_obs", "approx_equal"}