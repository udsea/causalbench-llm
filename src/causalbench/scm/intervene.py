from __future__ import annotations
from typing import Dict, Literal
import numpy as np

from causalbench.scm.generate import LinearGaussianSCM
from causalbench.scm.simulate import sample_observational

Label = Literal["obs_gt_do", "do_gt_obs", "approx_equal"]


def estimate_obs_prob(
    data: Dict[str, np.ndarray],
    x_node: str = "X",
    y_node: str = "Y",
    x_value: float = 1.0,
    band: float = 0.1,
) -> float:
    """
    Estimate P(Y > 0 | X approximately x_value) from observational samples.

    We approximate conditioning on a continuous variable using a band:
        |X - x_value| <= band

    Returns:
      - float in [0, 1] if there are samples in the band
      - np.nan if the band is empty
    """
    X = data[x_node]
    Y = data[y_node]

    mask = np.abs(X - x_value) <= band
    count = int(mask.sum())
    if count == 0:
        return float("nan")

    return float(np.mean(Y[mask] > 0))


def estimate_do_prob(
    scm: LinearGaussianSCM,
    do: Dict[str, float],
    n_mc: int = 5000,
    seed: int = 0,
    y_node: str = "Y",
) -> float:
    """
    Estimate P(Y > 0 | do(...)) via Monte Carlo by sampling with interventions.
    """
    data = sample_observational(scm, n=n_mc, seed=seed, interventions=do)
    Y = data[y_node]
    return float(np.mean(Y > 0))


def compare_obs_vs_do(
    scm: LinearGaussianSCM,
    do: Dict[str, float],
    n_obs: int = 20000,
    n_mc: int = 20000,
    seed: int = 0,
    tol: float = 0.02,
) -> Dict[str, object]:
    """
    Produce both probabilities and a label:
      - obs_prob = P(Y>0 | X≈x_value) from observational sampling
      - do_prob  = P(Y>0 | do(X=x_value)) from interventional sampling

    Label logic:
      - if abs(obs - do) <= tol => approx_equal
      - else => obs_gt_do or do_gt_obs
    """
    if len(do) != 1:
        raise ValueError("compare_obs_vs_do expects do to contain exactly one intervention, e.g. {'X': 1.0}")

    x_node, x_value = next(iter(do.items()))

    # Observational samples
    obs_data = sample_observational(scm, n=n_obs, seed=seed, interventions=None)
    obs_prob = estimate_obs_prob(obs_data, x_node=x_node, x_value=float(x_value), band=0.1)

    # Interventional samples
    do_prob = estimate_do_prob(scm, do=do, n_mc=n_mc, seed=seed + 1)

    # If obs_prob is NaN because band was empty, widen the band once (pragmatic v0 fix)
    if np.isnan(obs_prob):
        obs_prob = estimate_obs_prob(obs_data, x_node=x_node, x_value=float(x_value), band=0.2)

    # Decide label
    if np.isnan(obs_prob):
        # Still NaN? fall back to approx_equal to avoid crashing benchmarks.
        label: Label = "approx_equal"
    else:
        diff = float(obs_prob - do_prob)
        if abs(diff) <= tol:
            label = "approx_equal"
        elif diff > 0:
            label = "obs_gt_do"
        else:
            label = "do_gt_obs"

    return {"obs_prob": obs_prob, "do_prob": do_prob, "label": label}