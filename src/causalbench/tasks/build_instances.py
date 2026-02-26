from __future__ import annotations

import uuid
from collections.abc import Sequence

import numpy as np

from causalbench.scm.generate import ALL_SCM_KINDS, SCM_DESCRIPTIONS, make_scm
from causalbench.scm.intervene import compare_obs_vs_do, estimate_obs_prob
from causalbench.scm.simulate import sample_observational
from causalbench.tasks.instances import Instance

LABELS: tuple[str, ...] = ("obs_gt_do", "do_gt_obs", "approx_equal")
DEFAULT_SCM_KINDS: tuple[str, ...] = ("confounding",)


def parse_scm_kinds(spec: str) -> tuple[str, ...]:
    raw = spec.strip().lower()
    if not raw:
        return DEFAULT_SCM_KINDS
    if raw == "all":
        return ALL_SCM_KINDS

    kinds = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not kinds:
        return DEFAULT_SCM_KINDS

    unknown = [k for k in kinds if k not in ALL_SCM_KINDS]
    if unknown:
        raise ValueError(f"unknown scm kind(s): {', '.join(unknown)}")
    return kinds


def _target_label_counts(n: int) -> dict[str, int]:
    base = n // len(LABELS)
    remainder = n % len(LABELS)
    counts = {label: base for label in LABELS}
    for i in range(remainder):
        counts[LABELS[i]] += 1
    return counts


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _fmt_stat(value: float) -> str:
    return "nan" if np.isnan(value) else f"{value:.4f}"


def _build_prompt(
    scm_kind: str,
    obs_data: dict[str, np.ndarray],
    n_obs: int,
    x_value: float,
    x_band: float,
) -> str:
    x = obs_data["X"]
    y = obs_data["Y"]
    in_band = np.abs(x - x_value) <= x_band
    n_in_band = int(in_band.sum())

    obs_prob_est = estimate_obs_prob(obs_data, x_value=x_value, band=x_band)
    corr_xy = _safe_corr(x, y)

    observed_nodes = sorted(node for node in obs_data if node != "U")
    extra_lines = [f"- mean({node}): {_fmt_stat(float(np.mean(obs_data[node])))}" for node in observed_nodes]
    extra_stats = "\n".join(extra_lines)

    return (
        f"Task: intervention_compare_{scm_kind}\n"
        f"Causal DAG: {SCM_DESCRIPTIONS[scm_kind]}\n"
        "We care about P(Y > 0).\n\n"
        "From N observational samples, you are given these empirical summaries:\n"
        f"- N = {n_obs}\n"
        f"- Estimated A = P(Y > 0 | X ~ {x_value:.1f}) using band |X-{x_value:.1f}|<={x_band:.1f}: {_fmt_stat(obs_prob_est)}\n"
        f"- Count in band: {n_in_band}\n"
        f"- corr(X, Y): {_fmt_stat(corr_xy)}\n"
        f"{extra_stats}\n\n"
        "Now decide which is larger:\n"
        f"A = P(Y > 0 | X ~ {x_value:.1f})  (observational)\n"
        f"B = P(Y > 0 | do(X = {x_value:.1f})) (interventional)\n\n"
        'Return ONLY JSON: {"label":"obs_gt_do"} or {"label":"do_gt_obs"} or {"label":"approx_equal"}.\n'
    )


def build_intervention_compare_instances(
    n: int = 25,
    seed: int = 0,
    scm_kinds: Sequence[str] | None = None,
    balance_labels: bool = True,
    max_attempt_multiplier: int = 200,
    n_obs_samples: int = 8000,
    n_mc_samples: int = 8000,
    tol: float = 0.02,
    do_value: float = 1.0,
    x_band: float = 0.1,
) -> list[Instance]:
    if n <= 0:
        return []

    kinds = tuple(scm_kinds) if scm_kinds is not None else DEFAULT_SCM_KINDS
    if not kinds:
        raise ValueError("scm_kinds must contain at least one motif")

    unknown = [k for k in kinds if k not in ALL_SCM_KINDS]
    if unknown:
        raise ValueError(f"unknown scm kind(s): {', '.join(unknown)}")

    target_counts = _target_label_counts(n)
    accepted_counts = {label: 0 for label in LABELS}

    instances: list[Instance] = []
    max_attempts = max(10, n * max_attempt_multiplier)
    attempt = 0
    while len(instances) < n and attempt < max_attempts:
        scm_kind = kinds[attempt % len(kinds)]
        attempt_seed = seed + attempt

        scm = make_scm(kind=scm_kind, seed=attempt_seed)
        obs_data = sample_observational(
            scm=scm,
            n=n_obs_samples,
            seed=seed + 100_000 + attempt,
            interventions=None,
        )
        out = compare_obs_vs_do(
            scm=scm,
            do={"X": do_value},
            n_obs=n_obs_samples,
            n_mc=n_mc_samples,
            seed=seed + 200_000 + attempt,
            tol=tol,
            obs_data=obs_data,
            x_band=x_band,
        )
        label = str(out["label"])
        obs_prob = out["obs_prob"]
        do_prob = out["do_prob"]
        if not isinstance(obs_prob, float) or not isinstance(do_prob, float):
            raise TypeError("compare_obs_vs_do returned non-float probabilities")

        if balance_labels and accepted_counts[label] >= target_counts[label]:
            attempt += 1
            continue

        prompt = _build_prompt(
            scm_kind=scm_kind,
            obs_data=obs_data,
            n_obs=n_obs_samples,
            x_value=do_value,
            x_band=x_band,
        )
        inst_key = f"{seed}:{attempt}:{scm_kind}:{len(instances)}"
        instances.append(
            Instance(
                instance_id=str(uuid.uuid5(uuid.NAMESPACE_URL, inst_key)),
                task=f"intervention_compare_{scm_kind}",
                scm_kind=scm_kind,
                prompt=prompt,
                gold={
                    "label": label,
                    "obs_prob": obs_prob,
                    "do_prob": do_prob,
                },
            )
        )
        accepted_counts[label] += 1
        attempt += 1

    if len(instances) < n:
        msg = (
            f"could not build {n} instances with requested balance from scm kinds {kinds}. "
            f"accepted={len(instances)} label_counts={accepted_counts} target={target_counts}"
        )
        raise RuntimeError(msg)

    return instances
