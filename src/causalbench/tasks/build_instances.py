from __future__ import annotations

import uuid
from collections.abc import Sequence

import numpy as np

from causalbench.scm.generate import (
    ALL_SCM_KINDS,
    SCM_DESCRIPTIONS,
    SCM_EDGES,
    SCM_UNOBSERVED,
    make_scm,
)
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


def _target_motif_label_counts(
    n: int,
    kinds: Sequence[str],
) -> dict[tuple[str, str], int]:
    bucket_keys = [(kind, label) for kind in kinds for label in LABELS]
    base = n // len(bucket_keys)
    remainder = n % len(bucket_keys)
    counts = {key: base for key in bucket_keys}
    for i in range(remainder):
        counts[bucket_keys[i]] += 1
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
    label_order: tuple[str, str, str],
) -> str:
    x = obs_data["X"]
    y = obs_data["Y"]
    var_x = float(np.var(x))
    cov_xy = float(np.cov(x, y, ddof=0)[0, 1]) if x.size > 1 and y.size > 1 else 0.0
    beta_hat = cov_xy / var_x if var_x > 0.0 else 0.0

    baseline_prob = float(np.mean(y > 0))
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    std_x = float(np.std(x))
    std_y = float(np.std(y))
    in_band = np.abs(x - x_value) <= x_band
    n_in_band = int(in_band.sum())

    obs_prob_est = estimate_obs_prob(obs_data, x_value=x_value, band=x_band)
    delta_a_baseline = (
        float(obs_prob_est - baseline_prob) if not np.isnan(obs_prob_est) else float("nan")
    )
    if n_in_band > 0 and not np.isnan(obs_prob_est):
        se = float(np.sqrt(obs_prob_est * (1.0 - obs_prob_est) / n_in_band))
        ci_low = float(max(0.0, obs_prob_est - 1.96 * se))
        ci_high = float(min(1.0, obs_prob_est + 1.96 * se))
    else:
        ci_low = float("nan")
        ci_high = float("nan")
    corr_xy = _safe_corr(x, y)

    edges = ", ".join(f"{src}->{dst}" for src, dst in SCM_EDGES[scm_kind])
    unobserved_nodes = SCM_UNOBSERVED[scm_kind]
    unobserved_text = ", ".join(unobserved_nodes) if unobserved_nodes else "none"

    observed_nodes = sorted(node for node in obs_data if node not in set(unobserved_nodes))
    extra_lines = [
        f"- mean({node}): {_fmt_stat(float(np.mean(obs_data[node])))}"
        for node in observed_nodes
        if node not in {"X", "Y"}
    ]
    extra_stats = "\n".join(extra_lines)

    allowed_json = " or ".join(f'{{"label":"{label}"}}' for label in label_order)

    return (
        f"Task: intervention_compare_{scm_kind}\n"
        f"Causal DAG edges: {edges}\n"
        f"Unobserved variables: {unobserved_text}\n"
        f"Motif note: {SCM_DESCRIPTIONS[scm_kind]}\n"
        "We care about P(Y > 0).\n\n"
        "From N observational samples, you are given these empirical summaries:\n"
        f"- N = {n_obs}\n"
        f"- Conditioning band: |X-{x_value:.1f}| <= {x_band:.1f}\n"
        f"- Estimated A = P(Y > 0 | X ~ {x_value:.1f}) using band |X-{x_value:.1f}|<={x_band:.1f}: {_fmt_stat(obs_prob_est)}\n"
        f"- Approx 95% CI for A_hat: [{_fmt_stat(ci_low)}, {_fmt_stat(ci_high)}]\n"
        f"- delta(A_hat - baseline): {_fmt_stat(delta_a_baseline)}\n"
        f"- Count in band: {n_in_band}\n"
        f"- Baseline P(Y > 0): {_fmt_stat(baseline_prob)}\n"
        f"- mean(X): {_fmt_stat(mean_x)}\n"
        f"- mean(Y): {_fmt_stat(mean_y)}\n"
        f"- std(X): {_fmt_stat(std_x)}\n"
        f"- std(Y): {_fmt_stat(std_y)}\n"
        f"- beta_hat (OLS slope of Y on X): {_fmt_stat(beta_hat)}\n"
        f"- corr(X, Y): {_fmt_stat(corr_xy)}\n"
        f"{extra_stats}\n\n"
        "Now decide which is larger:\n"
        f"A = P(Y > 0 | X ~ {x_value:.1f})  (observational)\n"
        f"B = P(Y > 0 | do(X = {x_value:.1f})) (interventional)\n\n"
        "Label mapping:\n"
        "- if B > A, return do_gt_obs\n"
        "- if A > B, return obs_gt_do\n"
        "- if A and B are close, return approx_equal\n"
        "Practical rubric for this benchmark:\n"
        "- when delta(A_hat - baseline) <= -0.08 with a tight CI, B is often larger (lean do_gt_obs)\n"
        "- when delta(A_hat - baseline) >= 0.08 with a tight CI, A is often larger (lean obs_gt_do)\n"
        "- when delta is near 0 or CI is wide, lean approx_equal\n"
        "Heuristic: if A_hat is very close to baseline P(Y > 0) and evidence is weak, prefer approx_equal.\n"
        f"Return ONLY JSON: {allowed_json}.\n"
    )


def _assign_label_with_margins(
    obs_prob: float,
    do_prob: float,
    eq_margin: float,
    dir_margin: float,
) -> str | None:
    diff = obs_prob - do_prob
    gap = abs(diff)
    if gap < eq_margin:
        return "approx_equal"
    if diff > dir_margin:
        return "obs_gt_do"
    if -diff > dir_margin:
        return "do_gt_obs"
    return None


def build_intervention_compare_instances(
    n: int = 25,
    seed: int = 0,
    scm_kinds: Sequence[str] | None = None,
    balance_labels: bool = True,
    max_attempt_multiplier: int = 200,
    n_prompt_obs_samples: int = 2000,
    n_obs_samples: int = 8000,
    n_mc_samples: int = 8000,
    tol: float = 0.02,
    eq_margin: float = 0.06,
    dir_margin: float = 0.06,
    discard_ambiguous: bool = True,
    stratify_motif_label: bool = False,
    do_value: float = 1.0,
    x_band: float = 0.25,
) -> list[Instance]:
    if n <= 0:
        return []

    kinds = tuple(scm_kinds) if scm_kinds is not None else DEFAULT_SCM_KINDS
    if not kinds:
        raise ValueError("scm_kinds must contain at least one motif")

    unknown = [k for k in kinds if k not in ALL_SCM_KINDS]
    if unknown:
        raise ValueError(f"unknown scm kind(s): {', '.join(unknown)}")

    if stratify_motif_label and len(kinds) == 0:
        raise ValueError("stratify_motif_label requires at least one motif")

    target_counts = _target_label_counts(n)
    accepted_counts = {label: 0 for label in LABELS}
    target_motif_label_counts = _target_motif_label_counts(n, kinds)
    accepted_motif_label_counts = {
        key: 0
        for key in target_motif_label_counts
    }

    instances: list[Instance] = []
    max_attempts = max(10, n * max_attempt_multiplier)
    if stratify_motif_label:
        max_attempts = max(max_attempts, n * max_attempt_multiplier * len(kinds))
    attempt = 0
    while len(instances) < n and attempt < max_attempts:
        if stratify_motif_label:
            candidate_kinds = [
                kind
                for kind in kinds
                if any(
                    accepted_motif_label_counts[(kind, label)]
                    < target_motif_label_counts[(kind, label)]
                    for label in LABELS
                )
            ]
            if not candidate_kinds:
                break
            scm_kind = candidate_kinds[attempt % len(candidate_kinds)]
        else:
            scm_kind = kinds[attempt % len(kinds)]
        attempt_seed = seed + attempt

        scm = make_scm(kind=scm_kind, seed=attempt_seed)
        gold_obs_data = sample_observational(
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
            obs_data=gold_obs_data,
            x_band=x_band,
        )
        obs_prob = out["obs_prob"]
        do_prob = out["do_prob"]
        if not isinstance(obs_prob, float) or not isinstance(do_prob, float):
            raise TypeError("compare_obs_vs_do returned non-float probabilities")
        gap = abs(obs_prob - do_prob)
        margin_label = _assign_label_with_margins(
            obs_prob=obs_prob,
            do_prob=do_prob,
            eq_margin=eq_margin,
            dir_margin=dir_margin,
        )
        if margin_label is None and discard_ambiguous:
            attempt += 1
            continue
        label = margin_label if margin_label is not None else str(out["label"])

        if balance_labels and not stratify_motif_label and accepted_counts[label] >= target_counts[label]:
            attempt += 1
            continue
        if (
            balance_labels
            and stratify_motif_label
            and accepted_motif_label_counts[(scm_kind, label)]
            >= target_motif_label_counts[(scm_kind, label)]
        ):
            attempt += 1
            continue

        prompt_obs_data = sample_observational(
            scm=scm,
            n=n_prompt_obs_samples,
            seed=seed + 300_000 + attempt,
            interventions=None,
        )
        perm = np.random.default_rng(seed + 400_000 + attempt).permutation(LABELS)
        label_order = (str(perm[0]), str(perm[1]), str(perm[2]))
        prompt = _build_prompt(
            scm_kind=scm_kind,
            obs_data=prompt_obs_data,
            n_obs=n_prompt_obs_samples,
            x_value=do_value,
            x_band=x_band,
            label_order=label_order,
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
                    "gap": gap,
                    "tol": tol,
                    "eq_margin": eq_margin,
                    "dir_margin": dir_margin,
                    "band": x_band,
                    "n_prompt_obs": n_prompt_obs_samples,
                },
            )
        )
        accepted_counts[label] += 1
        accepted_motif_label_counts[(scm_kind, label)] += 1
        attempt += 1

    if len(instances) < n:
        msg = (
            f"could not build {n} instances with requested balance from scm kinds {kinds}. "
            f"accepted={len(instances)} label_counts={accepted_counts} target={target_counts}"
        )
        raise RuntimeError(msg)

    return instances
