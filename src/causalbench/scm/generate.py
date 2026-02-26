from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from causalbench.scm.dag import DAG

SCM_EDGES: dict[str, tuple[tuple[str, str], ...]] = {
    "confounding": (("U", "X"), ("U", "Y"), ("X", "Y")),
    "confounding_only": (("U", "X"), ("U", "Y")),
    "no_confounding": (("X", "Y"),),
    "mediation": (("X", "M"), ("M", "Y")),
    "collider": (("X", "Y"), ("X", "Z"), ("Y", "Z")),
    "instrumental_variable": (("Z", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")),
    "anti_causal": (("Y", "X"), ("U", "X"), ("U", "Y")),
    "backdoor_adjustable": (("W", "X"), ("W", "Y"), ("X", "Y")),
}

SCM_UNOBSERVED: dict[str, tuple[str, ...]] = {
    "confounding": ("U",),
    "confounding_only": ("U",),
    "no_confounding": (),
    "mediation": (),
    "collider": (),
    "instrumental_variable": ("U",),
    "anti_causal": ("U",),
    "backdoor_adjustable": (),
}


@dataclass(frozen=True)
class LinearGaussianSCM:
    dag: DAG
    weights: dict[tuple[str, str], float]
    noise_std: dict[str, float]


def _build_scm(edges: list[tuple[str, str]], seed: int) -> LinearGaussianSCM:
    rng = np.random.default_rng(seed)
    dag = DAG.from_edges(edges)

    weights: dict[tuple[str, str], float] = {}
    for edge in edges:
        weights[edge] = float(rng.uniform(-2.0, 2.0))

    noise_std: dict[str, float] = {}
    for node in dag.nodes():
        noise_std[node] = float(rng.uniform(0.5, 1.5))

    return LinearGaussianSCM(dag=dag, weights=weights, noise_std=noise_std)


def make_confounding_scm(seed: int = 0) -> LinearGaussianSCM:
    """
    Unobserved confounding motif:
      U -> X
      U -> Y
      X -> Y
    """
    return _build_scm(edges=list(SCM_EDGES["confounding"]), seed=seed)


def make_confounding_only_scm(seed: int = 0) -> LinearGaussianSCM:
    """
    Confounding-only motif:
      U -> X
      U -> Y
    """
    return _build_scm(edges=list(SCM_EDGES["confounding_only"]), seed=seed)


def make_no_confounding_scm(seed: int = 0) -> LinearGaussianSCM:
    """
    No-confounding motif:
      X -> Y
    """
    return _build_scm(edges=list(SCM_EDGES["no_confounding"]), seed=seed)


def make_mediation_scm(seed: int = 0) -> LinearGaussianSCM:
    """
    Mediation motif:
      X -> M -> Y
    """
    return _build_scm(edges=list(SCM_EDGES["mediation"]), seed=seed)


def make_collider_scm(seed: int = 0) -> LinearGaussianSCM:
    """
    Collider / selection motif:
      X -> Z <- Y
      X -> Y
    """
    return _build_scm(edges=list(SCM_EDGES["collider"]), seed=seed)


def make_instrumental_variable_scm(seed: int = 0) -> LinearGaussianSCM:
    """
    Instrumental-variable motif:
      Z -> X -> Y
      U -> X
      U -> Y
    """
    return _build_scm(edges=list(SCM_EDGES["instrumental_variable"]), seed=seed)


def make_anti_causal_scm(seed: int = 0) -> LinearGaussianSCM:
    """
    Anti-causal motif:
      Y -> X
      U -> X
      U -> Y
    """
    return _build_scm(edges=list(SCM_EDGES["anti_causal"]), seed=seed)


def make_backdoor_adjustable_scm(seed: int = 0) -> LinearGaussianSCM:
    """
    Backdoor-adjustable confounding motif with observed W:
      W -> X
      W -> Y
      X -> Y
    """
    return _build_scm(edges=list(SCM_EDGES["backdoor_adjustable"]), seed=seed)


SCM_BUILDERS: dict[str, Callable[[int], LinearGaussianSCM]] = {
    "confounding": make_confounding_scm,
    "confounding_only": make_confounding_only_scm,
    "no_confounding": make_no_confounding_scm,
    "mediation": make_mediation_scm,
    "collider": make_collider_scm,
    "instrumental_variable": make_instrumental_variable_scm,
    "anti_causal": make_anti_causal_scm,
    "backdoor_adjustable": make_backdoor_adjustable_scm,
}

SCM_DESCRIPTIONS: dict[str, str] = {
    "confounding": "U -> X, U -> Y, X -> Y. U is unobserved.",
    "confounding_only": "U -> X, U -> Y. U is unobserved.",
    "no_confounding": "X -> Y with no shared confounder.",
    "mediation": "X -> M -> Y.",
    "collider": "X -> Y, X -> Z, Y -> Z (collider at Z).",
    "instrumental_variable": "Z -> X -> Y, U -> X, U -> Y, with Z independent of U.",
    "anti_causal": "Y -> X, U -> X, U -> Y.",
    "backdoor_adjustable": "W -> X, W -> Y, X -> Y. W is observed and backdoor-adjustable.",
}

ALL_SCM_KINDS: tuple[str, ...] = tuple(SCM_BUILDERS.keys())


def make_scm(kind: str, seed: int = 0) -> LinearGaussianSCM:
    try:
        return SCM_BUILDERS[kind](seed)
    except KeyError as exc:
        raise ValueError(f"unknown scm kind: {kind}") from exc
