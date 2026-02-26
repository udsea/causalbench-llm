from __future__ import annotations
from typing import List
import uuid

from causalbench.scm.generate import make_confounding_scm
from causalbench.scm.intervene import compare_obs_vs_do
from causalbench.tasks.instances import Instance


def build_intervention_compare_instances(n: int = 25, seed: int = 0) -> List[Instance]:
    instances: List[Instance] = []
    for i in range(n):
        scm = make_confounding_scm(seed=seed + i)

        out = compare_obs_vs_do(
            scm,
            do={"X": 1.0},
            n_obs=8000,
            n_mc=8000,
            seed=seed + 10_000 + i,
            tol=0.02,
        )

        prompt = (
            "You are given a causal system with variables U, X, Y.\n"
            "We care about the probability that Y is positive.\n\n"
            "Decide which is larger:\n"
            "A = P(Y > 0 | X ≈ 1)  (observational)\n"
            "B = P(Y > 0 | do(X = 1)) (interventional)\n\n"
            "Return ONLY a JSON object with key 'label' and value one of:\n"
            "  'obs_gt_do', 'do_gt_obs', 'approx_equal'\n"
        )

        instances.append(
            Instance(
                instance_id=str(uuid.uuid4()),
                task="intervention_compare",
                scm_kind="confounding",
                prompt=prompt,
                gold={"label": out["label"], "obs_prob": out["obs_prob"], "do_prob": out["do_prob"]},
            )
        )
    return instances