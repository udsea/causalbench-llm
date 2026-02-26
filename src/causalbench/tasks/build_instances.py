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

        N = 8000

        out = compare_obs_vs_do(
            scm,
            do={"X": 1.0},
            n_obs=N,
            n_mc=N,
            seed=seed + 10_000 + i,
            tol=0.02,
        )

        prompt = """Causal DAG: U -> X, U -> Y, X -> Y. U is an unobserved confounder.
            We care about P(Y > 0).

            From N observational samples, you are given these empirical summaries:
            - N = {N}
            - Estimated A = P(Y > 0 | X ~ 1) using band |X-1|<=0.1: {obs_prob_est:.4f}
            - Count in band: {n_in_band}
            - corr(X, Y): {corr_xy:.4f}
            - mean(X): {mean_x:.4f}
            - mean(Y): {mean_y:.4f}

            Now decide which is larger:
            A = P(Y > 0 | X ~ 1)  (observational)
            B = P(Y > 0 | do(X = 1)) (interventional)

            Return ONLY JSON: {{"label":"obs_gt_do"}} or {{"label":"do_gt_obs"}} or {{"label":"approx_equal"}}.
            """.format(
                N=N,
                obs_prob_est=obs_prob_est, # type: ignore
                n_in_band=n_in_band,# type: ignore
                corr_xy=corr_xy,# type: ignore
                mean_x=mean_x,# type: ignore
                mean_y=mean_y,# type: ignore
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