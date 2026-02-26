from __future__ import annotations
from typing import Dict,Tuple
import numpy as np
from dataclasses import dataclass
from causalbench.scm.dag import DAG

@dataclass(frozen=True)
class LinearGaussianSCM:
    dag: DAG 
    weights: Dict[Tuple[str,str],float]
    noise_std: Dict[str,float]

def make_confounding_scm(seed: int = 0) -> LinearGaussianSCM :
    """
    Confounding motif:
      U -> X
      U -> Y
      X -> Y   (include this; it creates a mix of causal + confounded signal)
    """
    rng = np.random.default_rng(seed)
    edges = [("U", "X"), ("U", "Y"), ("X", "Y")]
    dag = DAG.from_edges(edges)

    # TODO: sample weights for each edge (suggest: uniform(-2, 2))
    weights: Dict[Tuple[str, str], float] = {} 
    for e in edges:
        weights[e] = float(rng.uniform(-2,2))

    # TODO: sample noise std for each node (suggest: uniform(0.5, 1.5))
    noise_std: Dict[str, float] = {}
    for n in dag.nodes():
        noise_std[n] = float(rng.uniform(0.5,1.5))

    return LinearGaussianSCM(dag=dag, weights=weights, noise_std=noise_std)