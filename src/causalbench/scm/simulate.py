from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from causalbench.scm.generate import LinearGaussianSCM



def sample_observational(
    scm: LinearGaussianSCM,
    n: int,
    seed: int = 0,
    interventions: Optional[Dict[str, float]] = None
) -> Dict[str, np.ndarray]:
    """
    Sample n points from a LinearGaussianSCM.

    interventions implements the do-operator:
      interventions={"X": 1.0} forces X to be constant 1.0 for all samples.

    Returns: dict[node] -> np.ndarray (shape: (n,))
    """

    rng = np.random.default_rng(seed)
    interventions = interventions or {}

    order = scm.dag.topological_sort()

    data : Dict[str, np.ndarray] = {node: np.zeros(n,dtype=float) for node in order}
    for node in order:
        if node in interventions:
            data[node][:] =float(interventions[node])
            continue
        base = np.zeros(n,dtype=float)
        for parent in scm.dag.parents_of(node):
            w = scm.weights[(parent,node)]
            base += w*data[parent]

        std = scm.noise_std[node]
        noise = rng.normal(loc=0.0,size=n,scale=std)
        data[node] = base + noise
    return data



    
    
