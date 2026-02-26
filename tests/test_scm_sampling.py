import numpy as np
from causalbench.scm.generate import make_confounding_scm
from causalbench.scm.simulate import sample_observational

def test_sample_observational_shapes_and_keys():
    scm = make_confounding_scm(seed=0)
    data = sample_observational(scm, n=200, seed=123)
    assert set(["U", "X", "Y"]).issubset(data.keys())
    for v in data.values():
        assert isinstance(v, np.ndarray)
        assert v.shape == (200,)

def test_sample_observational_deterministic_seed():
    scm = make_confounding_scm(seed=0)
    d1 = sample_observational(scm, n=100, seed=999)
    d2 = sample_observational(scm, n=100, seed=999)
    for k in d1:
        assert np.allclose(d1[k], d2[k])

def test_do_operator_overrides_node():
    scm = make_confounding_scm(seed=0)
    data = sample_observational(scm, n=50, seed=7, interventions={"X": 1.0})
    assert np.allclose(data["X"], 1.0)