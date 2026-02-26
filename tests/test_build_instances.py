from __future__ import annotations

from causalbench.scm.generate import ALL_SCM_KINDS, make_scm
from causalbench.tasks.build_instances import build_intervention_compare_instances, parse_scm_kinds


def test_parse_scm_kinds_all_and_custom():
    assert parse_scm_kinds("all") == ALL_SCM_KINDS
    assert parse_scm_kinds("confounding, mediation") == ("confounding", "mediation")


def test_each_scm_kind_contains_x_and_y():
    for kind in ALL_SCM_KINDS:
        scm = make_scm(kind=kind, seed=0)
        assert "X" in scm.dag.nodes()
        assert "Y" in scm.dag.nodes()


def test_balanced_label_sampling_for_confounding():
    instances = build_intervention_compare_instances(
        n=6,
        seed=0,
        scm_kinds=("confounding",),
        balance_labels=True,
        n_obs_samples=1000,
        n_mc_samples=1000,
        tol=0.05,
    )
    counts: dict[str, int] = {}
    for inst in instances:
        counts.setdefault(inst.gold["label"], 0)
        counts[inst.gold["label"]] += 1

    assert len(instances) == 6
    assert counts == {"obs_gt_do": 2, "do_gt_obs": 2, "approx_equal": 2}


def test_multimotif_instances_include_requested_kinds():
    instances = build_intervention_compare_instances(
        n=6,
        seed=0,
        scm_kinds=("confounding", "mediation", "collider"),
        balance_labels=False,
        n_obs_samples=400,
        n_mc_samples=400,
    )
    kinds = {inst.scm_kind for inst in instances}
    assert {"confounding", "mediation", "collider"}.issubset(kinds)
