from __future__ import annotations

import json

from causalbench.eval.summarize import main


def test_summarize_includes_macro_and_confusion(tmp_path):
    results = [
        {
            "score": 1,
            "parse_ok": True,
            "pred": {"label": "obs_gt_do"},
            "scm_kind": "confounding",
            "gold": {"label": "obs_gt_do", "obs_prob": 0.9, "do_prob": 0.7, "gap": 0.2, "tol": 0.02},
        },
        {
            "score": 1,
            "parse_ok": True,
            "pred": {"label": "do_gt_obs"},
            "scm_kind": "confounding_only",
            "gold": {"label": "do_gt_obs", "obs_prob": 0.4, "do_prob": 0.5, "gap": 0.1, "tol": 0.02},
        },
        {
            "score": 0,
            "parse_ok": True,
            "pred": {"label": "do_gt_obs"},
            "scm_kind": "no_confounding",
            "gold": {"label": "approx_equal", "obs_prob": 0.49, "do_prob": 0.5, "gap": 0.01, "tol": 0.02},
        },
        {
            "score": 0,
            "parse_ok": False,
            "pred": None,
            "scm_kind": "mediation",
            "gold": {"label": "approx_equal", "obs_prob": 0.1, "do_prob": 0.2, "gap": 0.1, "tol": 0.02},
        },
    ]

    in_file = tmp_path / "results.jsonl"
    out_file = tmp_path / "results_table.md"
    in_file.write_text("\n".join(json.dumps(row) for row in results) + "\n", encoding="utf-8")

    main(results_jsonl=str(in_file), out_table=str(out_file), gap_tol=0.02)
    report = out_file.read_text(encoding="utf-8")

    assert "- Parse rate: 0.750" in report
    assert "- Accuracy: 0.500" in report
    assert "- Macro accuracy: 0.667" in report
    assert "- Macro F1: 0.556" in report
    assert "## Breakdown by gap difficulty" in report
    assert "## Confusion matrix (gold x pred)" in report
    assert "| obs_gt_do | 1 | 0 | 0 |" in report
