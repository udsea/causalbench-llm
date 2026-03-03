from __future__ import annotations

import json

from causalbench.eval.aggregate_reports import main


def _row(gold: str, pred: str, score: int):
    return {
        "score": score,
        "parse_ok": True,
        "pred": {"label": pred},
        "gold": {"label": gold},
    }


def test_aggregate_reports_builds_summary_table(tmp_path):
    root = tmp_path / "runs"
    run_a = root / "model_a"
    run_b = root / "model_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    rows_a = [
        _row("obs_gt_do", "obs_gt_do", 1),
        _row("do_gt_obs", "do_gt_obs", 1),
        _row("approx_equal", "approx_equal", 1),
    ]
    rows_b = [
        _row("obs_gt_do", "obs_gt_do", 1),
        _row("do_gt_obs", "obs_gt_do", 0),
        _row("approx_equal", "obs_gt_do", 0),
    ]

    (run_a / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows_a) + "\n",
        encoding="utf-8",
    )
    (run_b / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows_b) + "\n",
        encoding="utf-8",
    )

    out_table = tmp_path / "summary.md"
    main(results_root=str(root), out_table=str(out_table), collapse_threshold=0.9)

    report = out_table.read_text(encoding="utf-8")
    assert "# Model Grid Summary" in report
    assert "| model_a |" in report
    assert "| model_b |" in report
