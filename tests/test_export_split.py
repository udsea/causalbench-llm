from __future__ import annotations

import json

from causalbench.eval.export_split import main


def test_export_split_writes_instances_and_meta(tmp_path):
    out_jsonl = tmp_path / "fixed_split.jsonl"
    out_meta = tmp_path / "fixed_split_meta.json"

    main(
        out_jsonl=str(out_jsonl),
        out_meta_json=str(out_meta),
        n_instances=6,
        seed=0,
        scm_kinds="confounding,confounding_only",
        balance_labels=True,
        stratify_motif_label=True,
        n_prompt_obs_samples=200,
        n_obs_samples=400,
        n_mc_samples=400,
    )

    assert out_jsonl.exists()
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6

    first = json.loads(lines[0])
    assert "instance_id" in first
    assert "prompt" in first
    assert "gold" in first
    assert "scm_kind" in first

    meta = json.loads(out_meta.read_text(encoding="utf-8"))
    assert meta["n_instances"] == 6
    assert meta["balance_labels"] is True
    assert meta["stratify_motif_label"] is True
