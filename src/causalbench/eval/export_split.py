from __future__ import annotations

import json
from pathlib import Path

import typer

from causalbench.tasks.build_instances import build_intervention_compare_instances, parse_scm_kinds

app = typer.Typer()


@app.command()
def main(
    out_jsonl: str = "experiments/fixed_split_v1.jsonl",
    n_instances: int = 120,
    seed: int = 0,
    scm_kinds: str = "all",
    balance_labels: bool = True,
    stratify_motif_label: bool = True,
    n_prompt_obs_samples: int = 2000,
    n_obs_samples: int = 8000,
    n_mc_samples: int = 8000,
    x_reference_value: float = 0.0,
    x_band: float = 0.25,
    eq_margin: float = 0.06,
    dir_margin: float = 0.06,
    discard_ambiguous: bool = True,
    out_meta_json: str = "",
):
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    instances = build_intervention_compare_instances(
        n=n_instances,
        seed=seed,
        scm_kinds=parse_scm_kinds(scm_kinds),
        balance_labels=balance_labels,
        stratify_motif_label=stratify_motif_label,
        n_prompt_obs_samples=n_prompt_obs_samples,
        n_obs_samples=n_obs_samples,
        n_mc_samples=n_mc_samples,
        x_reference_value=x_reference_value,
        x_band=x_band,
        eq_margin=eq_margin,
        dir_margin=dir_margin,
        discard_ambiguous=discard_ambiguous,
    )

    by_label: dict[str, int] = {}
    by_motif: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as f:
        for inst in instances:
            by_label[inst.gold["label"]] = by_label.get(inst.gold["label"], 0) + 1
            by_motif[inst.scm_kind] = by_motif.get(inst.scm_kind, 0) + 1
            row = {
                "instance_id": inst.instance_id,
                "task": inst.task,
                "scm_kind": inst.scm_kind,
                "prompt": inst.prompt,
                "gold": inst.gold,
            }
            f.write(json.dumps(row) + "\n")

    if out_meta_json:
        meta_path = Path(out_meta_json)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "n_instances": len(instances),
            "seed": seed,
            "scm_kinds": parse_scm_kinds(scm_kinds),
            "balance_labels": balance_labels,
            "stratify_motif_label": stratify_motif_label,
            "n_prompt_obs_samples": n_prompt_obs_samples,
            "n_obs_samples": n_obs_samples,
            "n_mc_samples": n_mc_samples,
            "x_reference_value": x_reference_value,
            "x_band": x_band,
            "eq_margin": eq_margin,
            "dir_margin": dir_margin,
            "discard_ambiguous": discard_ambiguous,
            "label_counts": by_label,
            "motif_counts": by_motif,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        typer.echo(f"Wrote {meta_path}")

    typer.echo(f"Wrote {out_path}")
    typer.echo(f"Instances: {len(instances)}")
    typer.echo(f"Label counts: {by_label}")
    typer.echo(f"Motif counts: {dict(sorted(by_motif.items()))}")


if __name__ == "__main__":
    app()
