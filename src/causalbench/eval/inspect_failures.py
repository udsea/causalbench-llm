from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

app = typer.Typer()


def _pred_label(row: dict[str, Any]) -> str:
    pred = row.get("pred")
    if isinstance(pred, dict):
        label = pred.get("label")
        if isinstance(label, str):
            return label
    return "<none>"


@app.command()
def main(
    results_jsonl: str,
    gold_label: str = "do_gt_obs",
    only_incorrect: bool = True,
    limit: int = 10,
):
    path = Path(results_jsonl)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    selected: list[dict[str, Any]] = []
    for row in rows:
        gold = row.get("gold", {})
        if not isinstance(gold, dict):
            continue
        if gold.get("label") != gold_label:
            continue
        if only_incorrect and int(row.get("score", 0)) == 1:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break

    if not selected:
        typer.echo("No matching rows found.")
        raise typer.Exit(code=0)

    for i, row in enumerate(selected, start=1):
        gold = row["gold"]
        typer.echo(f"--- failure {i} ---")
        typer.echo(f"instance_id: {row.get('instance_id')}")
        typer.echo(f"motif: {row.get('scm_kind')}")
        typer.echo(f"gold_label: {gold.get('label')}")
        typer.echo(f"pred_label: {_pred_label(row)}")
        typer.echo(f"score: {row.get('score')}")
        typer.echo(
            f"gold_probs: obs_prob={gold.get('obs_prob')}, do_prob={gold.get('do_prob')}, gap={gold.get('gap')}"
        )
        typer.echo(f"raw_output: {row.get('raw_output')}")
        prompt = row.get("prompt")
        if isinstance(prompt, str):
            typer.echo("prompt:")
            typer.echo(prompt)
        else:
            typer.echo("prompt: <not present in results.jsonl>")
        typer.echo("")


if __name__ == "__main__":
    app()
