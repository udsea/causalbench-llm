from __future__ import annotations
import json
from pathlib import Path
import typer

app = typer.Typer()


@app.command()
def main(
    results_jsonl: str,
    out_table: str = "results_table.md",
):
    path = Path(results_jsonl)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    if not rows:
        raise typer.Exit(code=1)

    total = len(rows)
    acc = sum(r["score"] for r in rows) / total

    # breakdown by label (gold)
    by_label: dict[str, dict[str, int]] = {}
    for r in rows:
        lbl = r["gold"]["label"]
        by_label.setdefault(lbl, {"n": 0, "correct": 0})
        by_label[lbl]["n"] += 1
        by_label[lbl]["correct"] += r["score"]

    out = []
    out.append("# Results\n")
    out.append(f"- Total instances: {total}\n")
    out.append(f"- Accuracy: {acc:.3f}\n\n")

    out.append("## Breakdown by gold label\n\n")
    out.append("| label | n | acc |\n|---|---:|---:|\n")
    for lbl, stats in by_label.items():
        out.append(f"| {lbl} | {stats['n']} | {stats['correct']/stats['n']:.3f} |\n")

    Path(out_table).write_text("".join(out), encoding="utf-8")
    typer.echo(f"Wrote {out_table}")


if __name__ == "__main__":
    app()
