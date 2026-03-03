from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

LABELS: tuple[str, ...] = ("obs_gt_do", "do_gt_obs", "approx_equal")
app = typer.Typer()


@dataclass
class RunSummary:
    run_name: str
    total: int
    parse_rate: float
    accuracy: float
    macro_f1: float
    recall_obs: float
    recall_do: float
    recall_eq: float
    pred_obs_share: float
    pred_do_share: float
    pred_eq_share: float
    collapse: str


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _extract_pred_label(row: dict[str, Any]) -> str | None:
    pred = row.get("pred")
    if isinstance(pred, dict):
        label = pred.get("label")
        if isinstance(label, str) and label in LABELS:
            return label
    return None


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def _collapse_from_shares(
    pred_obs_share: float,
    pred_do_share: float,
    pred_eq_share: float,
    threshold: float,
) -> str:
    if pred_obs_share >= threshold:
        return "obs_gt_do"
    if pred_do_share >= threshold:
        return "do_gt_obs"
    if pred_eq_share >= threshold:
        return "approx_equal"
    return "none"


def _summarize_results(run_name: str, rows: list[dict[str, Any]], collapse_threshold: float) -> RunSummary:
    total = len(rows)
    parse_rate = _safe_div(sum(1 for r in rows if bool(r.get("parse_ok", False))), total)
    accuracy = _safe_div(sum(int(r.get("score", 0)) for r in rows), total)

    gold_counts = {label: 0 for label in LABELS}
    confusion: dict[str, dict[str, int]] = {
        gold: {pred: 0 for pred in LABELS}
        for gold in LABELS
    }
    invalid_by_gold = {label: 0 for label in LABELS}
    pred_counts = {label: 0 for label in LABELS}

    for row in rows:
        gold = row.get("gold")
        if not isinstance(gold, dict):
            continue
        gold_label = gold.get("label")
        if not isinstance(gold_label, str) or gold_label not in LABELS:
            continue

        gold_counts[gold_label] += 1
        pred_label = _extract_pred_label(row)
        if pred_label is None:
            invalid_by_gold[gold_label] += 1
            continue
        pred_counts[pred_label] += 1
        confusion[gold_label][pred_label] += 1

    f1s: dict[str, float] = {}
    recalls: dict[str, float] = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in LABELS if gold != label)
        fn = sum(confusion[label][pred] for pred in LABELS if pred != label) + invalid_by_gold[label]
        f1s[label] = _f1(tp=tp, fp=fp, fn=fn)
        recalls[label] = _safe_div(tp, gold_counts[label])

    pred_obs_share = _safe_div(pred_counts["obs_gt_do"], total)
    pred_do_share = _safe_div(pred_counts["do_gt_obs"], total)
    pred_eq_share = _safe_div(pred_counts["approx_equal"], total)

    return RunSummary(
        run_name=run_name,
        total=total,
        parse_rate=parse_rate,
        accuracy=accuracy,
        macro_f1=(sum(f1s.values()) / len(LABELS)),
        recall_obs=recalls["obs_gt_do"],
        recall_do=recalls["do_gt_obs"],
        recall_eq=recalls["approx_equal"],
        pred_obs_share=pred_obs_share,
        pred_do_share=pred_do_share,
        pred_eq_share=pred_eq_share,
        collapse=_collapse_from_shares(
            pred_obs_share=pred_obs_share,
            pred_do_share=pred_do_share,
            pred_eq_share=pred_eq_share,
            threshold=collapse_threshold,
        ),
    )


@app.command()
def main(
    results_root: str = "results/runs/model_grid_v1",
    out_table: str = "results/summary_model_grid.md",
    collapse_threshold: float = 0.9,
):
    root = Path(results_root)
    result_files = sorted(root.glob("*/results.jsonl"))
    if not result_files:
        raise typer.BadParameter(f"No results found under {root}/*/results.jsonl")

    summaries: list[RunSummary] = []
    for path in result_files:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        summaries.append(_summarize_results(run_name=path.parent.name, rows=rows, collapse_threshold=collapse_threshold))

    summaries.sort(key=lambda s: (s.macro_f1, s.accuracy), reverse=True)

    out_lines: list[str] = []
    out_lines.append("# Model Grid Summary\n\n")
    out_lines.append(f"- Runs found: {len(summaries)}\n")
    out_lines.append(f"- Collapse threshold: {collapse_threshold:.2f}\n\n")

    out_lines.append(
        "| run | total | parse_rate | acc | macro_f1 | recall_obs | recall_do | recall_eq | pred_obs | pred_do | pred_eq | collapse |\n"
    )
    out_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for s in summaries:
        out_lines.append(
            f"| {s.run_name} | {s.total} | {s.parse_rate:.3f} | {s.accuracy:.3f} | {s.macro_f1:.3f} | "
            f"{s.recall_obs:.3f} | {s.recall_do:.3f} | {s.recall_eq:.3f} | "
            f"{s.pred_obs_share:.3f} | {s.pred_do_share:.3f} | {s.pred_eq_share:.3f} | {s.collapse} |\n"
        )

    out_path = Path(out_table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(out_lines), encoding="utf-8")
    typer.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    app()
