from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

app = typer.Typer()

LABELS: tuple[str, ...] = ("obs_gt_do", "do_gt_obs", "approx_equal")


def _acc(stats: dict[str, int]) -> float:
    if stats["n"] == 0:
        return 0.0
    return stats["correct"] / stats["n"]


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


def _gap_bucket(gap: float, eq_margin: float) -> str:
    if gap < eq_margin:
        return "borderline (gap < tol)"
    if gap < 0.08:
        return "medium (tol <= gap < 0.08)"
    return "easy (gap >= 0.08)"


@app.command()
def main(
    results_jsonl: str,
    out_table: str = "results_table.md",
    gap_tol: float = 0.02,
):
    path = Path(results_jsonl)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    if not rows:
        raise typer.Exit(code=1)

    total = len(rows)
    parse_rate = sum(1 for r in rows if bool(r.get("parse_ok", False))) / total
    accuracy = sum(int(r["score"]) for r in rows) / total

    by_label: dict[str, dict[str, int]] = {lbl: {"n": 0, "correct": 0} for lbl in LABELS}
    by_gap_bucket: dict[str, dict[str, int]] = {
        "borderline (gap < tol)": {"n": 0, "correct": 0},
        "medium (tol <= gap < 0.08)": {"n": 0, "correct": 0},
        "easy (gap >= 0.08)": {"n": 0, "correct": 0},
    }
    by_scm_kind: dict[str, dict[str, int]] = {}
    by_pred: dict[str, int] = {lbl: 0 for lbl in LABELS}

    confusion: dict[str, dict[str, int]] = {
        gold: {pred: 0 for pred in LABELS}
        for gold in LABELS
    }
    invalid_pred_count = 0
    invalid_by_gold: dict[str, int] = {lbl: 0 for lbl in LABELS}

    for r in rows:
        gold = r["gold"]
        score = int(r["score"])
        gold_label = str(gold["label"])
        if gold_label not in LABELS:
            continue

        by_label[gold_label]["n"] += 1
        by_label[gold_label]["correct"] += score

        scm_kind = str(r.get("scm_kind", "unknown"))
        by_scm_kind.setdefault(scm_kind, {"n": 0, "correct": 0})
        by_scm_kind[scm_kind]["n"] += 1
        by_scm_kind[scm_kind]["correct"] += score

        if "gap" in gold:
            gap = float(gold["gap"])
        else:
            gap = abs(float(gold["obs_prob"]) - float(gold["do_prob"]))
        eq_margin = float(gold.get("eq_margin", gold.get("tol", gap_tol)))
        bucket = _gap_bucket(gap, eq_margin)
        by_gap_bucket[bucket]["n"] += 1
        by_gap_bucket[bucket]["correct"] += score

        pred_label = _extract_pred_label(r)
        if pred_label is None:
            invalid_pred_count += 1
            invalid_by_gold[gold_label] += 1
        else:
            by_pred[pred_label] += 1
            confusion[gold_label][pred_label] += 1

    macro_accuracy = sum(_acc(by_label[lbl]) for lbl in LABELS) / len(LABELS)

    per_label_f1: dict[str, float] = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in LABELS if g != label)
        fn = sum(confusion[label][p] for p in LABELS if p != label) + invalid_by_gold[label]
        per_label_f1[label] = _f1(tp=tp, fp=fp, fn=fn)
    macro_f1 = sum(per_label_f1.values()) / len(LABELS)

    out: list[str] = []
    out.append("# Results\n")
    out.append(f"- Total instances: {total}\n")
    out.append(f"- Parse rate: {parse_rate:.3f}\n")
    out.append(f"- Accuracy: {accuracy:.3f}\n")
    out.append(f"- Macro accuracy: {macro_accuracy:.3f}\n")
    out.append(f"- Macro F1: {macro_f1:.3f}\n")
    out.append(f"- Invalid/unknown predictions: {invalid_pred_count}\n\n")

    out.append("## Predicted label distribution\n\n")
    out.append("| predicted label | n | share |\n|---|---:|---:|\n")
    for lbl in LABELS:
        n_pred = by_pred[lbl]
        out.append(f"| {lbl} | {n_pred} | {n_pred/total:.3f} |\n")
    out.append("\n")

    out.append("## Breakdown by gold label\n\n")
    out.append("| label | n | acc | f1 |\n|---|---:|---:|---:|\n")
    for lbl in LABELS:
        stats = by_label[lbl]
        out.append(f"| {lbl} | {stats['n']} | {_acc(stats):.3f} | {per_label_f1[lbl]:.3f} |\n")

    out.append("\n## Breakdown by SCM motif\n\n")
    out.append("| scm_kind | n | acc |\n|---|---:|---:|\n")
    for kind in sorted(by_scm_kind.keys()):
        stats = by_scm_kind[kind]
        out.append(f"| {kind} | {stats['n']} | {_acc(stats):.3f} |\n")

    out.append("\n## Breakdown by gap difficulty\n\n")
    out.append("| difficulty bucket | n | acc |\n|---|---:|---:|\n")
    for bucket in ("borderline (gap < tol)", "medium (tol <= gap < 0.08)", "easy (gap >= 0.08)"):
        stats = by_gap_bucket[bucket]
        out.append(f"| {bucket} | {stats['n']} | {_acc(stats):.3f} |\n")

    out.append("\n## Confusion matrix (gold x pred)\n\n")
    out.append("| gold \\ pred | obs_gt_do | do_gt_obs | approx_equal |\n")
    out.append("|---|---:|---:|---:|\n")
    for gold in LABELS:
        out.append(
            f"| {gold} | {confusion[gold]['obs_gt_do']} | {confusion[gold]['do_gt_obs']} | {confusion[gold]['approx_equal']} |\n"
        )
    out.append(
        "\nInvalid/unknown predictions are excluded from the 3x3 matrix and counted above.\n"
    )

    Path(out_table).write_text("".join(out), encoding="utf-8")
    typer.echo(f"Wrote {out_table}")


if __name__ == "__main__":
    app()
