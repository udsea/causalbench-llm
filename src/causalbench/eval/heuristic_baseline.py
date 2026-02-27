from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import typer

from causalbench.tasks.scoring import score_label_strict

app = typer.Typer()

_A_HAT_RE = re.compile(r"(?:Estimated A = .*?:|A1_hat = P\(Y > 0 \| X ~ [^)]+\):)\s*([0-9.]+)")
_BASELINE_RE = re.compile(r"Baseline P\(Y > 0\):\s*([0-9.]+)")
_DELTA_RE = re.compile(r"delta\((?:A_hat|A1_hat) - baseline\):\s*([-0-9.]+)")
_N_IN_BAND_RE = re.compile(r"(?:Count in band:\s*([0-9]+)|A1 95% CI: .*count=([0-9]+))")
_CI_RE = re.compile(r"(?:Approx 95% CI for A_hat:|A1 95% CI:)\s*\[([0-9.]+),\s*([0-9.]+)\]")


def _extract_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _features_from_row(row: dict[str, Any]) -> tuple[float | None, float | None, int | None, float | None]:
    gold = row.get("gold", {})
    if not isinstance(gold, dict):
        gold = {}

    delta = _extract_float(gold.get("prompt_delta_a_baseline"))
    ci_width = _extract_float(gold.get("prompt_ci_width"))

    n_in_band_val = gold.get("prompt_n_in_band")
    n_in_band = int(n_in_band_val) if isinstance(n_in_band_val, (int, float)) else None

    if delta is None:
        prompt = row.get("prompt")
        if isinstance(prompt, str):
            delta_match = _DELTA_RE.search(prompt)
            if delta_match:
                delta = float(delta_match.group(1))
            a_match = _A_HAT_RE.search(prompt)
            b_match = _BASELINE_RE.search(prompt)
            if delta is None and a_match and b_match:
                delta = float(a_match.group(1)) - float(b_match.group(1))
            if n_in_band is None:
                n_match = _N_IN_BAND_RE.search(prompt)
                if n_match:
                    value = n_match.group(1) or n_match.group(2)
                    if value is not None:
                        n_in_band = int(value)
            if ci_width is None:
                ci_match = _CI_RE.search(prompt)
                if ci_match:
                    ci_low = float(ci_match.group(1))
                    ci_high = float(ci_match.group(2))
                    ci_width = ci_high - ci_low

    return delta, ci_width, n_in_band, _extract_float(gold.get("prompt_a_hat"))


def heuristic_predict_from_features(
    delta_a_baseline: float | None,
    ci_width: float | None,
    n_in_band: int | None,
    *,
    dir_threshold: float = 0.08,
    min_n_in_band: int = 50,
    max_ci_width: float = 0.20,
) -> str:
    if n_in_band is None or n_in_band < min_n_in_band:
        return "approx_equal"
    if ci_width is not None and not (ci_width != ci_width) and ci_width > max_ci_width:
        return "approx_equal"
    if delta_a_baseline is None or (delta_a_baseline != delta_a_baseline):
        return "approx_equal"
    if delta_a_baseline <= -dir_threshold:
        return "do_gt_obs"
    if delta_a_baseline >= dir_threshold:
        return "obs_gt_do"
    return "approx_equal"


@app.command()
def from_results(
    results_jsonl: str,
    out_jsonl: str = "results/runs/heuristic/results.jsonl",
    dir_threshold: float = 0.08,
    min_n_in_band: int = 50,
    max_ci_width: float = 0.20,
):
    in_path = Path(results_jsonl)
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    out_rows: list[dict[str, Any]] = []
    correct = 0
    for row in rows:
        delta, ci_width, n_in_band, _ = _features_from_row(row)
        label = heuristic_predict_from_features(
            delta_a_baseline=delta,
            ci_width=ci_width,
            n_in_band=n_in_band,
            dir_threshold=dir_threshold,
            min_n_in_band=min_n_in_band,
            max_ci_width=max_ci_width,
        )
        pred = {"label": label}

        gold = row.get("gold")
        if not isinstance(gold, dict):
            gold = {}
        score = score_label_strict(pred=pred, gold=gold)
        correct += score

        out_row = {
            "instance_id": row.get("instance_id"),
            "task": row.get("task"),
            "scm_kind": row.get("scm_kind"),
            "prompt": row.get("prompt"),
            "gold": gold,
            "raw_output": json.dumps(pred),
            "parse_ok": True,
            "pred": pred,
            "score": score,
            "model_name": "heuristic_delta_baseline",
            "meta": {
                "dir_threshold": dir_threshold,
                "min_n_in_band": min_n_in_band,
                "max_ci_width": max_ci_width,
                "delta_a_baseline": delta,
                "ci_width": ci_width,
                "n_in_band": n_in_band,
            },
        }
        out_rows.append(out_row)

    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    acc = (correct / len(out_rows)) if out_rows else 0.0
    typer.echo(f"Wrote {out_path}")
    typer.echo(f"Heuristic accuracy: {acc:.3f} ({correct}/{len(out_rows)})")


if __name__ == "__main__":
    app()
