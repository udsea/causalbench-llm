from __future__ import annotations
import json
from pathlib import Path
import typer
from tqdm import tqdm

from causalbench.models.hf_runner import HFRunner
from causalbench.tasks.build_instances import build_intervention_compare_instances, parse_scm_kinds
from causalbench.tasks.scoring import extract_first_json_obj, score_label_strict

app = typer.Typer()


@app.command()
def main(
    out_dir: str = "results/runs/dev",
    model_name: str = "distilgpt2",
    device: str = "cpu",
    n_instances: int = 30,
    seed: int = 0,
    scm_kinds: str = "all",
    balance_labels: bool = True,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    instances = build_intervention_compare_instances(
        n=n_instances,
        seed=seed,
        scm_kinds=parse_scm_kinds(scm_kinds),
        balance_labels=balance_labels,
    )
    runner = HFRunner(model_name=model_name, device_preference=device)

    results_file = out_path / "results.jsonl"
    with results_file.open("w", encoding="utf-8") as f:
        for inst in tqdm(instances):
            raw = runner.generate(inst.prompt)
            ok, pred = extract_first_json_obj(raw)
            score = score_label_strict(pred, inst.gold) if ok and pred else 0

            record = {
                "instance_id": inst.instance_id,
                "task": inst.task,
                "scm_kind": inst.scm_kind,
                "gold": inst.gold,
                "raw_output": raw,
                "parse_ok": ok,
                "pred": pred,
                "score": score,
            }
            f.write(json.dumps(record) + "\n")

    typer.echo(f"Wrote {results_file}")


if __name__ == "__main__":
    app()
