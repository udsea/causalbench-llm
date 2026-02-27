from __future__ import annotations
import json
import os
from pathlib import Path
import typer
from tqdm import tqdm

from causalbench.models.hf_runner import HFRunner
from causalbench.models.openrouter_runner import OpenRouterRunner
from causalbench.tasks.build_instances import build_intervention_compare_instances, parse_scm_kinds
from causalbench.tasks.scoring import extract_first_json_obj, score_label_strict

app = typer.Typer()


def _resolve_runner(
    *,
    backend: str,
    model_name: str,
    device: str,
    temperature: float,
    max_new_tokens: int,
    openrouter_api_base: str,
    openrouter_api_key_env: str,
    openrouter_site_url: str,
    openrouter_app_name: str,
    request_timeout_s: float,
):
    selected = backend.lower().strip()
    resolved_model_name = model_name

    if selected not in {"hf", "openrouter", "auto"}:
        raise typer.BadParameter("backend must be one of: hf, openrouter, auto")

    if selected == "auto":
        prefix = "openrouter/"
        if model_name.startswith(prefix):
            selected = "openrouter"
            resolved_model_name = model_name[len(prefix) :]
        else:
            selected = "hf"

    if selected == "hf":
        return HFRunner(
            model_name=resolved_model_name,
            device_preference=device,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    site_url = openrouter_site_url or os.getenv("OPENROUTER_SITE_URL", "")
    app_name = openrouter_app_name or os.getenv("OPENROUTER_APP_NAME", "causalbench-llm")
    return OpenRouterRunner(
        model_name=resolved_model_name,
        api_key_env=openrouter_api_key_env,
        base_url=openrouter_api_base,
        timeout_s=request_timeout_s,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        site_url=site_url,
        app_name=app_name,
    )


@app.command()
def main(
    out_dir: str = "results/runs/dev",
    model_name: str = "distilgpt2",
    device: str = "cpu",
    n_instances: int = 30,
    seed: int = 0,
    scm_kinds: str = "all",
    balance_labels: bool = True,
    n_prompt_obs_samples: int = 2000,
    x_reference_value: float = 0.0,
    x_band: float = 0.25,
    eq_margin: float = 0.06,
    dir_margin: float = 0.06,
    discard_ambiguous: bool = True,
    stratify_motif_label: bool = False,
    backend: str = "hf",
    temperature: float = 0.0,
    max_new_tokens: int = 64,
    openrouter_api_base: str = "https://openrouter.ai/api/v1",
    openrouter_api_key_env: str = "OPENROUTER_API_KEY",
    openrouter_site_url: str = "",
    openrouter_app_name: str = "causalbench-llm",
    request_timeout_s: float = 120.0,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    instances = build_intervention_compare_instances(
        n=n_instances,
        seed=seed,
        scm_kinds=parse_scm_kinds(scm_kinds),
        balance_labels=balance_labels,
        n_prompt_obs_samples=n_prompt_obs_samples,
        x_reference_value=x_reference_value,
        x_band=x_band,
        eq_margin=eq_margin,
        dir_margin=dir_margin,
        discard_ambiguous=discard_ambiguous,
        stratify_motif_label=stratify_motif_label,
    )
    try:
        runner = _resolve_runner(
            backend=backend,
            model_name=model_name,
            device=device,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            openrouter_api_base=openrouter_api_base,
            openrouter_api_key_env=openrouter_api_key_env,
            openrouter_site_url=openrouter_site_url,
            openrouter_app_name=openrouter_app_name,
            request_timeout_s=request_timeout_s,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

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
                "prompt": inst.prompt,
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
