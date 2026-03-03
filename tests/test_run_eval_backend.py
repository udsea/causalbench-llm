from __future__ import annotations

import json

import pytest
import typer

from causalbench.eval import run_eval


class _DummyHF:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyOpenRouter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyGenerator:
    def generate(self, prompt: str) -> str:
        _ = prompt
        return '{"label":"obs_gt_do"}'


def test_resolve_runner_auto_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval, "HFRunner", _DummyHF)
    monkeypatch.setattr(run_eval, "OpenRouterRunner", _DummyOpenRouter)

    runner = run_eval._resolve_runner(
        backend="auto",
        model_name="distilgpt2",
        device="cpu",
        temperature=0.0,
        max_new_tokens=64,
        openrouter_api_base="https://openrouter.ai/api/v1",
        openrouter_api_key_env="OPENROUTER_API_KEY",
        openrouter_site_url="",
        openrouter_app_name="causalbench-llm",
        request_timeout_s=120.0,
    )

    assert isinstance(runner, _DummyHF)
    assert runner.kwargs["model_name"] == "distilgpt2"
    assert runner.kwargs["torch_dtype"] == "auto"
    assert runner.kwargs["quantization"] == "none"


def test_resolve_runner_hf_forwards_quantization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval, "HFRunner", _DummyHF)
    monkeypatch.setattr(run_eval, "OpenRouterRunner", _DummyOpenRouter)

    runner = run_eval._resolve_runner(
        backend="hf",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device="cuda",
        torch_dtype="float16",
        quantization="4bit",
    )

    assert isinstance(runner, _DummyHF)
    assert runner.kwargs["torch_dtype"] == "float16"
    assert runner.kwargs["quantization"] == "4bit"


def test_resolve_runner_auto_openrouter_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval, "HFRunner", _DummyHF)
    monkeypatch.setattr(run_eval, "OpenRouterRunner", _DummyOpenRouter)

    runner = run_eval._resolve_runner(
        backend="auto",
        model_name="openrouter/openai/gpt-4o-mini",
        device="cpu",
        temperature=0.0,
        max_new_tokens=64,
        openrouter_api_base="https://openrouter.ai/api/v1",
        openrouter_api_key_env="OPENROUTER_API_KEY",
        openrouter_site_url="",
        openrouter_app_name="causalbench-llm",
        request_timeout_s=120.0,
    )

    assert isinstance(runner, _DummyOpenRouter)
    assert runner.kwargs["model_name"] == "openai/gpt-4o-mini"


def test_resolve_runner_invalid_backend_raises() -> None:
    with pytest.raises(typer.BadParameter):
        run_eval._resolve_runner(
            backend="bad",
            model_name="distilgpt2",
            device="cpu",
            temperature=0.0,
            max_new_tokens=64,
            openrouter_api_base="https://openrouter.ai/api/v1",
            openrouter_api_key_env="OPENROUTER_API_KEY",
            openrouter_site_url="",
            openrouter_app_name="causalbench-llm",
            request_timeout_s=120.0,
        )


def test_main_reads_instances_jsonl_and_writes_metadata(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "instance_id": "id-1",
            "task": "intervention_compare_confounding",
            "scm_kind": "confounding",
            "prompt": "prompt-1",
            "gold": {"label": "obs_gt_do"},
        },
        {
            "instance_id": "id-2",
            "task": "intervention_compare_confounding",
            "scm_kind": "confounding",
            "prompt": "prompt-2",
            "gold": {"label": "do_gt_obs"},
        },
    ]
    split = tmp_path / "split.jsonl"
    split.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    monkeypatch.setattr(run_eval, "_resolve_runner", lambda **kwargs: _DummyGenerator())

    out_dir = tmp_path / "out"
    run_eval.main(
        out_dir=str(out_dir),
        model_name="openrouter/openai/gpt-4o-mini",
        backend="auto",
        instances_jsonl=str(split),
    )

    result_file = out_dir / "results.jsonl"
    assert result_file.exists()
    out_rows = [json.loads(line) for line in result_file.read_text(encoding="utf-8").splitlines()]
    assert len(out_rows) == 2
    assert out_rows[0]["backend"] == "openrouter"
    assert out_rows[0]["model_name"] == "openai/gpt-4o-mini"
