from __future__ import annotations

import pytest
import typer

from causalbench.eval import run_eval


class _DummyHF:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyOpenRouter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


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
