from __future__ import annotations

from dataclasses import dataclass
import json
import os
import socket
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SYSTEM_INSTRUCTION = (
    'Return ONLY valid JSON with exactly one key: "label". '
    'Allowed values: "obs_gt_do", "do_gt_obs", "approx_equal".'
)


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response missing non-empty 'choices' list")

    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("OpenRouter response has invalid 'choices[0]' type")

    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter response missing 'message' object")

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
        joined = "".join(chunks).strip()
        if joined:
            return joined

    raise ValueError("OpenRouter response missing textual message content")


@dataclass
class OpenRouterRunner:
    model_name: str
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_s: float = 120.0
    max_new_tokens: int = 64
    temperature: float = 0.0
    site_url: str = ""
    app_name: str = "causalbench-llm"
    max_retries: int = 2
    retry_backoff_s: float = 1.5

    def __post_init__(self) -> None:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key in env var {self.api_key_env}. "
                "Set it before running with --backend openrouter."
            )
        self._api_key = api_key
        self._chat_url = self.base_url.rstrip("/") + "/chat/completions"

    def _request_payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }

    def generate(self, prompt: str) -> str:
        body = json.dumps(self._request_payload(prompt)).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = Request(self._chat_url, data=body, headers=headers, method="POST")
                with urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8")
                payload = json.loads(raw)
                return _extract_message_content(payload)
            except HTTPError as err:
                response_body = err.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"OpenRouter HTTP {err.code} for model '{self.model_name}': {response_body}"
                ) from err
            except (URLError, TimeoutError, socket.timeout, json.JSONDecodeError, ValueError) as err:
                last_error = err
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_s * (attempt + 1))

        raise RuntimeError(
            f"OpenRouter request failed for model '{self.model_name}' after "
            f"{self.max_retries + 1} attempts: {last_error}"
        )
