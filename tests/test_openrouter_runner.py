from __future__ import annotations

import pytest

from causalbench.models.openrouter_runner import _extract_message_content


def test_extract_message_content_string() -> None:
    payload = {"choices": [{"message": {"content": '{"label":"obs_gt_do"}'}}]}
    assert _extract_message_content(payload) == '{"label":"obs_gt_do"}'


def test_extract_message_content_parts() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": '{"label":"'},
                        {"type": "text", "text": 'do_gt_obs"}'},
                    ]
                }
            }
        ]
    }
    assert _extract_message_content(payload) == '{"label":"do_gt_obs"}'


def test_extract_message_content_missing_choices_raises() -> None:
    with pytest.raises(ValueError, match="choices"):
        _extract_message_content({"id": "123"})
