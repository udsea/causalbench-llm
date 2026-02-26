from __future__ import annotations
from typing import Any, Dict, Tuple
import json


ALLOWED_LABELS = {"obs_gt_do", "do_gt_obs", "approx_equal"}


def extract_first_json_obj(text: str) -> Tuple[bool, Dict[str, Any] | None]:
    """
    Try to extract and parse the first JSON object from text.
    Returns (ok, obj).
    """
    if not text:
        return False, None

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return False, None

    blob = text[start : end + 1]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return False, None

    if not isinstance(obj, dict):
        return False, None
    return True, obj


def score_label_strict(pred: Dict[str, Any], gold: Dict[str, Any]) -> int:
    """
    Strict scoring:
      - pred must be exactly {"label": <allowed>}
      - gold must contain "label"
      - correct => 1 else 0
    """
    if set(pred.keys()) != {"label"}:
        return 0
    label = pred.get("label")
    if label not in ALLOWED_LABELS:
        return 0
    return 1 if label == gold.get("label") else 0