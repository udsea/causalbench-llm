from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class Instance:
    instance_id: str
    task: str
    scm_kind: str
    prompt: str
    gold: Dict[str, Any]

    