"""A W&B run's summary, with a fallback for keys the W&B server dropped or hasn't served yet.

The W&B server has twice lost keys that train.py wrote to a run's summary just before wandb.finish()
(final_checkpoint, hf_url, training_time_s), though the wandb-summary.json file the client uploaded had them; and a
summary update made through the API wasn't visible to another process 2 s later (docs/phase2-obstacles.md §16-17).
launch.py and play.py read those keys through here. Needs only the wandb client, no Isaac Lab.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def saved_summary(run) -> dict:
    """The run's uploaded wandb-summary.json, or {} if it can't be read. Never raises."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run.file("wandb-summary.json").download(root=tmp, replace=True)
            return json.loads((Path(tmp) / "wandb-summary.json").read_text())
    except Exception:  # noqa: BLE001
        return {}


def summary_value(run, key: str):
    """run.summary[key], else the value in the run's wandb-summary.json file (None if neither has it)."""
    value = run.summary.get(key)
    return value if value is not None else saved_summary(run).get(key)
