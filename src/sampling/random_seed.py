from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


_CURRENT_SAMPLING_SEED = None


@dataclass
class SamplingSeedRecord:
    seed: int
    label: str
    timestamp_utc: str


def set_sampling_seed(seed, *, label="sampling", metadata_file=None, print_report=True):
    """Seed Python random and NumPy for reproducible sampling."""
    global _CURRENT_SAMPLING_SEED
    if seed is None:
        return None
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    _CURRENT_SAMPLING_SEED = seed
    record = SamplingSeedRecord(
        seed=seed,
        label=str(label),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    if metadata_file:
        with open(metadata_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.__dict__, sort_keys=True) + "\n")
    if print_report:
        print(f"Sampling random seed [{record.label}]: {record.seed}", flush=True)
    return record


def current_sampling_seed():
    return _CURRENT_SAMPLING_SEED
