"""Monte Carlo simulation driver for Changchun Mahjong eggs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .analysis import count_types, hand_to_counter, max_egg_combo, type_feasible
from .eggs import EGG_TYPES
from .samples import HandSamples, ID_TO_TILE, generate_samples


@dataclass
class SimulationConfig:
    trials: int
    seed: Optional[int] = None


@dataclass
class SimulationResult:
    config: SimulationConfig
    frame: pd.DataFrame


def _ensure_samples(config: SimulationConfig, samples: HandSamples | None) -> HandSamples:
    if samples is not None:
        return samples
    return generate_samples(config.trials, config.seed)


def run_simulation(config: SimulationConfig, samples: HandSamples | None = None) -> SimulationResult:
    hand_samples = _ensure_samples(config, samples)
    trials = hand_samples.array.shape[0]
    if trials != config.trials:
        config = SimulationConfig(trials=trials, seed=config.seed)

    lookup = np.array(ID_TO_TILE)

    records: List[Dict[str, object]] = []

    for trial, sample_ids in enumerate(hand_samples.array, start=1):
        hand_tiles = lookup[sample_ids]
        counter = hand_to_counter(hand_tiles)

        pure_flags: Dict[str, bool] = {}
        with_lai_flags: Dict[str, bool] = {}

        for egg_type in EGG_TYPES:
            pure_flags[egg_type.key] = type_feasible(counter, egg_type, allow_laizi=False)[0]
            with_lai_flags[egg_type.key] = type_feasible(counter, egg_type, allow_laizi=True)[0]

        num_with_lai, combo_with_lai = max_egg_combo(counter, allow_laizi=True)
        num_pure_combo, combo_pure = max_egg_combo(counter, allow_laizi=False)

        record: Dict[str, object] = {
            "trial": trial,
            "num_types_any": num_with_lai,
            "num_types_pure": num_pure_combo,
            "any_egg": num_with_lai > 0,
            "any_pure": num_pure_combo > 0,
            "combo_types_any": ",".join(combo_with_lai),
            "combo_types_pure": ",".join(combo_pure),
        }

        for egg_type in EGG_TYPES:
            key = egg_type.key
            record[f"{key}_pure"] = pure_flags[key]
            record[f"{key}_with_lai"] = with_lai_flags[key]
            record[f"{key}_lai_only"] = with_lai_flags[key] and not pure_flags[key]

        records.append(record)

    frame = pd.DataFrame.from_records(records)
    return SimulationResult(config=config, frame=frame)
