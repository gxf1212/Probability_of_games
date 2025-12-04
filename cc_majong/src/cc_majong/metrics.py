"""Aggregate statistics for Changchun Mahjong simulations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

import json
import pandas as pd

import numpy as np

from .eggs import EGG_TYPES, TYPE_LABELS

MAX_RUNNING_POINTS = 2000


@dataclass
class SummaryData:
    egg_probabilities: Dict[str, Dict[str, float]]
    any_prob: float
    pure_prob: float
    laizi_only_prob: float
    combo_distribution: Dict[int, float]
    running_trials: List[int]
    running_any: List[float]
    running_pure: List[float]
    running_by_type: Dict[str, List[float]]
    trials: int | None = None
    seed: int | None = None

    def none_prob(self) -> float:
        return 1.0 - self.any_prob

    def to_json_dict(self) -> Dict[str, object]:
        data = {
            "egg_probabilities": self.egg_probabilities,
            "any_prob": self.any_prob,
            "pure_prob": self.pure_prob,
            "laizi_only_prob": self.laizi_only_prob,
            "combo_distribution": {str(k): v for k, v in self.combo_distribution.items()},
            "running_trials": self.running_trials,
            "running_any": self.running_any,
            "running_pure": self.running_pure,
            "running_by_type": self.running_by_type,
        }
        if self.trials is not None:
            data["trials"] = self.trials
        if self.seed is not None:
            data["seed"] = self.seed
        return data

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> "SummaryData":
        combo_distribution = {int(k): float(v) for k, v in data["combo_distribution"].items()}
        return cls(
            egg_probabilities={
                key: {
                    sub_key: float(sub_value)
                    for sub_key, sub_value in sub_map.items()
                }
                for key, sub_map in data["egg_probabilities"].items()
            },
            any_prob=float(data["any_prob"]),
            pure_prob=float(data["pure_prob"]),
            laizi_only_prob=float(data["laizi_only_prob"]),
            combo_distribution=combo_distribution,
            running_trials=[int(x) for x in data["running_trials"]],
            running_any=[float(x) for x in data["running_any"]],
            running_pure=[float(x) for x in data["running_pure"]],
            running_by_type={key: [float(v) for v in values] for key, values in data["running_by_type"].items()},
            trials=int(data["trials"]) if "trials" in data else None,
            seed=int(data["seed"]) if "seed" in data else None,
        )

    def convergence_frame(self) -> pd.DataFrame:
        data = {
            "trial": self.running_trials,
            "任意蛋（含赖子）": self.running_any,
            "纯净蛋": self.running_pure,
        }
        for key, values in self.running_by_type.items():
            data[TYPE_LABELS.get(key, key)] = values
        return pd.DataFrame(data)


def _running_average(series: pd.Series) -> np.ndarray:
    counts = np.arange(1, len(series) + 1)
    return series.cumsum().to_numpy(dtype=float) / counts


def _select_indices(length: int) -> np.ndarray:
    if length <= MAX_RUNNING_POINTS:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, num=MAX_RUNNING_POINTS, dtype=int))


def compute_summary(frame: pd.DataFrame, config=None) -> SummaryData:
    egg_probabilities: Dict[str, Dict[str, float]] = {}
    for egg in EGG_TYPES:
        with_prob = frame[f"{egg.key}_with_lai"].mean()
        pure_prob = frame[f"{egg.key}_pure"].mean()
        lai_prob = frame[f"{egg.key}_lai_only"].mean()
        egg_probabilities[egg.key] = {
            "with_laizi": with_prob,
            "pure": pure_prob,
            "lai_only": lai_prob,
        }

    any_prob = frame["any_egg"].mean()
    pure_prob = frame["any_pure"].mean()
    laizi_only_prob = frame[[col for col in frame.columns if col.endswith("_lai_only")]].any(axis=1).mean()

    combo_counts = frame["num_types_any"].value_counts().sort_index()
    combo_distribution = {int(k): float(v / combo_counts.sum()) for k, v in combo_counts.items()}

    any_running = _running_average(frame["any_egg"].astype(int))
    pure_running = _running_average(frame["any_pure"].astype(int))
    indices = _select_indices(len(frame))

    running_trials = (indices + 1).tolist()
    running_any = any_running[indices].tolist()
    running_pure = pure_running[indices].tolist()

    running_by_type: Dict[str, List[float]] = {}
    for egg in EGG_TYPES:
        running = _running_average(frame[f"{egg.key}_with_lai"].astype(int))
        running_by_type[egg.key] = running[indices].tolist()

    trials = getattr(config, "trials", None)
    seed = getattr(config, "seed", None)

    return SummaryData(
        egg_probabilities=egg_probabilities,
        any_prob=float(any_prob),
        pure_prob=float(pure_prob),
        laizi_only_prob=float(laizi_only_prob),
        combo_distribution=combo_distribution,
        running_trials=running_trials,
        running_any=running_any,
        running_pure=running_pure,
        running_by_type=running_by_type,
        trials=trials,
        seed=seed,
    )


def save_summary(summary: SummaryData, path: Path) -> None:
    path.write_text(json.dumps(summary.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_summary(path: Path) -> SummaryData:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SummaryData.from_json_dict(data)
