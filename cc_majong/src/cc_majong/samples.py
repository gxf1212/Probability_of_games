"""Hand sample generation and storage utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

LAIZI_TILE = "1s"
MANZU = tuple(f"{n}m" for n in range(1, 10))
PINZU = tuple(f"{n}p" for n in range(1, 10))
SOUZU = tuple(f"{n}s" for n in range(1, 10))
WINDS = ("E", "S", "W", "N")
DRAGONS = ("Z", "F", "B")

ALL_TILE_TYPES = MANZU + PINZU + SOUZU + WINDS + DRAGONS
TILE_TO_ID: Dict[str, int] = {tile: idx for idx, tile in enumerate(ALL_TILE_TYPES)}
ID_TO_TILE = np.array(ALL_TILE_TYPES)


def _build_standard_deck() -> np.ndarray:
    deck: List[str] = []
    for tile in ALL_TILE_TYPES:
        deck.extend([tile] * 4)
    return np.array(deck)


STANDARD_DECK = _build_standard_deck()


DATA_COLUMNS = [f"tile_{i:02d}" for i in range(1, 15)]


@dataclass
class HandSamples:
    array: np.ndarray  # shape (n, 14), dtype=np.uint8

    @property
    def trials(self) -> int:
        return self.array.shape[0]


def generate_samples(trials: int, seed: int | None = None) -> HandSamples:
    deck = np.array(STANDARD_DECK)
    rng = np.random.default_rng(seed)
    samples = np.empty((trials, 14), dtype=np.uint8)
    for idx in range(trials):
        hand = rng.choice(deck, size=14, replace=False)
        samples[idx] = [TILE_TO_ID[tile] for tile in hand]
    return HandSamples(array=samples)


def save_samples(samples: HandSamples, path: Path) -> None:
    df = pd.DataFrame(samples.array, columns=DATA_COLUMNS)
    df.insert(0, "trial", np.arange(1, samples.trials + 1, dtype=np.int32))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_samples(path: Path) -> HandSamples:
    df = pd.read_parquet(path)
    data = df[DATA_COLUMNS].to_numpy(dtype=np.uint8, copy=True)
    return HandSamples(array=data)


def samples_to_tiles(samples: HandSamples) -> np.ndarray:
    return ID_TO_TILE[samples.array]
