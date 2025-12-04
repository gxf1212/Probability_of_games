"""Estimate next-draw completion probabilities for egg patterns."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

from .analysis import type_feasible
from .eggs import EGG_TYPES
from .samples import ALL_TILE_TYPES, HandSamples, ID_TO_TILE, LAIZI_TILE

REMAINING_TILES_AFTER_DEAL = 83  # 136 - (13*4 + 1)


@dataclass
class NextDrawStats:
    per_egg: Dict[str, Dict[str, float]]
    union_prob: float

    def to_dict(self) -> Dict[str, object]:
        return {"per_egg": self.per_egg, "union_prob": self.union_prob}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "NextDrawStats":
        return cls(per_egg=data["per_egg"], union_prob=data["union_prob"])


@lru_cache(maxsize=None)
def _candidate_tiles(egg_key: str) -> Tuple[str, ...]:
    if egg_key == "gang_dan":
        return tuple(ALL_TILE_TYPES)
    tiles: Set[str] = set()
    for egg in EGG_TYPES:
        if egg.key != egg_key:
            continue
        for pattern in egg.patterns:
            tiles.update(tile for tile, _ in pattern.tiles)
            if pattern.allows_laizi:
                tiles.add(LAIZI_TILE)
        break
    return tuple(sorted(tiles))


def _winning_tiles(counter: Counter, egg_key: str) -> Tuple[Set[str], Set[str]]:
    pure_tiles: Set[str] = set()
    laizi_tiles: Set[str] = set()
    egg = next(e for e in EGG_TYPES if e.key == egg_key)
    for tile in _candidate_tiles(egg_key):
        if counter[tile] >= 4:
            continue
        counter[tile] += 1
        pure = type_feasible(counter, egg, allow_laizi=False)[0]
        with_lai = type_feasible(counter, egg, allow_laizi=True)[0]
        counter[tile] -= 1
        if pure:
            pure_tiles.add(tile)
        elif with_lai:
            laizi_tiles.add(tile)
    return pure_tiles, laizi_tiles


def compute_next_draw_stats(samples: HandSamples) -> NextDrawStats:
    per_egg_counts: Dict[str, Dict[str, float]] = {
        egg.key: {"pure": 0.0, "lai": 0.0} for egg in EGG_TYPES
    }
    union_count = 0.0
    hands = samples.array[:, :13]
    for hand in hands:
        tiles = ID_TO_TILE[hand]
        counter = Counter(tiles)
        available = {tile: max(0, 4 - counter[tile]) for tile in ALL_TILE_TYPES}
        union_tiles: Set[str] = set()
        for egg in EGG_TYPES:
            pure_tiles, laizi_tiles = _winning_tiles(counter, egg.key)
            per_egg_counts[egg.key]["pure"] += sum(available[t] for t in pure_tiles)
            per_egg_counts[egg.key]["lai"] += sum(available[t] for t in laizi_tiles)
            union_tiles.update(pure_tiles)
            union_tiles.update(laizi_tiles)
        union_count += sum(available[t] for t in union_tiles)
    denom = len(hands) * REMAINING_TILES_AFTER_DEAL
    per_egg_probs = {
        key: {
            "pure": counts["pure"] / denom,
            "lai": counts["lai"] / denom,
            "total": (counts["pure"] + counts["lai"]) / denom,
        }
        for key, counts in per_egg_counts.items()
    }
    union_prob = union_count / denom
    return NextDrawStats(per_egg=per_egg_probs, union_prob=union_prob)


def save_next_draw_stats(stats: NextDrawStats, path: Path) -> None:
    path.write_text(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_next_draw_stats(path: Path) -> NextDrawStats:
    return NextDrawStats.from_dict(json.loads(path.read_text(encoding="utf-8")))
