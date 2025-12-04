"""Egg multiplicity statistics (number of eggs including duplicates)."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .eggs import EGG_TYPES
from .samples import HandSamples, ID_TO_TILE, LAIZI_TILE


@dataclass
class MultiplicityStats:
    total_distribution: Dict[int, float]
    per_type_distribution: Dict[str, Dict[int, float]]

    def to_json(self) -> Dict[str, Dict[str, float]]:
        return {
            "total_distribution": {str(k): v for k, v in self.total_distribution.items()},
            "per_type_distribution": {
                key: {str(cnt): prob for cnt, prob in dist.items()}
                for key, dist in self.per_type_distribution.items()
            },
        }

    @classmethod
    def from_json(cls, data: Dict[str, Dict[str, float]]) -> "MultiplicityStats":
        total = {int(k): float(v) for k, v in data["total_distribution"].items()}
        per_type = {
            key: {int(cnt): float(prob) for cnt, prob in dist.items()}
            for key, dist in data["per_type_distribution"].items()
        }
        return cls(total_distribution=total, per_type_distribution=per_type)


SPECIAL_GROUPS: Dict[str, Tuple[str, ...]] = {
    "xuan_feng": ("E", "S", "W", "N"),
    "xi_dan": ("Z", "F", "B"),
    "yao_dan": ("1m", "1p", "1s"),
    "jiu_dan": ("9m", "9p", "9s"),
}


def _consume_special(counter: Counter, tiles: Sequence[str], eggs: int) -> Counter | None:
    work = counter.copy()
    if eggs == 0:
        return work

    # base requirement: one of each tile (laizi can代替)
    for tile in tiles:
        if work[tile] > 0:
            work[tile] -= 1
        elif work[LAIZI_TILE] > 0:
            work[LAIZI_TILE] -= 1
        else:
            return None

    extras = eggs - 1
    if extras == 0:
        return work

    for tile in tiles:
        if extras == 0:
            break
        take = min(work[tile], extras)
        work[tile] -= take
        extras -= take

    if extras > 0:
        if work[LAIZI_TILE] >= extras:
            work[LAIZI_TILE] -= extras
            extras = 0
        else:
            return None
    return work


def _enumerate_special(counter: Counter, tiles: Sequence[str]) -> List[Tuple[int, Counter]]:
    total_pool = sum(counter[tile] for tile in tiles) + counter[LAIZI_TILE]
    options: List[Tuple[int, Counter]] = []
    for eggs in range(total_pool + 1):
        updated = _consume_special(counter, tiles, eggs)
        if updated is not None:
            options.append((eggs, updated))
    return options


def _enumerate_quads(counter: Counter, tiles: Iterable[str]) -> List[Tuple[int, Counter]]:
    work = counter.copy()
    max_total = 0
    for tile in tiles:
        max_total += work[tile] // 4
    if max_total == 0:
        return [(0, counter.copy())]

    def consume(k: int) -> Counter | None:
        temp = counter.copy()
        remaining = k
        for tile in tiles:
            available = temp[tile] // 4
            use = min(available, remaining)
            if use:
                temp[tile] -= use * 4
                remaining -= use
            if remaining == 0:
                break
        return temp if remaining == 0 else None

    options = [(0, counter.copy())]
    updated = consume(max_total)
    if updated is not None and max_total > 0:
        options.append((max_total, updated))
    return options


def _enumerate_counts_for_egg(counter: Counter, egg_key: str) -> List[Tuple[int, Counter]]:
    if egg_key in SPECIAL_GROUPS:
        return _enumerate_special(counter, SPECIAL_GROUPS[egg_key])
    if egg_key == "da_dan":
        return _enumerate_quads(counter, ("1s", "1p", "Z", "F", "B"))
    if egg_key == "gang_dan":
        excluded = {"1s", "1p", "Z", "F", "B"}
        tiles = [tile for tile in ID_TO_TILE if tile not in excluded]
        return _enumerate_quads(counter, tiles)
    return [(0, counter.copy())]


def _max_eggs_for_hand(counter: Counter) -> Tuple[int, Dict[str, int]]:
    best_total = 0
    best_counts = {egg.key: 0 for egg in EGG_TYPES}

    def dfs(idx: int, current: Counter, counts: Dict[str, int]) -> None:
        nonlocal best_total, best_counts
        if idx >= len(EGG_TYPES):
            total = sum(counts.values())
            if total > best_total:
                best_total = total
                best_counts = counts.copy()
            return

        egg = EGG_TYPES[idx]
        for amount, updated in _enumerate_counts_for_egg(current, egg.key):
            counts[egg.key] = amount
            dfs(idx + 1, updated, counts)
        counts[egg.key] = 0

    dfs(0, counter, {egg.key: 0 for egg in EGG_TYPES})
    return best_total, best_counts


def compute_multiplicity_stats(samples: HandSamples) -> MultiplicityStats:
    tiles = ID_TO_TILE[samples.array]
    total_counts = Counter()
    per_type_counts: Dict[str, Counter] = {egg.key: Counter() for egg in EGG_TYPES}

    for row in tiles:
        counter = Counter(row)
        total, per_type = _max_eggs_for_hand(counter)
        total_counts[total] += 1
        for egg in EGG_TYPES:
            per_type_counts[egg.key][per_type[egg.key]] += 1

    n = samples.array.shape[0]
    total_distribution = {k: v / n for k, v in sorted(total_counts.items())}
    per_type_distribution = {
        key: {cnt: count / n for cnt, count in sorted(dist.items())}
        for key, dist in per_type_counts.items()
    }
    return MultiplicityStats(total_distribution=total_distribution, per_type_distribution=per_type_distribution)


def save_multiplicity_stats(stats: MultiplicityStats, path: Path) -> None:
    path.write_text(
        json.dumps(stats.to_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_multiplicity_stats(path: Path) -> MultiplicityStats:
    data = json.loads(path.read_text(encoding="utf-8"))
    return MultiplicityStats.from_json(data)
