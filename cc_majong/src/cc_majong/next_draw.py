"""Estimate next-draw completion probabilities for egg patterns."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple
from itertools import combinations

from .analysis import type_feasible
from .eggs import EGG_TYPES
from .samples import ALL_TILE_TYPES, HandSamples, ID_TO_TILE, LAIZI_TILE

UNSEEN_TILES = 123  # 136 total - 自己手牌 13 张


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
    if egg_key == "gang_dan":
        candidate_tiles = tuple(tile for tile in ALL_TILE_TYPES if counter[tile] == 3)
    else:
        candidate_tiles = _candidate_tiles(egg_key)
    for tile in candidate_tiles:
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
        satisfied = {
            egg.key: type_feasible(counter, egg, allow_laizi=True)[0]
            for egg in EGG_TYPES
        }
        available = {tile: max(0, 4 - counter[tile]) for tile in ALL_TILE_TYPES}
        union_tiles: Set[str] = set()
        for egg in EGG_TYPES:
            if satisfied[egg.key]:
                continue
            pure_tiles, laizi_tiles = _winning_tiles(counter, egg.key)
            per_egg_counts[egg.key]["pure"] += sum(available[t] for t in pure_tiles)
            per_egg_counts[egg.key]["lai"] += sum(available[t] for t in laizi_tiles)
            union_tiles.update(pure_tiles)
            union_tiles.update(laizi_tiles)
        union_count += sum(available[t] for t in union_tiles)
    denom = len(hands) * UNSEEN_TILES
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


def compute_empirical_draw_stats(samples: HandSamples) -> NextDrawStats:
    per_counts = {egg.key: {"pure": 0.0, "lai": 0.0} for egg in EGG_TYPES}
    union_count = 0.0
    tiles = ID_TO_TILE[samples.array]
    for row in tiles:
        base = Counter(row[:13])
        draw_tile = row[13]
        hit_any = False
        for egg in EGG_TYPES:
            if type_feasible(base, egg, allow_laizi=True)[0]:
                continue
            with_draw = base.copy()
            with_draw[draw_tile] += 1
            pure_hit = type_feasible(with_draw, egg, allow_laizi=False)[0]
            lai_hit, _ = type_feasible(with_draw, egg, allow_laizi=True)
            if not lai_hit:
                continue
            if pure_hit:
                per_counts[egg.key]["pure"] += 1
            else:
                per_counts[egg.key]["lai"] += 1
            hit_any = True
        if hit_any:
            union_count += 1
    total = samples.trials
    per_egg_probs = {
        key: {
            "pure": counts["pure"] / total,
            "lai": counts["lai"] / total,
            "total": (counts["pure"] + counts["lai"]) / total,
        }
        for key, counts in per_counts.items()
    }
    return NextDrawStats(per_egg=per_egg_probs, union_prob=union_count / total)


def _completed_flags(counter: Counter) -> Dict[str, bool]:
    return {egg.key: type_feasible(counter, egg, allow_laizi=True)[0] for egg in EGG_TYPES}


def compute_combo_stats(samples: HandSamples, min_size: int = 2) -> Dict[str, Dict[str, float]]:
    tiles = ID_TO_TILE[samples.array]
    total_hands = tiles.shape[0]
    combos: Dict[Tuple[str, ...], Dict[str, float]] = {}

    for hand in tiles:
        base = Counter(hand[:13])
        draw_tile = hand[13]
        satisfied = _completed_flags(base)
        if sum(satisfied.values()) > 1:
            continue
        available = {tile: max(0, 4 - base[tile]) for tile in ALL_TILE_TYPES}

        egg_candidates: Dict[str, Tuple[Set[str], Set[str]]] = {}
        for egg in EGG_TYPES:
            if satisfied[egg.key]:
                continue
            pure_tiles, laizi_tiles = _winning_tiles(base, egg.key)
            if pure_tiles or laizi_tiles:
                egg_candidates[egg.key] = (pure_tiles, laizi_tiles)

        keys = sorted(egg_candidates.keys())
        if len(keys) < min_size:
            continue

        for r in range(min_size, len(keys) + 1):
            for combo in combinations(keys, r):
                entry = combos.setdefault(
                    combo,
                    {
                        "hand_count": 0.0,
                        "pure_sum": 0.0,
                        "lai_sum": 0.0,
                        "emp_pure": 0.0,
                        "emp_lai": 0.0,
                    },
                )
                entry["hand_count"] += 1

                pure_tiles_union: Set[str] = set()
                lai_tiles_union: Set[str] = set()
                for key in combo:
                    pure_tiles_union.update(egg_candidates[key][0])
                    lai_tiles_union.update(egg_candidates[key][1])

                pure_sum = sum(available[t] for t in pure_tiles_union)
                lai_sum = sum(available[t] for t in lai_tiles_union - pure_tiles_union)
                entry["pure_sum"] += pure_sum
                entry["lai_sum"] += lai_sum

                if draw_tile in pure_tiles_union:
                    entry["emp_pure"] += 1
                elif draw_tile in lai_tiles_union:
                    entry["emp_lai"] += 1

    combo_stats: Dict[str, Dict[str, float]] = {}
    for combo, data in combos.items():
        hand_count = data["hand_count"]
        denom = hand_count * UNSEEN_TILES
        label = "+".join(combo)
        combo_stats[label] = {
            "hand_freq": hand_count / total_hands,
            "pure_prob": data["pure_sum"] / denom if denom else 0.0,
            "lai_prob": data["lai_sum"] / denom if denom else 0.0,
            "emp_pure": data["emp_pure"] / hand_count if hand_count else 0.0,
            "emp_lai": data["emp_lai"] / hand_count if hand_count else 0.0,
        }
    return combo_stats


def save_combo_stats(stats: Dict[str, Dict[str, float]], path: Path) -> None:
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def save_next_draw_stats(stats: NextDrawStats, path: Path) -> None:
    path.write_text(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_next_draw_stats(path: Path) -> NextDrawStats:
    return NextDrawStats.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class MultiWaitStats:
    frequency: float
    theory_prob: float
    empirical_prob: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "frequency": self.frequency,
            "theory_prob": self.theory_prob,
            "empirical_prob": self.empirical_prob,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "MultiWaitStats":
        return cls(
            frequency=float(data["frequency"]),
            theory_prob=float(data["theory_prob"]),
            empirical_prob=float(data["empirical_prob"]),
        )


def compute_multi_wait_stats(samples: HandSamples, min_waits: int = 2) -> MultiWaitStats:
    tiles = ID_TO_TILE[samples.array]
    total_hands = tiles.shape[0]
    qualifying = 0
    theory_hits = 0.0
    empirical_hits = 0.0

    for hand in tiles:
        base = Counter(hand[:13])
        satisfied = _completed_flags(base)
        if sum(satisfied.values()) > 1:
            continue

        available = {tile: max(0, 4 - base[tile]) for tile in ALL_TILE_TYPES}
        egg_candidates: Dict[str, Tuple[Set[str], Set[str]]] = {}
        for egg in EGG_TYPES:
            if satisfied[egg.key]:
                continue
            pure_tiles, laizi_tiles = _winning_tiles(base, egg.key)
            if pure_tiles or laizi_tiles:
                egg_candidates[egg.key] = (pure_tiles, laizi_tiles)

        if len(egg_candidates) < min_waits:
            continue

        qualifying += 1
        pure_union: Set[str] = set()
        lai_union: Set[str] = set()
        for pure_tiles, laizi_tiles in egg_candidates.values():
            pure_union.update(pure_tiles)
            lai_union.update(laizi_tiles)

        theory_hits += sum(available[t] for t in pure_union)
        theory_hits += sum(available[t] for t in lai_union - pure_union)

        draw_tile = hand[13]
        if draw_tile in pure_union or draw_tile in lai_union:
            empirical_hits += 1

    if qualifying == 0:
        return MultiWaitStats(frequency=0.0, theory_prob=0.0, empirical_prob=0.0)

    frequency = qualifying / total_hands
    theory_prob = theory_hits / (qualifying * UNSEEN_TILES)
    empirical_prob = empirical_hits / qualifying
    return MultiWaitStats(frequency=frequency, theory_prob=theory_prob, empirical_prob=empirical_prob)


def save_multi_wait_stats(stats: MultiWaitStats, path: Path) -> None:
    path.write_text(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_multi_wait_stats(path: Path) -> MultiWaitStats:
    return MultiWaitStats.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class WaitLineStats:
    counts_all: Dict[int, int]
    counts_subset: Dict[int, int]
    total: int
    subset_total: int

    def to_dict(self) -> Dict[str, object]:
        prob_all = {str(k): (self.counts_all.get(k, 0) / self.total) for k in sorted(self.counts_all)}
        prob_subset = {
            str(k): (self.counts_subset.get(k, 0) / self.subset_total) if self.subset_total else 0.0
            for k in sorted(self.counts_subset)
        }
        return {
            "counts_all": {str(k): v for k, v in sorted(self.counts_all.items())},
            "prob_all": prob_all,
            "counts_subset": {str(k): v for k, v in sorted(self.counts_subset.items())},
            "prob_subset": prob_subset,
            "total": self.total,
            "subset_total": self.subset_total,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "WaitLineStats":
        counts_all = {int(k): int(v) for k, v in data["counts_all"].items()}
        counts_subset = {int(k): int(v) for k, v in data["counts_subset"].items()}
        return cls(
            counts_all=counts_all,
            counts_subset=counts_subset,
            total=int(data["total"]),
            subset_total=int(data["subset_total"]),
        )

    def prob_all(self, k: int) -> float:
        return self.counts_all.get(k, 0) / self.total if self.total else 0.0

    def prob_subset(self, k: int) -> float:
        return self.counts_subset.get(k, 0) / self.subset_total if self.subset_total else 0.0


def compute_wait_line_stats(samples: HandSamples) -> WaitLineStats:
    counts_all: Counter[int] = Counter()
    counts_subset: Counter[int] = Counter()
    total = samples.array.shape[0]
    subset_total = 0

    for hand in samples.array:
        base = Counter(ID_TO_TILE[hand[:13]])
        if any(type_feasible(base, egg, allow_laizi=True)[0] for egg in EGG_TYPES):
            continue
        waiting = 0
        for egg in EGG_TYPES:
            pure_tiles, laizi_tiles = _winning_tiles(base, egg.key)
            if pure_tiles or laizi_tiles:
                waiting += 1
        counts_all[waiting] += 1
        if waiting > 0:
            counts_subset[waiting] += 1
            subset_total += 1

    return WaitLineStats(
        counts_all=dict(counts_all),
        counts_subset=dict(counts_subset),
        total=total,
        subset_total=subset_total,
    )


def save_wait_line_stats(stats: WaitLineStats, path: Path) -> None:
    path.write_text(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_wait_line_stats(path: Path) -> WaitLineStats:
    return WaitLineStats.from_dict(json.loads(path.read_text(encoding="utf-8")))
