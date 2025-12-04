"""Detection utilities for Changchun Mahjong eggs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .eggs import EggPattern, EggType, EGG_TYPES
from .samples import LAIZI_TILE


@dataclass
class PatternMatch:
    remaining: Counter
    laizi_used: int


def _try_consume_pattern(counter: Counter, pattern: EggPattern, allow_laizi: bool) -> Optional[PatternMatch]:
    """Try to remove the tiles required for a pattern from ``counter``."""
    requirements = dict(pattern.tiles)
    work = counter.copy()
    laizi_cap = pattern.laizi_cap if (allow_laizi and pattern.allows_laizi) else 0
    laizi_used = 0

    for tile, needed in requirements.items():
        available = work[tile]
        use = min(available, needed)
        work[tile] -= use
        remaining_need = needed - use
        if remaining_need > 0:
            if laizi_cap <= 0:
                return None
            if laizi_cap < remaining_need:
                return None
            if work[LAIZI_TILE] < remaining_need:
                return None
            work[LAIZI_TILE] -= remaining_need
            laizi_cap -= remaining_need
            laizi_used += remaining_need
    return PatternMatch(remaining=work, laizi_used=laizi_used)


def type_feasible(counter: Counter, egg_type: EggType, allow_laizi: bool) -> Tuple[bool, bool]:
    """Return (feasible, uses_laizi)."""
    for pattern in egg_type.patterns:
        result = _try_consume_pattern(counter, pattern, allow_laizi)
        if result is not None:
            return True, result.laizi_used > 0
    return False, False


def count_types(counter: Counter, allow_laizi: bool) -> Dict[str, Tuple[bool, bool]]:
    """Return feasibility for each egg type."""
    summary: Dict[str, Tuple[bool, bool]] = {}
    for egg_type in EGG_TYPES:
        feasible, uses_laizi = type_feasible(counter, egg_type, allow_laizi)
        summary[egg_type.key] = (feasible, uses_laizi)
    return summary


def _counter_key(counter: Counter) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((tile, count) for tile, count in counter.items() if count > 0))


def _search_max_types(
    counter: Counter, allow_laizi: bool, idx: int = 0, cache=None
) -> Tuple[int, Tuple[str, ...]]:
    if cache is None:
        cache = {}
    key = (idx, _counter_key(counter))
    cached = cache.get(key)
    if cached is not None:
        return cached

    if idx >= len(EGG_TYPES):
        return (0, tuple())

    # Option 1: skip current type
    best_count, best_types = _search_max_types(counter, allow_laizi, idx + 1, cache)

    egg_type = EGG_TYPES[idx]
    for pattern in egg_type.patterns:
        result = _try_consume_pattern(counter, pattern, allow_laizi)
        if result is None:
            continue
        sub_count, sub_types = _search_max_types(result.remaining, allow_laizi, idx + 1, cache)
        total_count = 1 + sub_count
        total_types = tuple(sorted(sub_types + (egg_type.key,)))
        if total_count > best_count:
            best_count = total_count
            best_types = total_types
    cache[key] = (best_count, best_types)
    return cache[key]


def max_egg_combo(counter: Counter, allow_laizi: bool) -> Tuple[int, Tuple[str, ...]]:
    return _search_max_types(counter, allow_laizi)


def hand_to_counter(hand: Sequence[str]) -> Counter:
    return Counter(hand)
