"""Egg (special kong) definitions for Changchun Mahjong."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

from .samples import ALL_TILE_TYPES, LAIZI_TILE


@dataclass(frozen=True)
class EggPattern:
    """Concrete tile requirement for an egg type."""

    tiles: Tuple[Tuple[str, int], ...]
    laizi_cap: int = 0
    allows_laizi: bool = False

    @classmethod
    def from_dict(
        cls,
        requirements: Dict[str, int],
        *,
        laizi_cap: int = 0,
        allows_laizi: bool = False,
    ) -> "EggPattern":
        items = tuple(sorted(requirements.items()))
        return cls(items, laizi_cap=laizi_cap, allows_laizi=allows_laizi)

    def as_counter(self) -> Dict[str, int]:
        return dict(self.tiles)


@dataclass(frozen=True)
class EggType:
    key: str
    label: str
    description: str
    patterns: Tuple[EggPattern, ...]


EGG_TYPES: Tuple[EggType, ...] = (
    EggType(
        key="xuan_feng",
        label="旋风蛋",
        description="东南西北四风俱全。",
        patterns=(
            EggPattern.from_dict(
                {"E": 1, "S": 1, "W": 1, "N": 1},
                laizi_cap=3,
                allows_laizi=True,
            ),
        ),
    ),
    EggType(
        key="xi_dan",
        label="喜蛋",
        description="中发白三元牌齐全。",
        patterns=(
            EggPattern.from_dict(
                {"Z": 1, "F": 1, "B": 1},
                laizi_cap=2,
                allows_laizi=True,
            ),
        ),
    ),
    EggType(
        key="yao_dan",
        label="幺蛋",
        description="一万、一饼、一索各一张。",
        patterns=(
            EggPattern.from_dict(
                {"1m": 1, "1p": 1, "1s": 1},
                laizi_cap=2,
                allows_laizi=True,
            ),
        ),
    ),
    EggType(
        key="jiu_dan",
        label="九蛋",
        description="九万、九饼、九索各一张。",
        patterns=(
            EggPattern.from_dict(
                {"9m": 1, "9p": 1, "9s": 1},
                laizi_cap=2,
                allows_laizi=True,
            ),
        ),
    ),
    EggType(
        key="da_dan",
        label="大蛋",
        description="由幺鸡、幺饼、红中、发财、白板组成的任意明暗杠。",
        patterns=(
            EggPattern.from_dict({"1s": 4}),
            EggPattern.from_dict({"1p": 4}),
            EggPattern.from_dict({"Z": 4}),
            EggPattern.from_dict({"F": 4}),
            EggPattern.from_dict({"B": 4}),
        ),
    ),
    EggType(
        key="gang_dan",
        label="杠蛋",
        description="任意四张相同牌的暗杠（不可用赖子）。",
        patterns=tuple(
            EggPattern.from_dict({tile: 4})
            for tile in ALL_TILE_TYPES
            if tile not in {"1s", "1p", "Z", "F", "B"}
        ),
    ),
)


TYPE_LOOKUP = {egg.key: egg for egg in EGG_TYPES}
TYPE_LABELS = {egg.key: egg.label for egg in EGG_TYPES}
