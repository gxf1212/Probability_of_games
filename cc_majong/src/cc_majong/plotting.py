"""Visualization helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from .eggs import EGG_TYPES
from .metrics import SummaryData
from .multiplicity import MultiplicityStats
from .next_draw import MultiWaitStats, NextDrawStats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FONT_CHOICES = [
    ("LXGW WenKai Lite", PROJECT_ROOT / "assets" / "fonts" / "LXGWWenKaiLite-Regular.ttf"),
]
for family, path in FONT_CHOICES:
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        plt.rcParams["font.family"] = family
        break
else:
    plt.rcParams.setdefault("font.sans-serif", ["SimHei", "Microsoft YaHei", "sans-serif"])
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 19,
    "axes.titlesize": 23,
    "axes.labelsize": 19,
    "legend.fontsize": 17,
})

CHINESE_LABELS = {
    "xuan_feng": "旋风蛋",
    "xi_dan": "喜蛋",
    "yao_dan": "幺蛋",
    "jiu_dan": "九蛋",
    "da_dan": "大蛋",
    "gang_dan": "杠蛋",
}


FIGURE_DPI = 200


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_egg_pie(summary: SummaryData, output_path: Path) -> None:
    data = []
    merged_big = 0.0
    for egg in EGG_TYPES:
        prob = summary.egg_probabilities[egg.key]["with_laizi"]
        if egg.key in {"da_dan", "gang_dan"}:
            merged_big += prob
            continue
        label = CHINESE_LABELS.get(egg.key, egg.label)
        data.append((label, prob))
    if merged_big > 0:
        data.append(("大蛋/杠蛋", merged_big))
    data.append(("未出蛋", summary.none_prob()))

    labels = [label for label, _ in data]
    sizes = [prob for _, prob in data]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=FIGURE_DPI)
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#ffffff")
    colors = plt.cm.Pastel2(np.linspace(0, 1, len(data)))
    wedges, _ = ax.pie(sizes, startangle=90, colors=colors)
    for wedge, (label, prob) in zip(wedges, data):
        ang = (wedge.theta2 + wedge.theta1) / 2
        x = 0.7 * np.cos(np.deg2rad(ang))
        y = 0.7 * np.sin(np.deg2rad(ang))
        ax.text(x, y, f"{label}\n{prob:.1%}", ha="center", va="center", fontsize=14, color="#333")
    ax.set_title("起手各类蛋出现概率")
    ax.axis('equal')

    _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_combo_distribution(summary: SummaryData, output_path: Path) -> None:
    probs = pd.Series(summary.combo_distribution).sort_index()

    fig, ax = plt.subplots(figsize=(7, 4), dpi=FIGURE_DPI)
    ax.bar(probs.index, probs.values, width=0.6, color="#4C72B0")
    ax.set_xlabel("单手可凑蛋的数量（含赖子）")
    ax.set_ylabel("概率 (%)")
    ax.set_xticks(probs.index)
    for x, y in zip(probs.index, probs.values):
        ax.text(x, y + 0.002, f"{y:.2%}", ha="center", va="bottom", fontsize=15)
    ax.set_ylim(0, max(probs.values) * 1.15)
    ax.set_title("蛋组合数量分布")
    ax.yaxis.set_major_formatter(lambda val, _: f"{val*100:.0f}%")

    _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_purity_bars(summary: SummaryData, output_path: Path) -> None:
    labels = [CHINESE_LABELS.get(egg.key, egg.label) for egg in EGG_TYPES]
    pure = [summary.egg_probabilities[egg.key]["pure"] for egg in EGG_TYPES]
    lai = [summary.egg_probabilities[egg.key]["lai_only"] for egg in EGG_TYPES]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=FIGURE_DPI)
    ax.bar(x - width / 2, pure, width, label="纯净概率", color="#6baed6")
    ax.bar(x + width / 2, lai, width, label="赖子协助概率", color="#fd8d3c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("概率 (%)")
    ax.set_title("纯净 vs. 赖子协助命中率")
    ax.legend()
    ymax = max(max(pure), max(lai)) * 1.2
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(lambda val, _: f"{val*100:.0f}%")

    _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_convergence(summary: SummaryData, output_path: Path) -> None:
    frame = summary.convergence_frame()

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=FIGURE_DPI)
    color_cycle = plt.cm.tab20(np.linspace(0, 1, frame.shape[1] - 1))
    for idx, column in enumerate(frame.columns[1:]):
        ax.plot(frame["trial"], frame[column], label=column, color=color_cycle[idx])
    ax.set_xlabel("模拟次数")
    ax.set_ylabel("累计概率估计")
    ax.set_title("蛋型收敛曲线")
    ax.legend(ncol=2, fontsize=14)
    ax.grid(alpha=0.2)

    _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_next_draw_bars(theory: NextDrawStats, empirical: NextDrawStats, output_path: Path) -> None:
    labels = [CHINESE_LABELS.get(egg.key, egg.label) for egg in EGG_TYPES]
    theory_total = [theory.per_egg[egg.key]["total"] for egg in EGG_TYPES]
    empirical_total = [empirical.per_egg[egg.key]["total"] for egg in EGG_TYPES]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=FIGURE_DPI)
    ax.bar(x - width / 2, theory_total, width, color="#6c8ebf", label="理论")
    ax.bar(x + width / 2, empirical_total, width, color="#f6a05a", label="实测")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("下一摸命中率 (%)")
    ax.set_title("差一张→下一摸触发概率")
    ax.legend()

    ymax = max(max(theory_total), max(empirical_total)) * 1.25 if theory_total and empirical_total else 0.05
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(lambda val, _: f"{val*100:.0f}%")



    fig.tight_layout()
    _ensure_parent(output_path)
    fig.savefig(output_path)
    plt.close(fig)


def plot_multiplicity_distribution(stats: MultiplicityStats, output_path: Path) -> None:
    items = sorted(stats.total_distribution.items())
    x = [count for count, _ in items]
    y = [prob for _, prob in items]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=FIGURE_DPI)
    ax.bar(x, y, width=0.5, color="#7b8dda")
    ax.set_xlabel("单手可组成蛋的总数量（可重复计）")
    ax.set_ylabel("概率 (%)")
    ax.set_title("最大蛋数量分布")
    ax.set_xticks(x)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.005, f"{yi:.2%}", ha="center", va="bottom", fontsize=14)
    ax.yaxis.set_major_formatter(lambda val, _: f"{val*100:.0f}%")

    _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_type_count_curves(stats: MultiplicityStats, output_path: Path) -> None:
    counts = sorted({k for dist in stats.per_type_distribution.values() for k in dist.keys()})
    x = np.arange(len(counts))
    n_types = len(EGG_TYPES)
    width = 0.8 / n_types if n_types else 0.5
    offsets = (np.arange(n_types) - (n_types - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(9, 5), dpi=FIGURE_DPI)
    for idx, egg in enumerate(EGG_TYPES):
        dist = stats.per_type_distribution[egg.key]
        values = [dist.get(count, 0.0) for count in counts]
        bars = ax.bar(x + offsets[idx], values, width, label=CHINESE_LABELS.get(egg.key, egg.label))
        for bar, val in zip(bars, values):
            if val <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    ax.set_xlabel("该蛋型可同时凑出的数量")
    ax.set_ylabel("概率 (%)")
    ax.set_title("各蛋型可叠数量分布")
    ax.set_xticks(x)
    ax.set_xticklabels(counts)
    ax.yaxis.set_major_formatter(lambda val, _: f"{val*100:.1f}%")
    ax.set_ylim(0, max(max(dist.values()) for dist in stats.per_type_distribution.values()) * 1.2)
    ax.legend(ncol=2)

    _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _combo_label(combo_key: str) -> str:
    parts = [CHINESE_LABELS.get(key, key) for key in combo_key.split("+")]
    parts = [name.replace("蛋", "") for name in parts]
    return "+".join(parts)


def plot_combo_waits(combo_stats: Dict[str, Dict[str, float]], output_path: Path, top_n: int = 6) -> None:
    if not combo_stats:
        return
    items = sorted(combo_stats.items(), key=lambda kv: kv[1]["hand_freq"], reverse=True)[:top_n]
    labels = [_combo_label(key) for key, _ in items]
    theory = [data["pure_prob"] + data["lai_prob"] for _, data in items]
    empirical = [data["emp_pure"] + data["emp_lai"] for _, data in items]

    x = np.arange(len(items))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=FIGURE_DPI)
    ax.bar(x - width / 2, theory, width, color="#8ecae6", label="理论命中")
    ax.bar(x + width / 2, empirical, width, color="#ffb703", label="实测命中")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("下一摸命中率 (%)")
    ax.set_title("多线差一张：下一摸触发率（Top组合）")
    ax.yaxis.set_major_formatter(lambda val, _: f"{val*100:.0f}%")
    ymax = max(max(theory), max(empirical)) * 1.2 if theory and empirical else 0.1
    ax.set_ylim(0, ymax)
    ax.legend()

    _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_multi_wait_summary(stats: MultiWaitStats, output_path: Path) -> None:
    labels = ["多线待牌占比", "理论下一摸命中", "实测下一摸命中"]
    values = [stats.frequency, stats.theory_prob, stats.empirical_prob]

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=FIGURE_DPI)
    bars = ax.bar(labels, values, color=["#90caf9", "#a5d6a7", "#ffcc80"])
    ax.set_ylabel("比例 (%)")
    ax.set_title("双线及以上差一张：规模与命中率")
    ax.yaxis.set_major_formatter(lambda val, _: f"{val*100:.0f}%")
    ymax = max(values) if values else 0.0
    if ymax == 0:
        ymax = 0.05
    ax.set_ylim(0, ymax * 1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.1%}", ha="center", va="bottom", fontsize=14)

    _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
