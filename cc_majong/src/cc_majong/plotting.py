"""Visualization helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from .eggs import EGG_TYPES
from .metrics import SummaryData

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
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
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
    for egg in EGG_TYPES:
        prob = summary.egg_probabilities[egg.key]["with_laizi"]
        label = CHINESE_LABELS.get(egg.key, egg.label)
        data.append((label, prob))
    data.append(("未出蛋", summary.none_prob()))

    labels = [f"{label}\n({prob:.1%})" for label, prob in data]
    sizes = [prob for _, prob in data]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=FIGURE_DPI)
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#ffffff")
    colors = plt.cm.Pastel2(np.linspace(0, 1, len(data)))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors, textprops={"color": "#333"})
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
    ax.set_ylabel("概率")
    ax.set_xticks(probs.index)
    for x, y in zip(probs.index, probs.values):
        ax.text(x, y + 0.002, f"{y:.2%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(probs.values) * 1.15)
    ax.set_title("蛋组合数量分布")

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
    ax.set_ylabel("概率")
    ax.set_title("纯净 vs. 赖子协助命中率")
    ax.legend()
    ax.set_ylim(0, max(max(pure), max(lai)) * 1.2)

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
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)

    _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
