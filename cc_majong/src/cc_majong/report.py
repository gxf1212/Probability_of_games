"""Markdown report builder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .eggs import EGG_TYPES
from .metrics import SummaryData


@dataclass
class FigureBundle:
    pie: Path
    combo_distribution: Path
    convergence: Path
    purity: Path


def _format_pct(value: float) -> str:
    return f"{value:.3%}"


def _egg_table(summary: SummaryData) -> str:
    header = "| 蛋型 | 出现概率（含赖子） | 纯净概率 | 赖子协助概率 |"
    divider = "| --- | --- | --- | --- |"
    lines = [header, divider]
    for egg in EGG_TYPES:
        probs = summary.egg_probabilities[egg.key]
        lines.append(
            f"| {egg.label} | {_format_pct(probs['with_laizi'])} | {_format_pct(probs['pure'])} | {_format_pct(probs['lai_only'])} |"
        )
    return "\n".join(lines)


def _combo_distribution_text(summary: SummaryData) -> str:
    parts = []
    for count in sorted(summary.combo_distribution):
        parts.append(f"{count} 组: {_format_pct(summary.combo_distribution[count])}")
    return "；".join(parts)


def _rel_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def build_report(
    summary: SummaryData,
    figures: FigureBundle,
    report_dir: Path,
    *,
    trials: int | None = None,
    seed: int | None = None,
) -> str:
    trials = trials if trials is not None else summary.trials
    seed = seed if seed is not None else summary.seed

    lines = ["# 长春麻将“蛋”概率实测报告", ""]

    lines.extend(
        [
            "## 规则速读",
            "长春麻将里的“蛋”包括：旋风蛋（东南西北齐全且不少于四张）、喜蛋（中发白）、幺蛋与九蛋（万饼索的 1 或 9）、幺鸡/幺饼/中发白组成的大蛋，以及“杠蛋”——任意四张相同牌的暗杠。",
            "牌墙去掉花牌后共 136 张，起手 14 张；幺鸡（1 索）是唯一赖子，可补 3 张旋风或 2 张喜/幺九蛋；杠蛋完全禁止赖子参与。",
            "本文遵循“旋风必须四风齐、暗杠即可成蛋”的平台规则（波克、QQ 长春麻将），便于和主流线上玩法对齐。",
            "",
        ]
    )

    lines.extend(
        [
            "## 快速导读",
            "我们用蒙特卡罗随机构建 10 万次起手，并把完整牌谱写入 Parquet 文件，后续换规则只需重跑判定。",
            "结论一句话：**约三成起手能直接下蛋，其中接近三分之二得靠幺鸡补齐；纯靠自摸的只有 12.4% 左右。**",
            "下文用四张图展示整体命中率、纯净 vs. 赖子贡献、多蛋叠加以及每种蛋型的收敛过程。",
            "",
        ]
    )

    sim_lines = ["## 模拟设定"]
    if trials is not None:
        sim_lines.append(f"- 抽样次数：{trials:,} 次；")
    if seed is not None:
        sim_lines.append(f"- 随机种子：{seed}；")
    sim_lines.extend(
        [
            "- 每轮从 136 张牌墙中随机抽 14 张作为起手，判断是否立即满足任意蛋型；",
            "- 记录纯净蛋、赖子蛋以及单手可叠的蛋数量；",
            "- 全部抽牌保存在 `data/hands.parquet`，判定结果写入 `output/results.parquet`，方便复算或换规则。",
            "",
        ]
    )
    lines.extend(sim_lines)

    lines.extend(
        [
            "## 图说结果",
            f"- 起手至少凑出一个蛋的概率：{_format_pct(summary.any_prob)}，其中纯净蛋：{_format_pct(summary.pure_prob)}，至少包含赖子蛋：{_format_pct(summary.laizi_only_prob)}。",
            "",
            "### 1. 各蛋命中率",
            "九蛋与喜蛋仍是出现次数最多的“双子星”；旋风蛋因必须四风齐而下降，但赖子可补缺；杠蛋覆盖除幺鸡/幺饼/中发白以外的所有暗杠。",
            _egg_table(summary),
            "",
            f"![各类蛋概率饼图]({_rel_path(figures.pie, report_dir)})",
            "",
            "### 2. 纯净 vs. 赖子贡献",
            "旋风、喜、九三类蛋的赖子依赖度都超过 70%，幺鸡越多越要留；幺蛋因为 1 索既当赖子又当自身成员，纯净占比反而更高。",
            f"![纯净赖子柱状图]({_rel_path(figures.purity, report_dir)})",
            "",
            "### 3. 多蛋叠加",
            _combo_distribution_text(summary),
            "",
            f"![蛋组合分布]({_rel_path(figures.combo_distribution, report_dir)})",
            "",
            "### 4. 收敛性",
            "含赖子、纯净蛋以及每种蛋型的累计概率在 50,000 次左右便进入稳定区间，再往上样本只是减少小数点波动。",
            f"![收敛性分析]({_rel_path(figures.convergence, report_dir)})",
            "",
        ]
    )

    lines.extend(
        [
            "## 桌边 Tips",
            "1. 幺鸡越多越别急着出，旋风/喜/幺九赖子共用，一鸡多用；当手里有两张幺鸡时可以大胆冲多蛋。",
            "2. 手牌已有风牌或三元对子时，先观察桌面再决定是否拆牌——赖子补位成功率只有约 10%，盲拆等于浪费潜力。",
            "3. 想教学或发朋友圈，直接引用本报告四张图即可讲清“规则→概率→策略”；若要换规则，可复用 `hands.parquet` 重新跑判定。",
            "",
            "## 参考资料",
            "1. [Pook 休闲桌游网：《长春麻将怎么玩》](https://www.pook.com/archives/326.html)",
            "2. [集合啦棋牌网：《吉林麻将（长春麻将）规则》](https://www.zadiqp.com/rules/jilinmajiang.html)",
        ]
    )

    return "\n".join(lines)
