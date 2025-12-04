# Probability of Games

本仓库聚合了两个独立的概率 / 博弈分析项目：

- `cc_majong`：长春麻将“蛋”牌型的蒙特卡罗模拟、差一张补牌概率及图文报告生成。
- `dice`：25 颗骰子的叫号/开牌策略研究，包含多种玩家策略与完整模拟脚本。

## 环境准备

1. 建议使用 Python 3.11+，并创建独立虚拟环境：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 请改用 .venv\Scripts\activate
   ```
2. 安装麻将项目所需依赖：
   ```bash
   pip install -r cc_majong/requirements.txt
   ```
   `dice` 子项目只依赖标准库，可直接运行。

## 仓库结构

```
Probability_of_games/
├── README.md                # 本文件
├── cc_majong/               # 长春麻将概率模拟
│   ├── data/                # 牌谱样本（hands.parquet 等）
│   ├── output/              # 统计结果、图表与报告
│   ├── requirements.txt     # numpy/pandas/matplotlib/pyarrow
│   └── src/cc_majong/       # 模拟、绘图、报告源码
└── dice/                    # 骰子策略模拟
    ├── players/             # 各类策略实现
    ├── main.py              # 入口，写入 simulation_log_*.json
    └── README.md            # 背景说明
```

## 子项目速览

### cc_majong

- 通过 `cc_majong.pipeline` 抽样 10 万副 14 张起手，输出 `output/results.parquet`、`summary.json` 以及四张核心图表。
- `next_draw.py` 可在相同样本上计算“差一张→下一摸”的命中概率，并生成 `data/next_draw*.json`（包含理论、实测、多线组合以及 multi-wait 统计四种口径）。
- `multiplicity.py` 会枚举“蛋的数量”（允许重复）并输出 `data/egg_multiplicity.json`，与 `plot_type_count_curves` 等图表配套。
- `plotting.py` 会创建中文图表（包含 `next_draw.png`），`report.py` 则把数据写入 `output/report.md`，适合直接发布到公众号。

常用命令：
```bash
# 重新生成整套模拟、汇总与图文报告
PYTHONPATH=cc_majong/src python -m cc_majong.pipeline --seed 20251204
```

在同一 `PYTHONPATH=cc_majong/src` 环境下，可直接运行下面的 Python 片段来重算“差一张”统计、保存 JSON，并刷新所有配套图表：

```python
from pathlib import Path

from cc_majong.samples import load_samples
from cc_majong.next_draw import (
    compute_next_draw_stats,
    compute_empirical_draw_stats,
    compute_combo_stats,
    compute_multi_wait_stats,
    compute_wait_line_stats,
    save_next_draw_stats,
    save_combo_stats,
    save_multi_wait_stats,
    save_wait_line_stats,
)
from cc_majong.multiplicity import (
    compute_multiplicity_stats,
    save_multiplicity_stats,
)
from cc_majong.plotting import (
    plot_next_draw_bars,
    plot_combo_waits,
    plot_multiplicity_distribution,
    plot_type_count_curves,
    plot_wait_line_counts,
    plot_wait_line_subset,
)

base = Path('data')
fig_dir = Path('output/figures')
fig_dir.mkdir(parents=True, exist_ok=True)

samples = load_samples(base / 'hands.parquet')
theory = compute_next_draw_stats(samples)
empirical = compute_empirical_draw_stats(samples)
save_next_draw_stats(theory, base / 'next_draw.json')
save_next_draw_stats(empirical, base / 'next_draw_empirical.json')

combos = compute_combo_stats(samples, min_size=2)
save_combo_stats(combos, base / 'next_draw_combos.json')
multi_wait = compute_multi_wait_stats(samples, min_waits=2)
save_multi_wait_stats(multi_wait, base / 'next_draw_multiwait.json')
wait_lines = compute_wait_line_stats(samples)
save_wait_line_stats(wait_lines, base / 'next_draw_wait_counts.json')
mult_stats = compute_multiplicity_stats(samples)
save_multiplicity_stats(mult_stats, base / 'egg_multiplicity.json')

plot_next_draw_bars(theory, empirical, fig_dir / 'next_draw.png')
plot_combo_waits(combos, fig_dir / 'multi_wait_combos.png', top_n=6)
plot_multiplicity_distribution(mult_stats, fig_dir / 'egg_multiplicity.png')
plot_type_count_curves(mult_stats, fig_dir / 'egg_type_counts.png')
plot_wait_line_counts(wait_lines, fig_dir / 'wait_line_counts.png')
plot_wait_line_subset(wait_lines, fig_dir / 'wait_line_subset.png')
```

### dice

- `main.py` 是总入口，使用 `players/strategies.py` 中定义的 5 种策略进行长轮次模拟。
- 对局日志与最终分数会保存在仓库根部的 `dice/simulation_log_*.json`。

常用命令：
```bash
python dice/main.py
```

## 输出位置

- `cc_majong/data/hands.parquet`：10 万副牌谱，可在不同规则下复用。
- `cc_majong/output/`：Parquet 结果、JSON 摘要、PNG 图以及 Markdown 报告。
- `dice/simulation_log_*.json`：每次骰子模拟的配置与结果。

## 贡献建议

- 新增 Python 依赖时，请更新相应子目录下的 `requirements.txt` 并在 README 中注明安装方法。
- 运行任何脚本前，务必通过 `PYTHONPATH=cc_majong/src`（或把该目录加入 `sys.path`）以便加载本地模块。
- 若拟扩展 Dice 策略，请在 `players/strategies.py` 中实现，并在 `main.py` 的 `CONFIG["player_strategies"]` 注册。

## 常见问题

1. **图表字体缺失怎么办？** `cc_majong/assets/fonts` 已提供「霞鹜文楷」轻量版，Matplotlib 会自动回退；如仍报错，可自行安装中文字体并设置 `MPLCONFIGDIR`。
2. **如何复现官方报告？** 运行 `PYTHONPATH=cc_majong/src python -m cc_majong.pipeline --seed 20251204` 重新生成结果，再按上文命令重算差一张统计并查看 `cc_majong/output/`。
