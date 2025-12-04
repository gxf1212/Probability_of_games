import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.font_manager as fm


def run_simulation(num_dice=25, num_trials=1_000_000):
    """
    运行蒙特卡洛模拟来计算最大相同点数数量的分布。

    Args:
        num_dice (int): 骰子的总数。
        num_trials (int): 模拟的总次数。

    Returns:
        list: 包含每次试验中最大相同点数数量的列表。
    """
    max_counts = []
    for _ in range(num_trials):
        # 1. 模拟投掷25个骰子
        rolls = np.random.randint(1, 7, size=num_dice)

        # 2. 分类计数
        # 使用bincount可以高效地统计每个数字出现的次数
        # minlength=7确保1-6都有位置，尽管索引0不会被使用
        counts = np.bincount(rolls, minlength=7)

        num_wilds = counts[1]  # 点数1是赖子
        face_counts = counts[2:]  # 点数2, 3, 4, 5, 6的数量

        # 3. 计算最大数量
        # 找出2-6中出现次数最多的点数的数量
        max_of_faces = np.max(face_counts)

        # 将赖子全部加到数量最多的点数上，得到最终的最大值
        total_max = max_of_faces+num_wilds
        max_counts.append(total_max)

    return max_counts


def analyze_and_plot(results):
    """
    分析模拟结果并绘制概率分布柱状图。

    Args:
        results (list): 包含每次试验结果的列表。
    """
    num_trials = len(results)

    # 统计每个结果出现的频次
    count_freq = Counter(results)

    # 将频次转换为概率
    # 按点数数量排序
    sorted_counts = sorted(count_freq.keys())
    probabilities = [count_freq[count] / num_trials for count in sorted_counts]
    print(f"最大相同点数数量的概率分布：")
    for count, prob in zip(sorted_counts, probabilities):
        print(f"{count}: {prob:.3%}")

    # --- 可视化 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # 设置中文字体
    try:
        # 优先尝试常见的黑体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except:
        print("警告：未找到SimHei字体，可能无法正确显示中文。请尝试安装或指定其他中文字体。")

    bars = ax.bar(sorted_counts, probabilities, color='#4682B4', edgecolor='black', zorder=2)

    # 在柱状图上显示概率值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x()+bar.get_width() / 2.0, height, f'{height:.3%}', ha='center', va='bottom', fontsize=10)

    # 设置图表标题和坐标轴标签
    ax.set_title('25个骰子中最大相同点数数量的概率分布 (点数1为赖子)', fontsize=18, pad=20)
    ax.set_xlabel('最大相同点数的数量', fontsize=14)
    ax.set_ylabel('概率', fontsize=14)

    # 设置坐标轴刻度
    ax.set_xticks(range(min(sorted_counts), max(sorted_counts)+1))
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    # 添加网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加描述性文本
    fig.text(0.5, 0.02, f'基于 {num_trials:,} 次模拟投掷的结果', ha='center', style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


if __name__ == '__main__':
    # 运行主程序
    simulation_results = run_simulation()
    analyze_and_plot(simulation_results)
