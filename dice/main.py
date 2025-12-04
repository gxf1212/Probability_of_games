import random
import json
from datetime import datetime
from collections import Counter

from game import Game
from players.strategies import (
    PlayerStrategy1_HonestExpectedValue,
    PlayerStrategy2_SafetyThreshold,
    PlayerStrategy3_BayesianInference,
    PlayerStrategy4_MinimumLoss,
    PlayerStrategy5_Hybrid
)

# --- 模拟参数配置 ---
CONFIG = {
    "num_games": 1,
    "num_rounds_per_game": 10000,
    "num_dice_per_player": 5,
    "initial_scores": 2000,
    "player_strategies": [
        PlayerStrategy1_HonestExpectedValue,
        PlayerStrategy2_SafetyThreshold,
        PlayerStrategy3_BayesianInference,
        PlayerStrategy4_MinimumLoss,
        PlayerStrategy5_Hybrid
    ],
    # 可以为特定策略调整参数
    "strategy_params": {
        "PlayerStrategy2_SafetyThreshold": {"safety_threshold": 0.55},
        "PlayerStrategy3_BayesianInference": {"belief_update_factor": 1.2}
    }
}


def run_simulation():
    """运行完整的多游戏模拟"""
    num_players = len(CONFIG["player_strategies"])

    # 初始化玩家实例
    players = []
    for i, strategy_class in enumerate(CONFIG["player_strategies"]):
        player_id = f"{strategy_class.__name__}_{i+1}"
        params = CONFIG["strategy_params"].get(strategy_class.__name__, {})
        players.append(strategy_class(player_id, num_players, CONFIG["num_dice_per_player"], **params))

    all_game_logs = []
    final_scores_agg = Counter()

    print(f"--- 开始模拟 {CONFIG['num_games']} 场游戏, 每场 {CONFIG['num_rounds_per_game']} 轮 ---")

    for i in range(CONFIG["num_games"]):
        game = Game(players, CONFIG["num_dice_per_player"], CONFIG["initial_scores"])
        final_scores = game.play_game(CONFIG["num_rounds_per_game"])

        for player_id, score in final_scores.items():
            final_scores_agg[player_id] += score

        all_game_logs.append(game.log)
        if (i+1) % 10 == 0:
            print(f"已完成 {i+1}/{CONFIG['num_games']} 场游戏...")

    # --- 保存结果 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"simulation_log_{timestamp}.json"

    output_data = {
        "config": {k: str(v) for k, v in CONFIG.items()},  # 将类转为字符串以便序列化
        "results": {
            "final_scores": final_scores_agg,
            "average_scores": {pid: score / CONFIG["num_games"] for pid, score in final_scores_agg.items()}
        },
        "logs": all_game_logs
    }

    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n--- 模拟结束 ---")
    print(f"详细对局记录已保存至: {log_filename}")

    # --- 命令行简单输出 ---
    print("\n--- 最终平均得分排名 ---")
    sorted_scores = sorted(output_data["results"]["average_scores"].items(), key=lambda item: item[1], reverse=True)

    for i, (player_id, avg_score) in enumerate(sorted_scores):
        print(f"{i+1}. {player_id:<45} 平均得分: {avg_score:.2f}")

    print("\n--- 算法平均决策时间 ---")
    for player in players:
        avg_time = player.get_average_decision_time()
        print(f"{str(player):<45} 平均决策时间: {avg_time * 1000:.4f} ms")


if __name__ == "__main__":
    run_simulation()
