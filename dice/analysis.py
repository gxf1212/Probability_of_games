import json
import argparse
from collections import defaultdict
import pandas as pd


class LogAnalyzer:
    """
    用于解析和分析吹牛骰子模拟器日志文件的类。
    """

    def __init__(self, filepath):
        print(f"Loading log file: {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            self.log_data = json.load(f)
        self.config = self.log_data.get("config", {})
        self.all_events = self._flatten_logs()
        self.df = pd.DataFrame(self.all_events)
        print("Log file processed successfully.")

    def _get_strategy_name(self, player_id):
        """从 player_id 中提取基础策略名。"""
        if not player_id: return ""
        return "_".join(player_id.split('_')[:-1])

    def _flatten_logs(self):
        """将嵌套的日志结构扁平化，并为每一轮创建唯一ID。"""
        flat_list = []
        for game_idx, game_log in enumerate(self.log_data.get("logs", [])):
            current_hands = {}
            player_order = []
            round_events = []
            for event in game_log:
                # 每轮开始时重置回合事件
                if event.get('type') == 'round_start':
                    if round_events:
                        flat_list.extend(round_events)
                    round_events = []
                    current_hands = event.get('hands', {})
                    player_order = event.get('player_order', [])

                event['unique_round_id'] = f"g{game_idx}_r{event.get('round', -1)}"
                event['strategy'] = self._get_strategy_name(event.get('player', ''))
                event['hands'] = current_hands
                event['player_order'] = player_order
                round_events.append(event)
            flat_list.extend(round_events)  # 添加最后一轮的事件
        return flat_list

    def _calculate_true_count(self, hands, face):
        """计算场上某个点数的真实数量（含赖子）。"""
        if not isinstance(hands, dict): return 0
        all_dice = [d for hand in hands.values() for d in hand]
        return all_dice.count(1)+all_dice.count(face)

    def _evaluate_hand_strength(self, hand):
        """评估手牌强度：赖子数 + 最多同点数的数量。"""
        if not hand: return 0
        wilds = hand.count(1)
        counts = defaultdict(int)
        for card in hand:
            if card != 1:
                counts[card] += 1
        max_face_count = max(counts.values()) if counts else 0
        return wilds+max_face_count

    def run_full_analysis(self):
        """运行所有分析并打印报告。"""
        self.analyze_challenge_behavior()
        self.analyze_bluffing_behavior()
        self.analyze_situational_performance()

    def analyze_challenge_behavior(self):
        """1. 分析挑战行为：频率与准确率。"""
        print("\n"+"=" * 80)
        print("ANALYSIS 1: CHALLENGE BEHAVIOR (挑战行为分析)")
        print("=" * 80)

        decisions_df = self.df[self.df['type'] == 'decision'].copy()

        # **FIX:** 使用更精确的数据处理流程，避免重复计算
        round_ends_df = self.df[self.df['reason'] == 'challenge_resolved'][
            ['unique_round_id', 'loser']].drop_duplicates(subset=['unique_round_id'], keep='last')
        loser_map = round_ends_df.set_index('unique_round_id')['loser'].to_dict()

        results = []
        unique_strategies = sorted([s for s in self.df['strategy'].unique() if s])
        for strategy in unique_strategies:

            player_decisions_df = decisions_df[decisions_df['strategy'] == strategy]
            total_decisions = len(player_decisions_df)

            player_challenges_df = player_decisions_df[
                player_decisions_df['decision'].apply(lambda d: d and d[0] == 'challenge')].copy()
            num_challenges = len(player_challenges_df)

            challenge_freq = (num_challenges / total_decisions) if total_decisions > 0 else 0

            if num_challenges > 0:
                player_challenges_df['loser'] = player_challenges_df['unique_round_id'].map(loser_map)
                player_challenges_df.fillna({'loser': ''}, inplace=True)

                player_challenges_df['challenge_won'] = player_challenges_df['player'] != player_challenges_df['loser']
                wins = player_challenges_df['challenge_won'].sum()
            else:
                wins = 0

            accuracy = (wins / num_challenges) if num_challenges > 0 else 0
            results.append({
                "Strategy": strategy,
                "Challenge Freq.": f"{challenge_freq:.2%}",
                "Challenges Made": num_challenges,
                "Challenge Accuracy": f"{accuracy:.2%}",
                "Correct Challenges": int(wins)
            })

        report_df = pd.DataFrame(results).sort_values("Challenge Accuracy", ascending=False)
        print(report_df.to_string(index=False))
        print("\n- Challenge Freq.: 该策略做出'开'的决策占其所有决策的比例。")
        print("- Challenge Accuracy: 该策略做出'开'的决策后，赢的概率。")

    def analyze_bluffing_behavior(self):
        """2. 分析欺骗行为：频率与成功率。"""
        print("\n"+"=" * 80)
        print("ANALYSIS 2: BLUFFING BEHAVIOR (欺骗行为分析)")
        print("=" * 80)

        decisions_df = self.df[self.df['type'] == 'decision'].copy()
        bids_df = decisions_df[decisions_df['decision'].apply(lambda d: d and d[0] == 'bid')].copy()

        bids_df['true_count'] = bids_df.apply(lambda row: self._calculate_true_count(row['hands'], row['decision'][2]),
                                              axis=1)
        bids_df['is_bluff'] = bids_df.apply(lambda row: row['decision'][1] > row['true_count'], axis=1)

        next_decision_action = decisions_df['decision'].shift(-1).apply(
            lambda d: d[0] if isinstance(d, (list, tuple)) and d else None)
        is_in_same_round = decisions_df['unique_round_id'].shift(-1) == decisions_df['unique_round_id']
        decisions_df['bluff_failed'] = (next_decision_action == 'challenge') & is_in_same_round

        bids_df['bluff_failed'] = bids_df.index.map(decisions_df['bluff_failed'])
        bids_df['bluff_successful'] = bids_df['is_bluff'] & ~bids_df['bluff_failed']

        results = []
        unique_strategies = sorted([s for s in self.df['strategy'].unique() if s])
        for strategy in unique_strategies:
            player_bids = bids_df[bids_df['strategy'] == strategy]
            total_bids = len(player_bids)
            num_bluffs = player_bids['is_bluff'].sum()
            num_successful_bluffs = player_bids['bluff_successful'].sum()

            bluff_freq = (num_bluffs / total_bids) if total_bids > 0 else 0
            success_rate = (num_successful_bluffs / num_bluffs) if num_bluffs > 0 else 0

            results.append({
                "Strategy": strategy,
                "Bluff Freq.": f"{bluff_freq:.2%}",
                "Total Bluffs": int(num_bluffs),
                "Bluff Success Rate": f"{success_rate:.2%}",
                "Successful Bluffs": int(num_successful_bluffs)
            })

        report_df = pd.DataFrame(results).sort_values("Bluff Success Rate", ascending=False)
        print(report_df.to_string(index=False))
        print("\n- Bluff Freq.: 该策略的叫号中，超过场上实际数量（即欺骗）的比例。")
        print("- Bluff Success Rate: 该策略欺骗后，下家没有'开'（即欺骗成功）的概率。")

    def analyze_situational_performance(self):
        """3. 分析情境表现：顺风/逆风局表现。"""
        print("\n"+"=" * 80)
        print("ANALYSIS 3: SITUATIONAL PERFORMANCE (情境表现分析)")
        print("=" * 80)

        rounds_df = self.df[self.df['type'] == 'round_start'].copy()

        # **FIX:** 使用同样健壮的 map 逻辑
        losers_df_unique = self.df[self.df['type'] == 'round_end'][['unique_round_id', 'loser']].drop_duplicates(
            subset=['unique_round_id'], keep='last')
        loser_map = losers_df_unique.set_index('unique_round_id')['loser'].to_dict()

        rounds_df['loser'] = rounds_df['unique_round_id'].map(loser_map)
        rounds_df.fillna({'loser': ''}, inplace=True)

        player_data = []
        for _, row in rounds_df.iterrows():
            if isinstance(row['hands'], dict):
                for player_id, hand in row['hands'].items():
                    strength = self._evaluate_hand_strength(hand)
                    lost_round = (player_id == row['loser'])
                    player_data.append({
                        "strategy": self._get_strategy_name(player_id),
                        "hand_strength": strength,
                        "lost": lost_round
                    })

        perf_df = pd.DataFrame(player_data)

        good_hand_threshold = 4
        bad_hand_threshold = 2

        results = []
        unique_strategies = sorted([s for s in self.df['strategy'].unique() if s])
        for strategy in unique_strategies:
            if not strategy: continue

            strat_df = perf_df[perf_df['strategy'] == strategy]

            good_hands = strat_df[strat_df['hand_strength'] >= good_hand_threshold]
            bad_hands = strat_df[strat_df['hand_strength'] <= bad_hand_threshold]

            loss_rate_good = good_hands['lost'].mean() if not good_hands.empty else 0
            loss_rate_bad = bad_hands['lost'].mean() if not bad_hands.empty else 0

            results.append({
                "Strategy": strategy,
                "Loss Rate (Good Hand)": f"{loss_rate_good:.2%}",
                "Loss Rate (Bad Hand)": f"{loss_rate_bad:.2%}",
                "Good Hand Count": len(good_hands),
                "Bad Hand Count": len(bad_hands)
            })

        report_df = pd.DataFrame(results).sort_values("Loss Rate (Bad Hand)")
        print(report_df.to_string(index=False))
        print(f"\n- Good Hand: 手牌强度 >= {good_hand_threshold} (例如, 两个赖子+两个对子, 或一个赖子+三条)。")
        print(f"- Bad Hand: 手牌强度 <= {bad_hand_threshold} (例如, 只有一个赖子+杂牌, 或一对+杂牌)。")
        print("- Loss Rate: 在该种手牌情况下，输掉本轮的概率。一个好的策略在逆风（Bad Hand）时输率也应较低。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Liar's Dice simulation logs.")
    parser.add_argument("--file", type=str, required=True, help="Path to the simulation log JSON file.")
    args = parser.parse_args()

    analyzer = LogAnalyzer(args.file)
    analyzer.run_full_analysis()

