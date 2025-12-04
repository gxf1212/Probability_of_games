from abc import ABC, abstractmethod
import time


class BasePlayer(ABC):
    """
    所有AI玩家策略的抽象基类。
    每个策略都必须实现 make_decision 方法。
    """

    def __init__(self, player_id, num_players, num_dice_per_player):
        self.player_id = player_id
        self.num_players = num_players
        self.num_dice_per_player = num_dice_per_player
        self.hand = []
        self.decision_times = []

    def set_hand(self, hand):
        """为新一轮设置手牌"""
        self.hand = sorted(hand)

    def _get_my_count(self, face):
        """计算自己手牌对某个点数的贡献（包括赖子）"""
        wilds = self.hand.count(1)
        return self.hand.count(face)+wilds

    def get_average_decision_time(self):
        """获取平均决策时间"""
        if not self.decision_times:
            return 0
        return sum(self.decision_times) / len(self.decision_times)

    @abstractmethod
    def make_decision(self, current_bid, history):
        """
        核心决策方法。

        Args:
            current_bid (tuple): (数量, 点数) or None for the first player.
            history (list): 本轮的出价历史。

        Returns:
            tuple: ('bid', 数量, 点数) or ('challenge', None, None).
        """
        pass

    def decide(self, current_bid, history):
        """包裹决策方法以计算时间"""
        start_time = time.perf_counter()
        decision = self.make_decision(current_bid, history)
        end_time = time.perf_counter()
        self.decision_times.append(end_time-start_time)
        return decision

    def __str__(self):
        return f"{self.__class__.__name__}_{self.player_id}"

