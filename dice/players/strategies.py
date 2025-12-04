from .base_player import BasePlayer
from utils import binomial_prob_gte


class PlayerStrategy1_HonestExpectedValue(BasePlayer):
    """思路一：诚实期望值算法"""

    def make_decision(self, current_bid, history):
        total_dice = self.num_players * self.num_dice_per_player
        unknown_dice = total_dice-len(self.hand)

        # 计算所有点数的总期望值
        expected_values = {}
        for face in range(2, 7):
            my_count = self._get_my_count(face)
            # 每个未知骰子是该点数或赖子的概率是 2/6 = 1/3
            expected_others = unknown_dice * (1 / 3)
            expected_values[face] = my_count+expected_others

        if current_bid is None:
            # 第一个叫，叫自己期望最高的
            best_face = max(expected_values, key=expected_values.get)
            return ('bid', int(expected_values[best_face]), best_face)

        bid_quantity, bid_face = current_bid

        # 决策：如果叫的数量大于我的期望，就开
        if bid_quantity > expected_values[bid_face]:
            return ('challenge', None, None)
        else:
            # 叫号：找到一个合法的、期望最高的叫号
            best_face = max(expected_values, key=expected_values.get)
            if best_face > bid_face:
                return ('bid', bid_quantity, best_face)
            else:
                return ('bid', bid_quantity+1, best_face)


class PlayerStrategy2_SafetyThreshold(BasePlayer):
    """思路二：安全阈值算法"""

    def __init__(self, *args, **kwargs):
        # **FIX:** Pop the custom argument before calling the parent's init
        self.safety_threshold = kwargs.pop('safety_threshold', 0.5)
        super().__init__(*args, **kwargs)

    def _get_prob(self, quantity, face):
        my_count = self._get_my_count(face)
        needed_from_others = quantity-my_count

        total_dice = self.num_players * self.num_dice_per_player
        unknown_dice = total_dice-len(self.hand)

        return binomial_prob_gte(needed_from_others, unknown_dice, 1 / 3)

    def make_decision(self, current_bid, history):
        if current_bid is None:
            # 第一个叫，叫一个对自己来说概率最高的号
            best_prob = -1
            best_bid = None
            for q in range(len(self.hand), self.num_players * self.num_dice_per_player // 2):
                for f in range(2, 7):
                    prob = self._get_prob(q, f)
                    if prob > best_prob:
                        best_prob = prob
                        best_bid = ('bid', q, f)
            return best_bid or ('bid', self._get_my_count(6)+1, 6)

        bid_quantity, bid_face = current_bid

        # 决策：计算当前叫号为真的概率
        prob_true = self._get_prob(bid_quantity, bid_face)
        if prob_true < self.safety_threshold:
            return ('challenge', None, None)
        else:
            # 叫号：找到下一个概率最高的合法叫号
            # 简单策略：仅增加数量或点数
            prob_if_inc_face = 0
            if bid_face < 6:
                prob_if_inc_face = self._get_prob(bid_quantity, bid_face+1)

            prob_if_inc_quantity = self._get_prob(bid_quantity+1, 2)

            if prob_if_inc_face > prob_if_inc_quantity and prob_if_inc_face > self.safety_threshold:
                return ('bid', bid_quantity, bid_face+1)
            else:
                # 默认增加数量，从最小的点数2开始叫
                return ('bid', bid_quantity+1, 2)


class PlayerStrategy3_BayesianInference(BasePlayer):
    """思路三：基于叫号历史的贝叶斯推理算法 (简化版)"""

    def __init__(self, *args, **kwargs):
        # **FIX:** Pop the custom argument before calling the parent's init
        self.belief_update_factor = kwargs.pop('belief_update_factor', 1.5)
        super().__init__(*args, **kwargs)

    def _get_prob_with_belief(self, quantity, face, history):
        my_count = self._get_my_count(face)

        total_dice = self.num_players * self.num_dice_per_player
        unknown_dice = total_dice-len(self.hand)

        # 简化贝叶斯：根据历史叫号调整期望值
        # 统计历史中每个点数被叫的次数
        face_calls = {f: 0 for f in range(2, 7)}
        for h_quant, h_face in history:
            face_calls[h_face] += 1

        # 假设每次叫号，都意味着场上该点数比平均多 belief_update_factor 个
        belief_adjustment = face_calls.get(face, 0) * self.belief_update_factor

        # 调整后的期望需要的数量
        adjusted_needed = quantity-my_count-belief_adjustment

        return binomial_prob_gte(int(adjusted_needed), unknown_dice, 1 / 3)

    def make_decision(self, current_bid, history):
        if current_bid is None:
            # 简化处理：同策略2
            return ('bid', len(self.hand), max(self.hand) if max(self.hand) > 1 else 6)

        bid_quantity, bid_face = current_bid

        prob_true = self._get_prob_with_belief(bid_quantity, bid_face, history)

        if prob_true < 0.5:  # 固定阈值
            return ('challenge', None, None)
        else:
            # 简单叫号策略
            if bid_face < 6:
                return ('bid', bid_quantity, bid_face+1)
            return ('bid', bid_quantity+1, 2)


class PlayerStrategy4_MinimumLoss(BasePlayer):
    """思路四：最小损失决策算法 (简化版)"""

    def _get_prob(self, quantity, face):
        my_count = self._get_my_count(face)
        needed = quantity-my_count
        total_dice = self.num_players * self.num_dice_per_player
        unknown_dice = total_dice-len(self.hand)
        return binomial_prob_gte(needed, unknown_dice, 1 / 3)

    def make_decision(self, current_bid, history):
        if current_bid is None:
            return ('bid', len(self.hand), max(self.hand) if max(self.hand) > 1 else 6)

        bid_quantity, bid_face = current_bid

        # 1. 计算“开”的期望损失
        prob_true = self._get_prob(bid_quantity, bid_face)
        loss_challenge = -prob_true  # 输的概率 * -1

        # 2. 计算“叫号”的期望损失 (简化)
        # 寻找最安全的下一个叫号
        next_bids = []
        if bid_face < 6:
            next_bids.append((bid_quantity, bid_face+1))
        for f in range(2, 7):
            next_bids.append((bid_quantity+1, f))

        safest_bid_prob = 0
        best_next_bid = ('bid', bid_quantity+1, 2)  # Default bid
        for b_q, b_f in next_bids:
            prob = self._get_prob(b_q, b_f)
            if prob > safest_bid_prob:
                safest_bid_prob = prob
                best_next_bid = ('bid', b_q, b_f)

        # 假设叫号的损失 = (被下一家开并输的概率) * -1
        # 简化假设：下一家开你的概率是 (1 - safest_bid_prob)
        # 你输的概率是 safest_bid_prob
        loss_bid = -(1-safest_bid_prob) * safest_bid_prob

        if loss_challenge > loss_bid:
            return ('challenge', None, None)
        else:
            return best_next_bid


class PlayerStrategy5_Hybrid(BasePlayer):
    """思路五：综合策略与安全边界算法"""
    GLOBAL_DIST = {13: 0.1, 14: 0.05, 15: 0.025}  # 简化的全局概率分布安全边界

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 内部实例化低级策略以便调用
        self.strategy2 = PlayerStrategy2_SafetyThreshold(*args, **kwargs)
        self.strategy3 = PlayerStrategy3_BayesianInference(*args, **kwargs)

    def set_hand(self, hand):
        super().set_hand(hand)
        self.strategy2.set_hand(hand)
        self.strategy3.set_hand(hand)

    def make_decision(self, current_bid, history):
        if current_bid is None:
            return ('bid', len(self.hand), max(self.hand) if max(self.hand) > 1 else 6)

        bid_quantity, bid_face = current_bid

        # 晚期 (进入危险区)
        if bid_quantity > 13:
            # 严重偏向于“开”，除非自己是天牌
            my_max_support = max(self._get_my_count(f) for f in range(2, 7))
            if my_max_support < 4:  # 如果自己支持不强
                # 使用策略2计算概率，但用一个更激进的阈值
                prob = self.strategy2._get_prob(bid_quantity, bid_face)
                if prob < 0.65:  # 只要有超过35%的胜算就开
                    return ('challenge', None, None)

        # 中期 (博弈关键区)
        if 9 <= bid_quantity <= 13:
            # 使用带信念更新的策略3
            return self.strategy3.make_decision(current_bid, history)

        # 早期 (安全区)
        else:  # bid_quantity < 9
            # 使用基础的安全阈值策略2
            return self.strategy2.make_decision(current_bid, history)


