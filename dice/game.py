import random
import json


class Game:
    def __init__(self, players, num_dice_per_player=5, initial_scores=10):
        self.players = players
        self.num_players = len(players)
        self.num_dice_per_player = num_dice_per_player
        self.initial_scores = initial_scores
        self.scores = {p.player_id: initial_scores for p in self.players}
        self.log = []

    def _log_event(self, event_type, data):
        self.log.append({'type': event_type, **data})

    def play_game(self, num_rounds):
        for round_num in range(1, num_rounds+1):
            # 每轮开始时随机化玩家顺序
            random.shuffle(self.players)
            self._play_round(round_num)
        return self.scores

    def _play_round(self, round_num):
        # 1. 掷骰子
        all_hands = {p.player_id: [random.randint(1, 6) for _ in range(self.num_dice_per_player)] for p in self.players}
        for p in self.players:
            p.set_hand(all_hands[p.player_id])

        self._log_event('round_start', {'round': round_num, 'scores': self.scores.copy(),
                                        'player_order': [p.player_id for p in self.players], 'hands': all_hands})

        current_bid = None
        history = []
        active = True
        turn = 0

        while active:
            player_index = turn % self.num_players
            current_player = self.players[player_index]

            # 2. 玩家决策
            decision = current_player.decide(current_bid, history)

            self._log_event('decision', {'player': current_player.player_id, 'hand': current_player.hand,
                                         'current_bid': current_bid, 'decision': decision})

            action, quantity, face = decision

            if action == 'challenge':
                # 3. 处理开牌
                challenger = current_player
                previous_player_index = (player_index-1+self.num_players) % self.num_players
                challenged_player = self.players[previous_player_index]

                self._resolve_challenge(challenger, challenged_player, current_bid, all_hands)
                active = False
            else:  # action == 'bid'
                # 4. 验证并更新叫号
                if current_bid and (
                        quantity < current_bid[0] or (quantity == current_bid[0] and face <= current_bid[1])):
                    # 非法叫号，判负
                    self.scores[current_player.player_id] -= 1
                    self._log_event('round_end', {'reason': 'invalid_bid', 'loser': current_player.player_id,
                                                  'details': f"Bid {decision} is invalid against {current_bid}"})
                    active = False
                else:
                    current_bid = (quantity, face)
                    history.append(current_bid)
                    turn += 1

    def _resolve_challenge(self, challenger, challenged_player, bid, all_hands):
        bid_quantity, bid_face = bid

        # 统计场上所有骰子
        total_count = 0
        all_dice_flat = [dice for hand in all_hands.values() for dice in hand]
        wilds = all_dice_flat.count(1)
        face_count = all_dice_flat.count(bid_face)
        total_count = wilds+face_count

        if total_count >= bid_quantity:
            # 开牌者输
            loser = challenger
            winner = challenged_player
            result = 'challenger_lost'
        else:
            # 被开者输
            loser = challenged_player
            winner = challenger
            result = 'challenged_lost'

        self.scores[loser.player_id] -= 1
        self._log_event('round_end', {
            'reason': 'challenge_resolved',
            'loser': loser.player_id,
            'winner': winner.player_id,
            'bid': bid,
            'actual_count': total_count,
            'result': result
        })
