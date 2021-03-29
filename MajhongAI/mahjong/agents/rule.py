import numpy as np
import random
import math
from mahjong.snapshot import Snapshot
from mahjong.consts import COMMAND

from mahjong.settings import FeatureTracer


class RuleAgent(object):
    def __init__(self, player_id: int):
        self.name = 'rule'
        self.__player_id = player_id
        self.state_tensor = np.zeros((9, 8, 10))

    def decide(self, snapshot: Snapshot, feature_tracer: FeatureTracer, trace:list, deck:list):
        player = snapshot.players[self.__player_id]
        legal_actions = player['legal_actions']
        if not legal_actions or len(legal_actions) == 0:
            return
        if len(legal_actions) == 1:
            player['choice'] = legal_actions[0]
            return
        # 选缺
        colors = [0] * 3
        for card in player['hands']:
            colors[math.floor(card/10)] += 1
        if legal_actions[0] >= 600:
            player['choice'] = COMMAND.COLOR.value + colors.index(min(colors))
            return
        player['choice'] = max(legal_actions)
        if player['choice'] < 100:
            cards = [0] * 30
            for card in player['hands']:
                cards[card] += 1
            for card in range(30):
                if cards[card] == 1:
                    player['choice'] = card
                    return
            for card in range(30):
                if cards[card] == 2:
                    player['choice'] = card
                    return
            player['choice'] = random.choice(legal_actions)
        # player['choice'] = random.choice(player['legal_actions'])
