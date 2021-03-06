import random
# from mahjong.consts import *
from mahjong.snapshot import Snapshot
from mahjong.settings import FeatureTracer


class RandomAgent:
    def __init__(self, player_id: int, seed=None):
        self.name = 'random'
        self.__player_id = player_id
        self.seed = seed
        random.seed(self.seed)

    def decide(self, snapshot: Snapshot, feature_tracer: FeatureTracer, trace:list, deck:list):
        player = snapshot.players[self.__player_id]
        if not player['legal_actions'] or len(player['legal_actions']) == 0:
            return
        if len(player['legal_actions']) == 1:
            player['choice'] = player['legal_actions'][0]
        else:
            player['choice'] = random.choice(player['legal_actions'])