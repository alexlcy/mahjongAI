import numpy as np
import random
import math
from mahjong.snapshot import Snapshot
from mahjong.consts import COMMAND

# To do for everyone
# TODO: Improve the discard process of the rule agent, it is stupid now

# To do for alex
# TODO: Select discard tile

# To do for Koning
# TODO: Select action from legal action


class DeepLearningAgent(object):
    def __init__(self, player_id: int):
        self.name = 'rule'
        self.__player_id = player_id
        self.state_tensor = np.zeros((9, 8, 10))

    def decide(self, snapshot: Snapshot, trace:list, deck:list):
        """
        Decide which action to take based on the legal action and data available

        Args:
            snapshot ():
            trace ():
            deck ():

        Returns:
        """

        player = snapshot.players[self.__player_id]
        legal_actions = player['legal_actions']

        # Exception handling
        if not legal_actions or len(legal_actions) == 0:
            return
        if len(legal_actions) == 1:
            player['choice'] = legal_actions[0]
            return

        # 选缺 process
        colors = [0] * 3
        for card in player['hands']:
            colors[math.floor(card/10)] += 1
        if legal_actions[0] >= 600:
            player['choice'] = COMMAND.COLOR.value + colors.index(min(colors))
            return

        # Choose which Action
        player['choice'] = max(legal_actions)

        # Choose which one to discard
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


    def decide_kong(self):
        # Decide whether to make a Kong
        whether_kong = True
        score = 1

        return whether_kong, score

    def decide_pong(self):
        # Decide whether to make a Pong
        whether_pong = True
        score = 1

        return whether_pong, score

    def decide_win(self):
        # Later will implementa decide win model
        raise NotImplementedError

    def decide_discard(self):
        """
        If there are no 缺 in the tile, then we will call the decide_tile_discard

        Returns:
        """
        raise NotImplementedError

    def decide_tile_discard(self):
        """


        Returns:
        """
        raise NotImplementedError

        raise NotImplementedError

