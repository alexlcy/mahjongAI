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
        self.name = 'DL'
        self.__player_id = player_id
        self.state_tensor = np.zeros((9, 8, 10))

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
        # Later will implement decide win model
        return True

    def decide_discard(self, card):
        """
        If there are no 缺 in the tile, then we will call the decide_tile_discard

        Returns:
        """
        return card

    def decide_tile_discard(self):
        """


        Returns:
        """
        return

    # TODO: Debug & optimize some repeat part - Koning
    def decide(self, snapshot: Snapshot, trace: list, deck: list):
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
            colors[math.floor(card / 10)] += 1
        if legal_actions[0] >= 600:
            player['choice'] = COMMAND.COLOR.value + colors.index(min(colors))
            return

        # Choose which Action
        player['choice'] = max(legal_actions)
        pos = -1

        # Choose which one to discard
        if player['choice'] < 100:
            player['choice'] = self.decide_discard(player['choice'])
            return

        # Choose whether win
        if player['choice'] >= 500:
            whether_win = self.decide_win()
            if whether_win is True:
                return
            else:
                pos -= 1
                player['choice'] = legal_actions[pos]
                # TODO: If no bug, will remove below print
                print(f"1 Checking: new choice - {player['choice']} should be >=100 & < 500~~")

        # Choose whether zhi kong
        if player['choice'] >= 400:
            # check whether pong
            # retrieve confidence score of pong and kong
            whether_pong, pong_score = self.decide_pong()
            whether_kong, kong_score = self.decide_kong()
            if all([whether_pong, whether_kong]):
                if pong_score > kong_score:
                    player['choice'] = legal_actions[pos - 1]
                    # TODO: If no bug, will remove below print
                    print(f"2 Checking: new choice - {player['choice']} should be >=100 & < 200~~")
                    return
                else:
                    return
            elif whether_kong is True and whether_pong is False:
                return
            elif whether_kong is False and whether_pong is True:
                player['choice'] = legal_actions[pos - 1]
                # TODO: If no bug, will remove below print
                print(f"3 Checking: new choice - {player['choice']} should be >=100 & < 200~~")
                return
            elif whether_kong is False and whether_pong is False:
                player['choice'] = legal_actions[pos - 2]
                # TODO: If no bug, will remove below print
                print(f"4 Checking: new choice - {player['choice']} should be <= 0~~")
                return

        # Choose whether bu kong
        if player['choice'] >= 300:
            # check whether kong
            whether_kong, _ = self.decide_kong()
            if whether_kong is True:
                return
            else:
                player['choice'] = legal_actions[pos - 1]
                # TODO: If no bug, will remove below print
                print(f"5 Checking: new choice - {player['choice']} should be <= -1~~")
                return

        # Choose whether an kong
        if player['choice'] >= 200:
            # check whether kong
            whether_kong, _ = self.decide_kong()
            if whether_kong is True:
                return
            else:
                player['choice'] = legal_actions[pos - 1]
                # TODO: If no bug, will remove below print
                print(f"6 Checking: new choice - {player['choice']} should be <= -1~~")
                return

        # Choose whether pong
        if player['choice'] >= 100:
            # check whether pong
            whether_pong, _ = self.decide_pong()
            if whether_pong is True:
                return
            else:
                player['choice'] = legal_actions[pos - 1]
                # TODO: If no bug, will remove below print
                print(f"7 Checking: new choice - {player['choice']} should be <= -1~~")
                return

