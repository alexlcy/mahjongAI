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
        self.name = 'deeplearning'
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

        # Step 1: 选缺 process (Only trigger once in a round)
        colors = [0] * 3
        for card in player['hands']:
            colors[math.floor(card / 10)] += 1
        if legal_actions[0] >= 600:
            player['choice'] = COMMAND.COLOR.value + colors.index(min(colors))
            return

        # Step 2: Choose which Action
        player['choice'] = max(legal_actions)
        pos = -1

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
        
        # Step 3: Choose which one to discard
        if player['choice'] < 100:
            discard_tile = self.decide_discard(player)
            # Call discard function to discard a tile
            if discard_tile is not None:
                player['choice'] = discard_tile
                return      
                      
        player['choice'] = random.choice(legal_actions)
          
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
        raise NotImplementedError

    def decide_discard(self, player):
        """
        The tile is discarded based on the below sequence
        1. Discard color tile if there exist
        2. Discard based on model suggestion
        3. Discard based on naive rules

        Returns:
            discard_tile: The tile want to discard
        """

        # Priority 1: Discard based on color
        color_discard_tile = self.decide_discard_by_color()
        if color_discard_tile is not None:
            return color_discard_tile

        # Priority 2: Discard based on AI model
        ai_discard_tile = self.decide_discard_by_AI()
        if ai_discard_tile in player['hands']:
            return ai_discard_tile

        # Priority 3: Discard based on naive rule
        return self.decide_discard_by_rule

    def decide_discard_by_AI(self):
        """
        Call the discard model and return the tile that we shoudl discard
        Returns:
        """
        raise NotImplementedError

    def decide_discard_by_color(self, player):
        """
        If there exist a tile that belongs to the color (缺), the player should
        discard it before it discard other tile.

        Args:
            player ():

        Returns:

        """
        # Discard based on color (缺)
        color = player['color']

        # Discard if the there are only one 缺 tile
        for tile_num in range(1, 10):
            if player['hands'][color + tile_num] == 1:
                return color + tile_num

        # Discard if the there are two 缺 tile
        for tile_num in range(1, 10):
            if player['hands'][color + tile_num] == 2:
                return color + tile_num

        # Discard if the there are only three tile
        for tile_num in range(1, 10):
            if player['hands'][color + tile_num] == 2:
                return color + tile_num

        # If no 缺 tile available, then skip the 缺 discard function
        return None


    def decide_discard_by_rule(self,player):
        """
        When there are no valid decision made by AI and no discard based on color.
        The discard will be conducted using the same naive rule as the rule agent.

        Args:
            player ():

        Returns:

        """
        # Discard based on rule
        cards = [0] * 30
        for card in player['hands']:
            cards[card] += 1

        for card in range(30):
            if cards[card] == 1:
                return card
        for card in range(30):
            if cards[card] == 2:
                return card

