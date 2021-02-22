import math
import logging
from mahjong.consts import CHINESE_MELD, CARD, MELD, CHINESE_SPECIAL, EVENT

#data transfer object

class Action:
    """
    行为对象
    """

    def __init__(self, player_id: int, card: int, event: EVENT, reward=0, desc=""):
        self.event = event
        self.player_id = player_id
        self.card = card
        self.reward = reward
        self.desc = desc

    def __str__(self):
        # if self.event.name.rjust(5) != ' PLAY':
        return f'{self.player_id}\t{self.event.name.rjust(5)}\t[{CARD[self.card]}]'
        # else:
        #     return 'Next'

    def __repr__(self):
        return self.__str__()

    def dump(self):
        return {
            "event": self.event.value,
            "player_id": self.player_id,
            "card": self.card,
            "reward": self.reward,
            "desc": self.desc,
        }


class Ground:
    def __init__(self, meld: MELD, card: int, src: int, special: dict = None):
        self.card = card
        self.src = src
        self.special = special
        self.meld = meld

    def __str__(self):
        return f'{self.meld}:{CARD[self.card]}:{self.src}'

    def __repr__(self):
        return self.__str__()

    def dump(self) -> dict:
        return {
            "card": self.card,
            "src": self.src,
            "special": self.special,
            "meld": self.meld.value,
        }

