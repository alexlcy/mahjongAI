import math
import logging
from mahjong.consts import CHINESE_MELD, CARD, MELD, CHINESE_SPECIAL, EVENT
import mahjong.settings

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
        # if self.event.name != 'DROP' and self.event.name != 'SHOW':
        #     if self.event.name == 'BU' or self.event.name == 'ZHI' or self.event.name == 'GANG':
        #         mahjong.settings.myList.append("%d\t%s\t[%s,%s,%s,%s]\t\n" % (self.player_id,self.event.name,CARD[self.card],CARD[self.card],CARD[self.card],CARD[self.card]))
        #     elif self.event.name == 'PENG':
        #         mahjong.settings.myList.append("%d\t%s\t[%s,%s,%s]\n" % (self.player_id, self.event.name, CARD[self.card], CARD[self.card], CARD[self.card]))
        #     else:
        #         mahjong.settings.myList.append("%d\t%s\t[%s]\n" % (self.player_id, self.event.name, CARD[self.card]))
        return f'玩家 {self.player_id}\t行为 {self.event.name.rjust(5)}\t{CARD[self.card]}\t奖励:{self.reward}\t{self.desc}'

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
