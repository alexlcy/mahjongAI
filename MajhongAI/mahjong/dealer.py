import random
from copy import deepcopy
from mahjong.player import Player


class Dealer:
    def __init__(self, rand_seed: float):
        # 填充牌堆
        random.seed(rand_seed)
        self.deck = []
        for x in range(3):
            for y in range(9):
                for _ in range(4):
                    self.deck.append(x * 10 + y + 1)
        self.shuffle()
        self.step = 0
        self.banker_id = None

    def shuffle(self):
        random.shuffle(self.deck)

    def deal_cards(self, player: Player, num: int = 1):
        player.fill(self.deck[self.step:self.step + num])
        self.step += num
    
    def next_card(self):
        self.step += 1
        if self.step >= len(self.deck):
            return None
        return self.deck[self.step]

    def get_banker(self):
        # 随机庄
        self.banker_id = 0  # random.randint(0, 2)
        return self.banker_id

    def jump(self, step: int):
        self.step = step

    def get_step(self) -> int:
        return self.step
