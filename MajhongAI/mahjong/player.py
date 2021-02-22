# coding: utf-8
import random
import logging
import math
from copy import deepcopy
from mahjong.consts import MELD, COLOR, CARD, EVENT
from mahjong.judger import judger
from mahjong.dto import Ground, Action


class Player:
    """
    玩家对象
    """

    def __init__(self, player_id: int):
        """
        初始化
        Args:
            player_id (int): 玩家索引
        """
        self.player_id = player_id
        # 手牌区
        self.hands = []
        # 落子区
        self.grounds = []
        # 弃子区
        self.drop = []
        # 过碰
        self.pass_peng = set([])
        # legal quest actions
        self.legal_actions = []

        self.is_finish = False
        # 分数
        self.score = 0
        # 天胡
        self.is_native = True
        # 缺
        self.color = -1

        self.step = 0

        self.max_special = None
        self.max_desc = None
        self.max_bet = -1

    def fill(self, cards):
        """
        发牌
        Args:
            cards (int[]): 初始手牌
        """
        self.hands = cards
        self.hands.sort()

    def get(self, card: int):
        """
        摸牌
        """
        self.hands.append(card)
        self.hands.sort()

    def get_possible_cards(self) -> list:
        """
        可能计算胡牌的牌
        Returns:
            list: 列表
        """
        cards = [0] * 30
        possible = []
        for card in self.hands:
            cards[card] += 1
        for ground in self.grounds:
            cards[ground.card] += 3
            if ground.meld == MELD.BU or ground.meld == MELD.ZHI or ground.meld == MELD.GANG:
                cards[ground.card] += 1
        for card, count in enumerate(cards):
            if card % 10 != 0 and count < 4 and not (self.color < card < self.color + 10):
                possible.append(card)
        return possible

    # 打牌
    def remove(self, card):
        self.hands.remove(card)

    # 听牌检测
    def test_hu(self, card: int) -> bool:
        cards = self.hands[:]
        cards.append(card)
        return judger.check_hu(cards, self.grounds)

    def process_legal_actions(self, card: int, trace: list):
        if self.is_finish:
            return False
        self.legal_actions = []
        previou_action = trace[-1]
        if previou_action.event == EVENT.INIT:
            self.legal_actions = [MELD.COLOR.value + i for i in range(3)]
        elif previou_action.event == EVENT.DRAW or previou_action.event == EVENT.PENG:
            self.__check_hands(card, previou_action)
        elif previou_action.event == EVENT.PLAY:
            self.__check_card(card, previou_action)
            if len(self.legal_actions) > 0:
                self.legal_actions.insert(0, -1)
        elif previou_action.event == EVENT.SHOW:
            self.__check_hu(card, previou_action)
            if len(self.legal_actions) > 0:
                self.legal_actions.insert(0, -1)
        self.legal_actions = list(set(self.legal_actions))
        self.legal_actions.sort()
        return len(self.legal_actions) > 0

    def clear_legal_actions(self):
        """
        清理行为
        """
        self.legal_actions = []

    def get_hands(self) -> list:
        return self.hands[:]

    def get_grounds(self) -> list:
        return self.grounds[:]

    def make_peng(self, card: int, src: int):
        """
        碰牌

        Args:
            card (int): 牌
            src (int): 来源
        """
        self.grounds.append(Ground(MELD.PENG, card, src))
        self.remove(card)
        self.remove(card)

    def make_bu(self, card: int):
        """
        补杠

        Args:
            card (int): 牌
        """
        ground = self.__find_ground(card, MELD.PENG)
        ground.meld = MELD.BU

    def make_zhi(self, card: int, src: int):
        """
        放杠

        Args:
            card (int): 牌
            src (int): 来源
        """
        self.grounds.append(Ground(MELD.ZHI, card, src))
        self.remove(card)
        self.remove(card)
        self.remove(card)

    def make_gang(self, card: int):
        """
        暗杠

        Args:
            card (int): 牌
        """
        self.grounds.append(Ground(MELD.GANG, card, self.player_id))
        self.remove(card)
        self.remove(card)
        self.remove(card)
        self.remove(card)

    def make_hu(self, action:Action):
        self.grounds.append(Ground(MELD.HU, action.card, action.player_id))

    def set_color(self, color: COLOR):
        """
        定缺

        Args:
            color (COLOR): 缺色
        """
        self.color = (color - MELD.COLOR.value) * 10

    def dump(self):
        grounds = []
        for ground in self.grounds:
            grounds.append(ground.dump())
        return deepcopy({
            'player_id': self.player_id,
            'hands': self.hands,
            'drop': self.drop,
            'score': self.score,
            'choice': None,
            'grounds': grounds,
            'legal_actions': self.legal_actions,
            'color': self.color,
            'is_finish': self.is_finish
        })

    def load(self, data):
        self.hands = data["hands"]
        self.drop = data["drop"]
        self.score = data["score"]
        self.grounds = []
        for ground in data["grounds"]:
            self.grounds.append(Ground(MELD(ground["meld"]), ground["card"], ground["src"], ground["special"]))
        self.legal_actions = data["legal_actions"]
        self.color = data["color"]
        self.is_finish = data["is_finish"]

    # 放子区查找
    def __find_ground(self, card: int, meld: MELD) -> Ground:
        return next((ground for ground in self.grounds if ground.card ==
                     card and ground.meld == meld), None)

    # 计算指定牌数量
    def __count_card(self, card: int) -> int:
        count = 0
        for tmp_card in self.hands:
            if card == tmp_card:
                count += 1
        return count

    # 碰检测
    def __check_peng(self, card: int):
        if self.color < card < self.color + 10:
            return
        if card not in self.pass_peng and self.__count_card(card) >= 2:
            self.legal_actions.append(MELD.PENG.value + card)

    # 放杠检测
    def __check_gang_zhi(self, card: int):
        if self.color < card < self.color + 10:
            return
        if self.__count_card(card) == 3:
            self.legal_actions.append(MELD.ZHI.value + card)

    # 补杠检测
    def __check_gang_bu(self, card: int):
        if self.color < card < self.color + 10:
            return
        if self.__find_ground(card, MELD.PENG):
            self.legal_actions.append(MELD.BU.value + card)

    # 暗杠检测
    def __check_gang_an(self):
        cards = {}
        for card in self.hands:
            if self.color < card < self.color + 10:
                continue
            if not cards.__contains__(card):
                cards[card] = 0
            cards[card] += 1
        for card in cards:
            if cards[card] == 4:
                self.legal_actions.append(MELD.GANG.value + card)

    # 胡牌检测
    def __check_hu(self, card: int, action: Action):
        cards = self.hands[:]
        if action.event != EVENT.DRAW:
            cards.append(card)
        for card in cards:
            if self.color < card < self.color + 10:
                return
        if judger.check_hu(cards, self.grounds):
            self.legal_actions.append(MELD.HU.value)

    def __check_hands(self, card: int, action: Action):
        self.legal_actions = []
        self.__check_gang_bu(card)
        self.__check_gang_an()
        if card > 0:
            self.__check_hu(card, action)
        self.__check_play()
        # return self.makeQuest()

    # 检查当前牌
    def __check_card(self, card: int, action: Action):
        if action.player_id == self.player_id:
            return None
        self.__check_gang_zhi(card)
        self.__check_peng(card)
        self.__check_hu(card, action)

    def __check_play(self):
        cards = []
        for card in self.hands:
            if self.color < card < self.color + 10:
                cards.append(card)
        if len(cards) == 0:
            cards = self.hands[:]
        self.legal_actions += cards
