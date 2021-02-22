# coding: utf-8
import pickle
import time
import copy
import math
import json
# from mahjong.utils import *
from mahjong.consts import MELD, EVENT


class Judger:
    """
    胡牌检测
    """

    def __init__(self):
        self.__tbl = {}
        self.__tbl_eye = {}
        self.__loadTable()

    # 加载胡牌牌型表
    def __loadTable(self):
        begin = time.time()
        # print("load start")
        with open('mahjong/data/gen_table.pickle', 'rb') as handle:
            self.__tbl = pickle.load(handle)
        with open('mahjong/data/gen_eye_table.pickle', 'rb') as handle:
            self.__tbl_eye = pickle.load(handle)
        # print("load end, use", time.time()-begin, "S")

    def check_hu(self, hands: list, grounds: list) -> bool:
        cards = [0] * 30
        count = [0] * 3

        colors = set([])  # 色
        pair = 0  # 对子数
        eyekey = 0  # 带眼码
        key = 0  # 不带眼码
        tun = 0

        # ================预处理开始================
        for card in hands:
            cards[card] += 1
            color = math.floor(card/10)
            count[color] += 1
            colors.add(color)
        all_cards = cards[:]
        for ground in grounds:
            all_cards[ground.card] += 3
            color = math.floor(ground.card/10)
            count[color] += 1
            colors.add(color)
            if ground.meld == MELD.BU or ground.meld == MELD.ZHI or ground.meld == MELD.GANG:
                all_cards[ground.card] += 1
        # ================预处理结束================

        for cnt in cards:
            if cnt == 0 or cnt == 1:
                continue
            if cnt == 2:
                pair += 1
            elif cnt == 3:
                tun += 1
            elif pair == 4:
                pair += 2
        # 特殊胡法
        if (tun + len(grounds) == 4 and pair == 1) or pair == 7:
            return True

        for player_id, cnt in enumerate(count):
            if cnt <= 0:
                continue
            tmp = cards[player_id * 10:player_id * 10+10]
            if cnt % 3 == 2:
                for card in tmp:
                    eyekey = eyekey * 10 + card
            else:
                for card in tmp:
                    key = key * 10 + card
        return eyekey in self.__tbl_eye and key in self.__tbl  # 有胡

    # 胡牌检查
    def make_hu(self, hands: list, grounds: list) -> dict:
        cards = [0] * 30
        count = [0] * 3

        special = {}  # 胡法
        colors = set([])  # 色
        tun = 0  # 屯数
        pair = 0  # 对子数
        bet = 0  # 根数
        jiang = True  # 将
        yao = False  # 带幺九

        # ================预处理开始================
        for card in hands:
            cards[card] += 1
            color = math.floor(card/10)
            count[color] += 1
            colors.add(color)

            base = card % 10
            if jiang and base != 2 and base != 5 and base != 8:
                jiang = False
            if not yao and (base == 1 or base == 9):
                yao = True
        all_cards = cards[:]
        for ground in grounds:
            all_cards[ground.card] += 3
            color = math.floor(ground.card/10)
            count[color] += 1
            colors.add(color)

            base = ground.card % 10
            if jiang and base != 2 and base != 5 and base != 8:
                jiang = False
            if not yao and (base == 1 or base == 9):
                yao = True
            if ground.meld == MELD.BU or ground.meld == MELD.ZHI or ground.meld == MELD.GANG:
                all_cards[ground.card] += 1
        # ================预处理结束================

        for cnt in all_cards:
            if cnt == 4:
                bet += 1
        for cnt in cards:
            if cnt == 0 or cnt == 1:
                continue
            if cnt == 2:
                pair += 1
            elif cnt == 3:
                tun += 1
            elif pair == 4:
                pair += 2
        # 特殊胡法
        if tun + len(grounds) == 4 and pair == 1:
            special['dui'] = 1  # 对对胡
            if tun == 0:
                special['dan'] = 1  # 单吊
            if yao:
                special['yao'] = 1  # 幺九 d
            if jiang:
                special['jiang'] = 1  # 将
            if len(colors) == 1:
                special['qing'] = 2
            if bet > 0:
                special['bet'] = bet
        elif pair == 7:
            special['qi'] = 1  # 七对
            if yao:
                special['yao'] = 1  # 幺九
            if len(colors) == 1:
                special['qing'] = 2
            if bet > 0:
                special['bet'] = bet
        return special


judger = Judger()
