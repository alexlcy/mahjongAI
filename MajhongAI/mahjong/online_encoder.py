# -*- coding: utf-8 -*-
# @FileName : online_encoder.py
# @Project  : MAHJONG AI
# @Author   : WANG Jianxing
# @Time     : 2021/2/17
import copy
import numpy as np
# import DataInformation
from collections import Counter
import os, os.path
from collections import deque
from mahjong import settings


def process_line(line_content):
    line_content = line_content.replace('\n', '')
    return line_content.split('\t')


def process_str_list(str_list_content):
    return str_list_content.replace('"', '').replace('\'', '').replace('[', '').replace(']', '').split(',')


def online_encoder(DL_player):
    # global Mylist

    lines = settings.myList

    # initial dicts
    # tiles_from_others = {str(i): [] for i in range(4)}
    discard = {i: [] for i in range(4)}
    # shown_tiles = {str(i): [] for i in range(4)}
    # unshown_tiles = {str(i): [] for i in range(4)}
    tiles = {i: [] for i in range(4)}
    open_meld = {i: [] for i in range(4)}
    # last_discard = {str(i): [] for i in range(4)}
    own_wind = {i: [] for i in range(4)}
    round_wind = {i: [] for i in range(4)}
    q_dict = {i: deque([[], [], [], [], []], maxlen=5) for i in range(4)}
    # q_dict = {i: deque(maxlen=5) for i in range(4)}
    q0 = deque(maxlen=5)

    for i in range(4):
        # 初始化起始手牌
        tiles[i] = process_str_list(process_line(lines[i])[1])
        
    for index in range(5, len(lines)):

        line = process_line(lines[index])
        # print(line)
        meld = process_str_list(line[2])
        # print(meld)
        card = meld[0]

        # print(card)
        player = int(line[0])
        features = copy.deepcopy([own_wind[player],
                                 round_wind[player],
                                 tiles[player],
                                 discard[player],
                                 discard[player + 1 if player + 1 <= 3 else (player + 1) % 4],
                                 discard[player + 2 if player + 2 <= 3 else (player + 2) % 4],
                                 discard[player + 3 if player + 3 <= 3 else (player + 3) % 4],
                                 open_meld[player],
                                 open_meld[player + 1 if player + 1 <= 3 else (player + 1) % 4],
                                 open_meld[player + 2 if player + 2 <= 3 else (player + 2) % 4],
                                 open_meld[player + 3 if player + 3 <= 3 else (player + 3) % 4]
                                ])
        q_dict[player].append(features)

        if line[1] == 'PLAY':
            tiles[player].remove(card)
            # unshown_tiles[player].remove(card)
            # last_discard[player] = [card]
            discard[player].append(card)
        elif line[1] == 'DRAW':
            tiles[player].append(card)
            # unshown_tiles[player].append(card)
        elif line[1] == 'PENG':
            open_meld[player].extend(meld)
            # tiles[player].append(card)
            meld.remove(card)
            # print(temp)
            for i in meld:
                tiles[player].remove(i)
        elif line[1] == 'BU':
            tiles[player].remove(card)
            # unshown_tiles[player].remove(card)
            for value in open_meld.values():
                if card in value:
                    value.append(card)
        elif line[1] == 'ZHI':
            open_meld[player].extend(meld)
            meld.remove(card)
            for i in meld:
                tiles[player].remove(i)
            # tiles[player].append(card)
        # elif line[1] == 'GANG':
        #     tiles[player].remove(card)
            # open_meld[player].append(meld)
        elif line[1] == 'HU':
            open_meld[player].append(card)

    # for info in q_dict[DL_player]:
    #     print(info)
    return q_dict[DL_player]


# print(online_encoder(0))

