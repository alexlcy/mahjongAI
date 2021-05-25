# -*- coding: utf-8 -*-
# @FileName : raw_data_processor.py
# @Project  : MAHJONG AI
# @Author   : WANG Jianxing
# @Time     : 2021/4/17
import copy
import numpy as np
# import DataInformation
from collections import Counter
import os, os.path
from collections import deque


# for i in range(10000):
#   execfile("filename.py")


def process_line(line_content):
    line_content = line_content.replace('\n', '')
    return line_content.split('\t')


def process_str_list(str_list_content):
    return str_list_content.replace('"', '').replace('\'', '').replace('[', '').replace(']', '').split(',')


def raw_data_processing(folder_path, times=None):
    # folder_path: 存储对局数据的文件夹位置
    # times: 要处理多少局的数据，默认为None则处理全部对局
    # return: txt格式的全部data

    if times:
        assert times <= len([name for name in os.listdir(folder_path)])
    else:
        times = len([name for name in os.listdir(folder_path)])
    print(f'Total times to deal with: {times}')
    raw_dataset_name = 'raw_dataset.txt'
    # file_path = '../mahjong5/datasets'
    # if not os.path.exists(file_path):
    #     os.mkdir(file_path)
    with open(raw_dataset_name, 'a') as write_file:
        # for i in mahjong.settings.myList:
        #     file.write(f'%s' % str(i))

        files = os.listdir(folder_path)  # 得到对局文件夹下的所有文件名称
        for time in range(times):  # 遍历times组对局
            with open(folder_path + "/" + files[time]) as read_file:
                lines = read_file.readlines()
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
                # history = {i: [[] for j in range(5)] for i in range(4)}
                q_dict = {i: deque(maxlen=5) for i in range(4)}
                # q0 = deque(maxlen=5)
                for i in range(4):
                    tiles[i] = process_str_list(process_line(lines[i])[1])
                    # unshown_tiles[str(i)] = process_str_list(process_line(lines[i])[1])

                #     flowers = [int(process_line(lines[2])[2]),
                #                int(process_line(lines[3])[2]),
                #                int(process_line(lines[4])[2]),
                #                int(process_line(lines[5])[2])]

                # rows_content = []
                for index in range(5, len(lines)):
                # for index in range(5, 7):
                    #         row_content = [-999] * len(DataInformation.header)
                    # index=4
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
                    if len(q_dict[player]) == 5:
                        for history_ in q_dict[player]:
                            write_file.write(str(history_)+'\n')
                        write_file.write('\n')
                        # write_file.write('[' + str(own_wind[player]) + ',')
                        # write_file.write(str(round_wind[player])+',')
                        # write_file.write(str(tiles[player])+',')
                        # write_file.write(str(discard[player])+',')
                        # write_file.write(str(discard[player + 1 if player + 1 <= 3 else (player + 1) % 4])+',')
                        # write_file.write(str(discard[player + 2 if player + 2 <= 3 else (player + 2) % 4])+',')
                        # write_file.write(str(discard[player + 3 if player + 3 <= 3 else (player + 3) % 4])+',')
                        # write_file.write(str(open_meld[player])+',')
                        # write_file.write(str(open_meld[player + 1 if player + 1 <= 3 else (player + 1) % 4])+',')
                        # write_file.write(str(open_meld[player + 2 if player + 2 <= 3 else (player + 2) % 4])+',')
                        # write_file.write(str(open_meld[player + 3 if player + 3 <= 3 else (player + 3) % 4])+']'+'\n')

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
                    # else:
                    #     return
                    # if line[0] == '0':
                    #     print(tiles['0'])
            print(f'{time+1}st is ok')
            # write_file.write('\n')
    # return tiles[str(who_are_you)], open_meld, discard, last_discard


# D:\DDM_spring\6980new\mahjong5_new\mahjong5\datasets
folder_path = "E:/DDM_spring/6980new/mahjong5_new/mahjong5/datasets"  # 文件夹目录
folder_path2 = '../../MajhongAI/datasets'
raw_data_processing(folder_path2)



