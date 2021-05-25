# -*- coding: utf-8 -*-
# @FileName : Serialization.py
# @Project  : MAHJONG AI
# @Author   : WANG Jianxing
# @Time     : 2021/4/17
import numpy as np
from collections import Counter
import os, os.path


def helper(size, array):
    w_dict = {'W' + str(i + 1): i for i in range(9)}  # 万
    b_dict = {'B' + str(i + 1): i + 9 for i in range(9)}  # 饼
    t_dict = {'T' + str(i + 1): i + 18 for i in range(9)}  # 条
    f_dict = {'F' + str(i + 1): i + 27 for i in range(4)}  # 风 东南西北
    j_dict = {'J' + str(i + 1): i + 31 for i in range(3)}  # （剑牌）中发白
    total_dict = {**w_dict, **b_dict, **t_dict, **f_dict, **j_dict}

    temp = np.zeros((size, 34), int)
    if not array:  # 如果当前feature为空，则直接返回
        return temp
    else:
        count = Counter(array)
        for i in count:
            if i[0] != 'H':
                tile_index = total_dict[i]
                nums = count[i]
                for j in range(nums):
                    try:
                        temp[j][tile_index] = 1
                    except:
                        with open("bug.txt", "w") as output:
                            output.write(str(array))
    return temp


def online_serialize(all_data):
    names = locals()
    # own wind 1 + round wind 1 + steal_card 1 + own hand 4 + all discards 4*4 + open melds 4*4 = 39
    all_features = {f'res{i}': np.zeros((39, 34), int) for i in range(5)}

    for index, data in enumerate(all_data):
        # print('all_data:')
        # print(len(all_data))
        # 刚开局没有历史，全部feature用空值代替
        if data:
            names['res' + str(index)] = False
            # print(names['res' + str(index)])
            for idx, array in enumerate(data):
                if idx in [0, 1, 3]:  # for wind and own wind and steal_card: each size (1,34)
                    if names['res'+str(index)] is False:
                        names['res' + str(index)] = helper(1, array)
                    else:
                        names['res' + str(index)] = np.concatenate((names['res' + str(index)], helper(1, array)), axis=0)
                # if idx == 0:  # for wind and own wind currently
                #     names['res'+str(index)] = np.zeros((1, 34), int)
                # elif idx == 1: # for wind and own wind currently
                #     temp = np.zeros((1, 34), int)
                #     names['res'+str(index)] = np.concatenate((names['res'+str(index)], temp), axis=0)

                # 目前为止： np.zeros((3,34))
                else:  # other features are all (4,34) size
                    names['res' + str(index)] = np.concatenate((names['res' + str(index)], helper(4, array)), axis=0)
                    # temp = np.zeros((4, 34), int)
                    # if not array:  # 如果当前feature为空，则直接在结果上加上np.zeros((4,34))
                    #     names['res' + str(index)] = np.concatenate((names['res' + str(index)], temp), axis=0)
                    # else:
                    #     count = Counter(array)
                    #     for i in count:
                    #         if i[0] != 'H':
                    #             tile_index = total_dict[i]
                    #             nums = count[i]
                    #             for j in range(nums):
                    #                 temp[j][tile_index] = 1
                    #     names['res' + str(index)] = np.concatenate((names['res' + str(index)], temp), axis=0)
            all_features[f'res{index}'] = names['res' + str(index)]
        # for value in all_features.values():
        #     print(value.shape)
        # res = np.concatenate((value for value in all_features.values()), axis=0)
    res = np.vstack(all_features.values())[:, :, np.newaxis]  # (195,34,1)
    # print('all serialized features size:', res.shape)
    return res


# def offline_serialize(data):
#     data = eval(data)
#     res = []
#     for idx, array in enumerate(data):
#         if len(array) == 0: # for wind and own wind currently
#             temp = np.zeros((1, 34), int)
#             # continue
#         elif len(array) == 1: # for 1 card
#             temp = np.zeros((1, 34), int)
#             count = Counter(array)
#             for i in count:
#                 if i[0] != 'H':
#                     index = total_dict[i]
#                     nums = count[i]
#                     for j in range(nums):
#                         temp[j][index] = 1
#         else:
#             temp = np.zeros((4, 34), int) # for many cards
#             count = Counter(array)
#             for i in count:
#                 if i[0] != 'H':
#                     index = total_dict[i]
#                     nums = count[i]
#                     for j in range(nums):
#                         temp[j][index] = 1
#         # print(index)
#         if idx == 0:
#             res = temp
#         else:
#             # print(res.shape)
#             res = np.concatenate((res, temp), axis=0)
#     return res


# with open('raw_dataset.txt', 'r') as fp:
#     lines = fp.readlines()
#     for i in range(5):
#         print(lines[i])
#         print(serialize(lines[i]))
#         print('\n')
# a = "[[],[],['W1', 'W6', 'W7', 'B1', 'B1', 'B3', 'B5', 'B6', 'T1', 'T4', 'T5', 'T5', 'T8', 'W6'],[],[],['W1'],['W2'],[],[],[],[]]"
# print(serialize(a))