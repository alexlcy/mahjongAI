import numpy as np
from collections import Counter
import os, os.path


w_dict = {'W' + str(i + 1): i for i in range(9)}  # 万
b_dict = {'B' + str(i + 1): i + 9 for i in range(9)}  # 饼
t_dict = {'T' + str(i + 1): i + 18 for i in range(9)}  # 条
f_dict = {'F' + str(i + 1): i + 27 for i in range(4)}  # 风 东南西北
j_dict = {'J' + str(i + 1): i + 31 for i in range(3)}  # （剑牌）中发白
total_dict = {**w_dict, **b_dict, **t_dict, **f_dict, **j_dict}


def serialize(data):
    data = eval(data)
    res = []
    for idx, array in enumerate(data):
        if len(array) == 0: # for wind and own wind currently
            temp = np.zeros((1, 34), int)
            # continue
        elif len(array) == 1: # for 1 card
            temp = np.zeros((1, 34), int)
            count = Counter(array)
            for i in count:
                if i[0] != 'H':
                    index = total_dict[i]
                    nums = count[i]
                    for j in range(nums):
                        temp[j][index] = 1
        else:
            temp = np.zeros((4, 34), int) # for many cards
            count = Counter(array)
            for i in count:
                if i[0] != 'H':
                    index = total_dict[i]
                    nums = count[i]
                    for j in range(nums):
                        temp[j][index] = 1
        # print(index)
        if idx == 0:
            res = temp
        else:
            # print(res.shape)
            res = np.concatenate((res, temp), axis=0)
    return res


# with open('raw_dataset.txt', 'r') as fp:
#     lines = fp.readlines()
#     for i in range(5):
#         print(lines[i])
#         print(serialize(lines[i]))
#         print('\n')
# a = "[[],[],['W1', 'W6', 'W7', 'B1', 'B1', 'B3', 'B5', 'B6', 'T1', 'T4', 'T5', 'T5', 'T8', 'W6'],[],[],['W1'],['W2'],[],[],[],[]]"
# print(serialize(a))