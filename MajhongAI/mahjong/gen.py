#coding: utf-8
import pickle
import time

possibility = []


def add_to_table(cards, level, cache_table):
    key = 0
    for i in range(10):
        key = key*10 + cards[i]

    if key in cache_table:
        return True
    cache_table.add(key)
    return False


def add_menzi(cards, level, cache_table):
    for i in possibility:
        if level == 0:
            print("index", i)
        if level > 4:
            break
        tmp = None
        if 0 <= i < 8 :
            tmp = [i, i+1, i+2]
        elif 11 <= i < 20:
            tmp = [i%10] * 3
        add = True
        for card in tmp:
            cards[card] += 1
            if add and cards[card] > 4:
                add = False
        if add:
            added = add_to_table(cards, level, cache_table)
            if not added:
                if level < 4:
                    add_menzi(cards, level+1, cache_table)
        for card in tmp:
            cards[card] -= 1

def genTableWithEye():
    cache_table = set([])
    cards = [0]*10
    begin = time.time()
    print("generate start")
    for i in range(1,10):
        cards[i] = 2
        print("eye", i)
        add_to_table(cards, 0, cache_table)
        add_menzi(cards, 0, cache_table)
        cards[i] = 0
    print(len(cache_table))
    with open('mahjong/data/gen_eye_table.pickle', 'wb') as handle:
        pickle.dump(cache_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("generate end, use", time.time()-begin, "S")


def genTableWithoutEye():
    cache_table = set([])
    cards = [0]*10
    begin = time.time()
    print("generate start")
    add_to_table(cards, 0, cache_table)
    add_menzi(cards, 0, cache_table)
    print(len(cache_table))
    with open('mahjong/data/gen_table.pickle', 'wb') as handle:
        pickle.dump(cache_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("generate end, use", time.time()-begin, "S")
    # for key in cache_table:
    #     print(key)


def main():

    # -- 1-7小的顺子
    for i in range(1, 8):
        possibility.append(i)
    for i in range(11, 20):
        possibility.append(i)
    genTableWithEye()
    genTableWithoutEye()


if __name__ == '__main__':
    main()
