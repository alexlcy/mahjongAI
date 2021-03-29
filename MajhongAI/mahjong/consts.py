# coding: utf-8
from enum import Enum

# Ground类型
class MELD(Enum):
    PENG = 100 # 碰
    GANG = 200 # 暗杠(直杠)
    BU = 300 # 补杠(边杠)
    ZHI = 400 # 直杠(点杠)
    HU = 500
    COLOR = 600

    def __str__(self):
        return self.name

# 指令 -> 输出
class COMMAND(Enum):
    PASS = -1
    PLAY = 0
    PENG = 100
    GANG = 200
    BU = 300
    ZHI = 400
    HU = 500
    COLOR = 600 # 600:萬 601:筒 602:条

class CHINESE_MELD(Enum):
    PENG = "碰"
    GANG = "杠"
    BU = "补"
    ZHI = "直"
    HU = "胡"

# 事件名
class EVENT(Enum):
    PLAY = 0  # 1-9:一萬-九萬 11-19:一筒-九筒 21-29:一条-九条
    PENG = 100 # 碰
    GANG = 200 # 暗杠(直杠)
    BU = 300 # 补杠(边杠)
    ZHI = 400 # 直杠(点杠)
    HU = 500 # 胡
    COLOR = 600 # 定缺
    INIT = 700 # 初始化
    DRAW = 800 # 摸牌
    SHOW = 900 # 展示牌(用于抢杠胡判定)
    DROP = 1000 # 弃牌
    TAX  = 1100 # 退税
    LOSE = 1200 # 查大叫


COLOR = {
    -1:"待定",
    0: "'W'",
    10: "'B'",
    20: "'T'"
}

CARD = ["",
        "W1", "W2", "W3",
        "W4", "W5", "W6",
        "W7", "W8", "W9",
        "",
        "B1", "B2", "B3",
        "B4", "B5", "B6",
        "B7", "B8", "B9",
        "",
        "T1", "T2", "T3",
        "T4", "T5", "T6",
        "T7", "T8", "T9"]

CARD_DICT = {value:index for index, value in enumerate(CARD)}


CHINESE_SPECIAL = {
    "tian":"天胡",
    "qiang":"抢杠胡",
    "zi":"自摸",
    "di":"地胡",
    "dui": "对对胡",
    "yao": "带幺九",
    "qing": "清一色",
    "dan": "单吊",
    "jiang": "将对",
    "gang": "杠上开花",
    "qi": "七对",
    "bet": "根"
}
