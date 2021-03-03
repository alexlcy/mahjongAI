import json
import logging

from mahjong.consts import COLOR, CARD, MELD
import mahjong.settings


def translate(cards: list) -> str:
    """
    转换输入
    Args:
        cards (list): 牌列表

    Returns:
        str: 字符串
    """
    result = []
    for card in cards:
        result.append(CARD[card])
    return f'[{",".join(result)}]'

def translate_grounds(grounds:list) -> str:
    result = []
    for ground in grounds:
        result.append(f'{MELD(ground["meld"])}:{CARD[ground["card"]]}:{ground["src"]}')
    return f'[{",".join(result)}]'

class Snapshot:
    def __init__(self, data=None):
        self.step_deck = 0
        self.step_trace = 0
        self.players = None
        self.player_id = 0
        self.is_finish = False
        if data:
            self.step_deck = data["step_deck"]
            self.step_trace = data["step_trace"]
            self.players = data["players"]
            self.player_id = data["player_id"]
            self.is_finish = data["is_finish"]


    def load(self, step_deck: int, step_trace: int, players: list, player_id: int):
        self.step_deck = step_deck
        self.step_trace = step_trace
        self.players = [player.dump() for player in players]
        self.player_id = player_id
        self.is_finish = self.__is_finish()

    def dump(self) -> dict:
        return {
            'step_deck': self.step_deck,
            'step_trace': self.step_trace,
            'players': self.players,
            'player_id' : self.player_id,
            'is_finish': self.is_finish
        }

    def dumps(self) -> str:
        return json.dumps(self.dump())

    def __is_finish(self):
        cnt = 0
        for player in self.players:
            if player['is_finish']:
                cnt += 1
        return cnt >= 3

    def print(self):
        for player in self.players:
            logging.info(f'''
玩家 {player["player_id"]} \t缺色 {COLOR[player["color"]]}\t得分 {player["score"]} 完成 {player["is_finish"]}
\t手牌 {translate(player["hands"])}
\t弃子 {translate(player["drop"])}
\t放子 {translate_grounds(player["grounds"])}
\t行为 {player["legal_actions"]}''')
        logging.info(f'牌堆 {self.step_deck}\t跟踪 {self.step_trace}')

    # def save(self):
    #     for player in self.players:
    #         mahjong.settings.myList.append("%d\t%s\n" % (player["player_id"], f'{translate(player["hands"])}'))

    def print_decides(self):
        for player in self.players:
            if player["choice"]:
                logging.info(f'玩家 {player["player_id"]}\t决策 {player["choice"]}')
                # if player["choice"] == 500:
                #     mahjong.settings.myList.append("%d\t决策%s\n" % (player["player_id"], f'{player["choice"]}'))