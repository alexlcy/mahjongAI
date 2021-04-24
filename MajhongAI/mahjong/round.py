import math
import logging
from enum import Enum
from mahjong.dealer import Dealer
from mahjong.dto import Action, Ground
from mahjong.judger import judger
from mahjong.player import Player
from mahjong.snapshot import Snapshot
from mahjong.consts import MELD, EVENT, COMMAND, CHINESE_SPECIAL, CARD
from mahjong.settings import FeatureTracer

import numpy as np
from copy import deepcopy

# 退税/和局/过碰
from mahjong.ReinforcementLearning.experience import ExperienceCollector
from mahjong.consts import COLOR


def parse_command(choice: int) -> COMMAND:
    if not choice or choice < 0:
        return COMMAND.PASS, 0
    return COMMAND(choice - choice % 100), choice % 100


class Round:
    """
    处理核心
    """

    def __init__(self, dealer: Dealer, players: list, config: dict, is_rl_agents):
        self.dealer = dealer
        self.players = players
        self.player_id = dealer.get_banker()
        self.config = config
        self.is_rl_agents = is_rl_agents
        self.trace = []
        self.player_num = len(players)
        self.feature_tracer = None
        self.temp = None

        # Experience Buffer
        # print(self.is_rl_agents)
        self.collectors = {0: ExperienceCollector(0, self.is_rl_agents[0]),
                           1: ExperienceCollector(1, self.is_rl_agents[1]),
                           2: ExperienceCollector(2, self.is_rl_agents[2]),
                           3: ExperienceCollector(3, self.is_rl_agents[3])}
        self.rewards = {0: None, 1: None, 2: None, 3: None}
        self.norm_rewards = {0: 0, 1: 0, 2: 0, 3: 0}
        self.action_num = 0

    def get_snapshot(self) -> Snapshot:
        """
        获取快照

        Returns:
            Snapshot: 快照副本
        """
        snapshot = Snapshot()
        snapshot.load(self.dealer.get_step(), len(self.trace), self.players, self.player_id)
        return snapshot

    def start(self):
        """
        游戏开始
        """
        raw_player_initial_hands = {player.player_id: player.hands for player in self.players}
        player_initial_hands = {i: [] for i in range(4)}
        for key, values in raw_player_initial_hands.items():
            for value in values:
                player_initial_hands[key].append(CARD[value])
        self.feature_tracer = FeatureTracer(player_initial_hands)

        card = self.dealer.next_card()
        self.current_player.get(card)
        self.trace.append(Action(self.player_id, card, EVENT.INIT))  # 初始化
        for player in self.players:
            player.process_legal_actions(card, self.trace)

        # raw_player_initial_hands = {player.player_id: player.hands for player in self.players}
        # player_initial_hands = {i: [] for i in range(4)}
        # for key, values in raw_player_initial_hands.items():
        #     for value in values:
        #         player_initial_hands[key].append(CARD[value])
        # self.feature_tracer = FeatureTracer(player_initial_hands)

    def next(self, snapshot: Snapshot) -> Snapshot:
        """
        下一步

        Args:
            snapshot (Snapshot): 快照(决策)

        Raises:
            Exception: 未知异常

        Returns:
            Snapshot: 快照副本
        """
        # self.action_num += 1

        previou_action = self.trace[-1]
        # logging.info(f'previou action {previou_action}')
        if previou_action.event == EVENT.INIT:
            for player_id, player in enumerate(snapshot.players):
                self.players[player_id].set_color(player['choice'])
                self.players[player_id].clear_legal_actions()
            self.__record(Action(self.player_id, previou_action.card, EVENT.DRAW))
            self.current_player.process_legal_actions(previou_action.card, self.trace)
            return self.get_snapshot()

        if previou_action.event == EVENT.HU:
            # 一炮双响 继续出牌
            if self.trace[-2].event != EVENT.HU:
                self.player_id = self.next_player_id
            return self.__draw()
        if previou_action.event == EVENT.GANG or previou_action.event == EVENT.BU or previou_action.event == EVENT.ZHI:
            return self.__draw()

        if previou_action.event == EVENT.DRAW or previou_action.event == EVENT.PENG:  # 摸牌后 或 碰之后
            choice = snapshot.players[previou_action.player_id]['choice']
            command, card = parse_command(choice)
            if command == COMMAND.PLAY:  # 打牌
                self.__record(Action(self.player_id, card, EVENT.PLAY))
                self.current_player.remove(card)
                for player in self.players:
                    player.process_legal_actions(card, self.trace)
            elif command == COMMAND.GANG:  # 暗杠
                # 算分
                reward = 0
                for player in self.players:
                    if not player.is_finish and player.player_id != self.player_id:
                        player.score -= 2
                        # An Kong
                        # if self.rewards[player.player_id] is None:
                        #     self.rewards[player.player_id] = -2
                        #     self.norm_rewards[player.player_id] = -1
                        # else:
                        #     self.rewards[player.player_id] -= 2
                        #     self.norm_rewards[player.player_id] -= 1
                        reward += 2
                self.current_player.score += reward
                # AN Kong
                # if self.rewards[self.player_id] is None:
                #     self.rewards[self.player_id] = reward
                #     self.norm_rewards[self.player_id] = int(reward / 2)
                # else:
                #     self.rewards[self.player_id] += reward
                #     self.norm_rewards[self.player_id] += int(reward / 2)
                self.__record(Action(self.player_id, card, EVENT.GANG, reward))
                self.current_player.make_gang(card)
            elif command == COMMAND.BU:  # 补杠
                # 这里算分特殊
                self.__record(Action(self.player_id, card, EVENT.SHOW))
                self.current_player.remove(card)
            elif command == COMMAND.HU:  # 胡
                special = judger.make_hu(self.current_player.get_hands(), self.current_player.get_grounds())
                special_event = self.trace[-2].event
                if special_event == EVENT.GANG or special_event == EVENT.ZHI or special_event == EVENT.BU:
                    special['gang'] = 1
                elif special_event == EVENT.INIT:  # 天胡
                    special['tian'] = 2
                elif self.dealer.get_banker() != self.player_id:
                    di = True
                    for action in self.trace:
                        if action.player_id == self.player_id and action.event != EVENT.DRAW:
                            di = False
                            break
                    if di:
                        special['di'] = 2  # 地胡
                desc = []
                bet = 0
                special['zi'] = 1  # 自摸加番
                for sp in special:
                    bet += special[sp]
                    desc.append(f'{CHINESE_SPECIAL[sp]}:{special[sp]}')
                bet = min(bet, 4)
                score = math.floor(math.pow(2, bet))
                reward = 0
                for player in self.players:
                    if not player.is_finish and player.player_id != self.player_id:
                        player.score -= score
                        if self.rewards[player.player_id] is None:
                            self.rewards[player.player_id] = -score
                        else:
                            self.rewards[player.player_id] -= score
                        self.norm_rewards[player.player_id] -= 1
                        reward += score
                self.current_player.score += reward
                if self.rewards[self.player_id] is None:
                    self.rewards[self.player_id] = reward
                else:
                    self.rewards[self.player_id] += reward
                self.norm_rewards[self.player_id] += int(reward / score)
                action = Action(self.player_id, previou_action.card, EVENT.HU, reward, ",".join(desc))
                self.__record(action)
                self.current_player.is_finish = True
                self.current_player.make_hu(action)
                self.current_player.clear_legal_actions()
            else:
                raise Exception("WRONG LOGIC")

            return self.get_snapshot()

        if previou_action.event == EVENT.PLAY:  # 出牌后
            hu_action = []
            for player_id, player in enumerate(snapshot.players):
                choice = snapshot.players[player_id]['choice']
                if not choice:
                    continue
                command, card = parse_command(choice)
                if command == COMMAND.HU:  # 放炮
                    win_player = self.players[player_id]
                    hands = win_player.get_hands()
                    hands.append(previou_action.card)
                    special = judger.make_hu(hands, win_player.get_grounds())
                    di = True
                    for action in self.trace:
                        if action.player_id == self.player_id and action.event != EVENT.DRAW:
                            di = False
                            break
                    if di:
                        special['di'] = 2  # 地胡
                    desc = []
                    bet = 0
                    for sp in special:
                        bet += special[sp]
                        desc.append(f'{CHINESE_SPECIAL[sp]}:{special[sp]}')
                    bet = min(bet, 4)
                    score = math.floor(math.pow(2, bet))
                    reward = score
                    if len(self.trace) > 3:  # 呼叫转移
                        action = self.trace[-3]
                        if action.event == EVENT.ZHI or action.event == EVENT.GANG or action.event == EVENT.BU:
                            reward += action.reward
                            desc.append("呼叫转移")
                    win_player.score += reward
                    self.current_player.score -= reward
                    if self.rewards[self.player_id] is None:
                        self.rewards[self.player_id] = -reward
                    else:
                        self.rewards[self.player_id] -= reward
                    self.norm_rewards[self.player_id] -= 1
                    if self.rewards[player_id] is None:
                        self.rewards[player_id] = reward
                    else:
                        self.rewards[player_id] += reward
                    self.norm_rewards[player_id] += 1
                    tmp_action = Action(player_id, previou_action.card, EVENT.HU, reward, ",".join(desc))
                    hu_action.append(tmp_action)
                    win_player.make_hu(tmp_action)
                    win_player.is_finish = True
                    win_player.clear_legal_actions()
            if len(hu_action) > 0:
                for action in hu_action:
                    self.__record(action)
                return self.get_snapshot()
            for player_id, player in enumerate(snapshot.players):
                choice = snapshot.players[player_id]['choice']
                command, card = parse_command(choice)
                if command == COMMAND.PASS:
                    continue
                if command == COMMAND.PENG:  # 碰
                    self.player_id = player_id
                    self.current_player.make_peng(card, previou_action.player_id)
                    self.__record(Action(self.player_id, card, EVENT.PENG))
                    self.current_player.process_legal_actions(-1, self.trace)
                elif command == COMMAND.ZHI:  # 直杠
                    self.player_id = player_id
                    # 算分
                    reward = 0
                    for player in self.players:
                        if not player.is_finish and player.player_id != player_id:
                            player.score -= 1
                            # ZHI
                            # if self.rewards[player.player_id] is None:
                            #     self.rewards[player.player_id] = -1
                            #     self.norm_rewards[player.player_id] = -1
                            # else:
                            #     self.rewards[player.player_id] -= 1
                            #     self.norm_rewards[player.player_id] -= 1
                            reward += 1
                        if not player.is_finish and player.player_id == previou_action.player_id:
                            player.score -= 1
                            # ZHI
                            # if self.rewards[player.player_id] is None:
                            #     self.rewards[player.player_id] = -1
                            #     self.norm_rewards[player.player_id] = -1
                            # else:
                            #     self.rewards[player.player_id] -= 1
                            #     self.norm_rewards[player.player_id] -= 1
                            reward += 1
                    self.current_player.score += reward
                    # ZHI
                    # if self.rewards[self.current_player.player_id] is None:
                    #     self.rewards[self.current_player.player_id] = reward
                    #     self.norm_rewards[self.current_player.player_id] = reward
                    # else:
                    #     self.rewards[self.current_player.player_id] += reward
                    #     self.norm_rewards[self.current_player.player_id] += reward
                    self.__record(Action(self.player_id, card, EVENT.ZHI, reward))
                    self.current_player.make_zhi(card, previou_action.player_id)
                else:
                    raise Exception("WRONG LOGIC")
                return self.get_snapshot()
            # 没人要
            self.__record(Action(self.player_id, previou_action.card, EVENT.DROP))
            self.current_player.drop.append(previou_action.card)
            for player in self.players:
                player.clear_legal_actions()
            self.player_id = self.next_player_id
            return self.__draw()

        if previou_action.event == EVENT.SHOW:  # 补杠
            hu_action = []
            for player_id, player in enumerate(snapshot.players):
                choice = snapshot.players[player_id]['choice']
                command, card = parse_command(choice)
                if command == COMMAND.PASS:
                    continue
                if command == COMMAND.HU:  # 放炮
                    win_player = self.players[player_id]
                    hands = win_player.get_hands()
                    hands.append(previou_action.card)
                    special = judger.make_hu(hands, win_player.get_grounds())
                    di = True
                    for action in self.trace:
                        if action.player_id == self.player_id and action.event != EVENT.DRAW:
                            di = False
                            break
                    if di:
                        special['di'] = 2  # 地胡
                    if previou_action.event == EVENT.SHOW:  # 抢杠胡
                        special['qiang'] = 1
                    desc = []
                    bet = 0
                    for sp in special:
                        bet += special[sp]
                        desc.append(f'{CHINESE_SPECIAL[sp]}:{special[sp]}')
                    bet = min(bet, 4)
                    reward = math.floor(math.pow(2, bet))
                    win_player.score += reward
                    self.current_player.score -= reward
                    self.rewards[player_id] = reward if self.rewards[player_id] is None else self.rewards[player_id] + reward
                    self.norm_rewards[player_id] += 1
                    self.rewards[self.current_player.player_id] = -reward if self.rewards[self.current_player.player_id] is None else self.rewards[self.current_player.player_id] - reward
                    self.norm_rewards[self.current_player.player_id] -= 1
                    tmp_action = Action(self.player_id, previou_action.card, EVENT.HU, reward, ",".join(desc))
                    hu_action.append(tmp_action)
                    win_player.make_hu(tmp_action)
                    win_player.is_finish = True
                    win_player.clear_legal_actions()
            if len(hu_action) > 0:
                for action in hu_action:
                    self.__record(action)
                return self.get_snapshot()
            # 无人抢杠胡
            hu = False
            for player in self.players:
                if player.process_legal_actions(previou_action.card, self.trace):
                    hu = True
            if hu:
                return self.get_snapshot()

            reward = 0
            for player in self.players:
                if not player.is_finish and player.player_id != self.player_id:
                    player.score -= 1
                    reward += 1
                    # BU
                    # self.rewards[player.player_id] = -1 if self.rewards[player.player_id] is None else self.rewards[player.player_id] - 1
                    # self.norm_rewards[player.player_id] = -1 if self.rewards[player.player_id] is None else self.norm_rewards[player.player_id] - 1
            self.current_player.score += reward
            # BU
            # self.rewards[self.player_id] = reward if self.rewards[self.player_id] is None else self.rewards[self.player_id] + reward
            # self.norm_rewards[self.player_id] = reward if self.rewards[self.player_id] is None else self.norm_rewards[self.player_id] + reward
            self.__record(Action(self.player_id, previou_action.card, EVENT.BU, reward))
            self.current_player.make_bu(previou_action.card)
            return self.__draw()
        raise Exception("WRONG LOGIC")

    def __record(self, action: Action):
        """
        记录器

        Args:
            action (Action): 行为
        """

        self.trace.append(action)
        raw_state = deepcopy(self.feature_tracer.tiles)
        if str(action).replace('\t', ' ').replace(':', ' ').split()[3] in ['PLAY', 'DRAW', 'PENG', 'BU', 'ZHI', 'GANG',
                                                                           'HU']:
            self.feature_tracer.update(action)
        # print(action)
        self.action_num += 1
        # if action.event.name == 'HU':

        current_hu_rewards = {}
        for i in self.rewards.keys():
            if self.rewards[i] is not None:
                current_hu_rewards[i] = self.rewards[i]
            else:
                current_hu_rewards[i] = 0

        # # TODO: checking, can delete
        # if action.event.name == 'HU':
        #     tmp = 0
        #     for i in range(4):
        #         tmp += current_hu_rewards[i]
        #     print(f'HU rewards: {tmp} !!')

        # if action.event.name == 'HU':
        #     print('Checking ~~')

        self.collectors[action.player_id].record_decision(self.action_num, raw_state[action.player_id],
                                                          self.feature_tracer.tiles[action.player_id],
                                                          self.feature_tracer.discard[action.player_id],
                                                          self.feature_tracer.open_meld[action.player_id],
                                                          self.feature_tracer.steal,
                                                          (action.event.name, CARD[action.card], action.player_id, action.player_id), action.reward,
                                                          self.current_player.score,
                                                          COLOR[self.players[action.player_id].color],
                                                          self.feature_tracer, current_hu_rewards[action.player_id],
                                                          self.norm_rewards[action.player_id])
        for player_id in self.rewards.keys():
            if player_id != action.player_id and self.rewards[player_id] is not None:
                self.collectors[player_id].record_decision(self.action_num, raw_state[player_id],
                                                           self.feature_tracer.tiles[player_id],
                                                           self.feature_tracer.discard[player_id],
                                                           self.feature_tracer.open_meld[player_id],
                                                           self.feature_tracer.steal,
                                                           (action.event.name, CARD[action.card], action.player_id, player_id),
                                                           self.rewards[player_id],
                                                           self.players[player_id].score,
                                                           COLOR[self.players[player_id].color],
                                                           self.feature_tracer, current_hu_rewards[player_id],
                                                           self.norm_rewards[player_id])

        # # TODO: checking, can delete
        # if action.event.name == 'HU':
        #     hu_reward_checking = {}
        #     tmp = 0
        #     for i in range(4):
        #         hu_reward_checking[i] = self.collectors[i].hu_rewards
        #         tmp += hu_reward_checking[i]
        #     print(f'HU rewards: {tmp} !! And {hu_reward_checking}')

        self.rewards = {0: None, 1: None, 2: None, 3: None}
        # self.temp = self.feature_tracer.get_features(0) # (190,34,1)
        if not self.config["show_log"]:
            return
        logging.info(action)

    def __game_over(self):
        """
        游戏结束 退税/查叫
        """
        for player in self.players:
            if player.is_finish:
                continue
            # 检查听牌
            cards = player.get_possible_cards()
            for card in cards:
                if not player.test_hu(card):
                    continue
                hands = player.get_hands()
                hands.append(card)
                special = judger.make_hu(hands, player.get_grounds())
                bet = -1
                desc = []
                for sp in special:
                    bet += special[sp]
                    desc.append(f'{CHINESE_SPECIAL[sp]}:{special[sp]}')
                bet = min(bet, 4)
                if bet > player.max_bet:
                    player.max_special = special
                    player.max_desc = ",".join(desc)
                    player.max_bet = bet
        win_players = []
        lose_players = []
        for player in self.players:
            if not player.is_finish:
                if player.max_bet < 0:
                    # # TODO 暂时扣分 不回填 # 退税
                    # for action in self.trace:
                    #     if action.player_id == player.player_id and (action.event == EVENT.BU or action.event == EVENT.GANG or action.event == EVENT.ZHI):
                    #         player.score -= action.reward
                    #         self.__record(Action(player.player_id, 0, EVENT.TAX, -action.reward))
                    lose_players.append(player)
                else:
                    win_players.append(player)
        for winner in win_players:
            for loser in lose_players:
                reward = math.floor(math.pow(2, winner.max_bet))
                winner.score += reward
                loser.score -= reward
                self.__record(Action(winner.player_id, 0, EVENT.LOSE, reward, winner.max_desc))
        snapshot = self.get_snapshot()
        snapshot.is_finish = True
        return snapshot

    def __draw(self) -> Snapshot:
        """
        摸牌

        Returns:
            Snapshot: 快照副本
        """
        card = self.dealer.next_card()
        if not card:
            return self.__game_over()
        self.current_player.get(card)
        # 摸牌行为
        self.__record(Action(self.player_id, card, EVENT.DRAW))
        # 计算合法行为
        self.current_player.process_legal_actions(card, self.trace)
        return self.get_snapshot()

    def __get_current_player(self) -> Player:
        return self.players[self.player_id]

    def __get_next_player_id(self) -> Player:
        for player_id in range(self.player_id + 1, self.player_id + self.player_num):
            if self.players[player_id % self.player_num].is_finish:
                continue
            return player_id % self.player_num
        return None

    current_player: Player = property(__get_current_player)
    next_player_id: int = property(__get_next_player_id)
