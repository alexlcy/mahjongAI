# coding: utf-8
import random
import time
import pickle
import logging
import os
from copy import deepcopy

from mahjong.player import Player
from mahjong.dealer import Dealer
from mahjong.round import Round
from mahjong.snapshot import Snapshot

class Game():
    def __init__(self):
        self.history = []
        self.rand_seed = None
        self.player_num = None
        self.dealer = None
        self.round = None
        self.player_id = 0

    def init_game(self, config: dict, is_rl_agents):
        self.config = config
        self.is_rl_agents = is_rl_agents
        self.rand_seed = config['seed']
        self.player_num = config['player_num']
        self.uuid = time.strftime('%Y%m%d%H%M%S') + str(random.randint(1000,9999))

    def new_game(self):
        self.history = []
        players = [Player(i) for i in range(self.player_num)]
        self.dealer = Dealer(self.rand_seed)
        for player in players:
            self.dealer.deal_cards(player, 13)
        self.round = Round(self.dealer, players, self.config, self.is_rl_agents)
        self.player_id = self.dealer.get_banker()
        self.round.start()
        snapshot = self.round.get_snapshot()
        self.history.append(snapshot)
        return snapshot

    def load_game(self, uuid, step) -> Snapshot:
        with open(f'logs/{uuid}_history.pickle', 'rb') as handle:
            self.history = pickle.load(handle)
        self.history = self.history[:step]
        self.dealer = Dealer(self.rand_seed)
        players = []
        snapshot = self.history[-1]

        for player_id, player_data in enumerate(snapshot.players):
            player = Player(player_id)
            player.load(player_data)
            players.append(player)
        
        self.round = Round(self.dealer, players, self.config)

        with open(f'logs/{uuid}_trace.pickle', 'rb') as handle:
            self.round.trace = pickle.load(handle)[:snapshot.step_trace]

        self.round.player_id = snapshot.player_id
        self.dealer.jump(snapshot.step_deck)
        return snapshot

    def step_back(self, step:int) -> Snapshot:
        self.history = self.history[:-step]
        players = []
        snapshot = self.history[-1]
        snapshot.print()
        for player_id, player_data in enumerate(snapshot.players):
            player = Player(player_id)
            player.load(player_data)
            players.append(player)
        self.round.players = players
        self.round.trace = self.round.trace[:snapshot.step_trace]
        self.round.player_id = snapshot.player_id
        self.dealer.jump(snapshot.step_deck)
        return snapshot


    def save(self):
        if not os.path.exists("logs"):
            os.mkdir("logs")
        with open(f'logs/{self.uuid}_history.pickle', 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'logs/{self.uuid}_trace.pickle', 'wb') as handle:
            pickle.dump(self.round.trace, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'logs/{self.uuid}_seed.pickle', 'wb') as handle:
            pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def next(self, snapshot) -> Snapshot:
        snapshot = self.round.next(snapshot)
        self.history.append(snapshot)
        return snapshot

    def print_trace(self):
        for action in self.round.trace:
            if not self.config["show_log"]:
                return
            logging.info(action)

    # def restart_game(self, )

    def replay(self):
        """
        Replay the whole game, set the state to the very beginning
        """
        # if not self.initial_storage:
        #     return False
        # self.round = self.initial_storage
        # self.players = self.__players
        # self.dealer = self.__round.dealer

        return True