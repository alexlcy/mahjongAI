from mahjong.snapshot import Snapshot

class HumanAgent(object):
    def __init__(self, player_id: int):
        self.name = 'Humen'
        self.__player_id = player_id

    def decide(self, snapshot: Snapshot, trace:list, deck:list):
        player = snapshot.players[self.__player_id]
        legal_actions = player['legal_actions']
        if not legal_actions or len(legal_actions) == 0:
            return
        action = input(f'>> {legal_actions}\n>> You choose action: ')
        action = int(action)
        # print(action)
        # print(self.legal_actions)

        while action not in legal_actions:
            print('Action illegel...')
            action = input(f'>> {legal_actions}\n>> Re-choose action: ')
            action = int(action)
        player['choice'] = action


    # def step(self, state):
    #     self.game.dump()
    #     for k in state['string_state'].keys():
    #         print(f'{k}:', end='\t')
    #         print(state['string_state'][k])
    #     self.legal_actions_chinese = state['string_state']['legal_actions']
    #     self.legal_actions = state['legal_actions']
    #     # display the actions
    #     print(f'You are player {self.game.currentPlayer.player_id}')
    #     if self.game.currentPlayer.needPlay():
    #         print(f'Play the card:\t{self.legal_actions_chinese}')
    #     else:
    #         print(f'Card On Quest: {CARD[self.game.currentCard.card]}')
    #         print('Answer the quest:')
    #         for i, quest in enumerate(self.legal_actions_chinese):
    #             print(f'{i}: {quest}')

    #     action = input('>> You choose action: ')
    #     action = self.string2actions(action)
    #     # print(action)
    #     # print(self.legal_actions)

    #     while action not in self.legal_actions:
    #         print('Action illegel...')
    #         action = input('>> Re-choose action: ')
    #         action = self.string2actions(action)
    #         print(action)
    #         print(self.legal_actions)
    #     return action

    # def string2actions(self, hand_input):
    #     """
    #     play card='x1' or 'd5' or 'dT'
    #     or quest_index
    #     """
    #     str_player_id = list(range(len(self.legal_actions)))
    #     str_player_id = [str(i) for i in str_player_id]
    #     # play the card
    #     if hand_input in input_space:
    #         suit, rank = hand_input
    #         if rank == 'T':
    #             rank = 10
    #         return int(suit == 'd') * 10 + int(rank) - 1
    #     # if it's card
    #     elif hand_input in str_player_id:
    #         return self.legal_actions[int(hand_input)]
    #     else:
    #         return