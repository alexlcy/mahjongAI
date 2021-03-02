# def init():
#     global myList
#     myList = []

class FeatureTracer():
	def __init__(self):
		self.discard = {i: [] for i in range(4)}
		self.steal = {i: [] for i in range(4)}
	    self.tiles = {i: [] for i in range(4)}
	    self.open_meld = {i: [] for i in range(4)}
	    self.own_wind = {i: [] for i in range(4)}
	    self.round_wind = {i: [] for i in range(4)}
	    self.q_dict = {i: deque([[], [], [], [], []], maxlen=5) for i in range(4)}

	def update(self, line):
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
                self.tiles[player].remove(i)
            # tiles[player].append(card)
        # elif line[1] == 'GANG':
        #     tiles[player].remove(card)
            # open_meld[player].append(meld)
        elif line[1] == 'HU':
            self.open_meld[player].append(card)

	def get_features(self, player_id):
		return copy.deepcopy([own_wind[player],
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