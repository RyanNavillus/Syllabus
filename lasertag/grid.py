import numpy as np


class Grid(object):
    def __init__(self, n_agents, width, height, dtype="U5"):
        self.n_agents = n_agents
        self.width = width
        self.height = height
        self.dtype = dtype
        self.level = np.full((width, height), ".", dtype=dtype)

    def put(self, indicies, values):
        np.put(self.level, indicies, values)

    def set_level(self, level):
        self.level = level
        self.height = level.shape[0]
        self.width = level.shape[1]

    def get_n_clutter(self):
        return (self.level == "w").sum()

    def get_first_symbol(self, symbol):
        try:
            return (
                np.where(self.level == symbol)[1][0],
                np.where(self.level == symbol)[0][0],
            )
        except IndexError:
            return None

    def get_agent_locs(self):
        locs = []
        for a in range(self.n_agents):
            symbol = "a" + str(a + 1)
            loc = self.get_first_symbol(symbol)
            locs.append(loc)

        return locs

    # for returning a level represntation
    def encode(self, metrics):
        return (self.level.tobytes("C"), self.width, self.height, metrics)

    def decode(self, level_encoding):
        level_bytes = level_encoding[0]
        self.width = level_encoding[1]
        self.height = level_encoding[2]

        self.level = np.frombuffer(level_bytes, dtype=self.dtype).reshape(
            (self.width, self.height)
        )

        return level_encoding[3]  # metrics
