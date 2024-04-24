import gym.envs.registration as register
import numpy as np

from .lasertag import Lasertag


class LasertagFixed(Lasertag):
    """A short but non-optimal path is 80 moves."""

    def __init__(self, *args, n_agents, level, max_steps=200, **kwargs):
        self.level = level
        self.max_steps = max_steps
        super().__init__(*args, n_agents=n_agents, max_steps=max_steps, **kwargs)

    def reset(self):
        return super().reset(level=self.level)


arena_1 = np.array(
    [
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "."],
        [".", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "."],
        ["a1", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "a2"],
        [".", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "."],
        [".", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    ]
)

arena_2 = np.array(
    [
        [".", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "."],
        ["a1[D]", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "."],
        [".", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "a2[U]"],
        [".", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "."],
    ]
)

corridor_1 = np.array(
    [
        ["a1[D]", "w", "w", "w", "."],
        [".", "w", "w", "w", "."],
        [".", "w", "w", "w", "."],
        [".", ".", ".", ".", "."],
        [".", "w", "w", "w", "."],
        [".", "w", "w", "w", "."],
        [".", "w", "w", "w", "a2[U]"],
    ]
)

corridor_2 = np.array(
    [
        [".", ".", ".", ".", "."],
        [".", "w", "w", "w", "."],
        [".", "w", "w", "w", "."],
        ["a1", "w", "w", "w", "a2"],
        [".", "w", "w", "w", "."],
        [".", "w", "w", "w", "."],
        [".", ".", ".", ".", "."],
    ]
)

maze1 = np.array(
    [
        [".", ".", ".", ".", ".", "w", "a2", ".", ".", ".", "w", ".", "."],
        [".", "w", "w", "w", ".", "w", "w", "w", "w", ".", "w", "w", "."],
        [".", "w", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "w", "w", "w", "w", "w", "w", "w", "w", ".", "w", "w", "w"],
        [".", ".", ".", ".", ".", ".", ".", ".", "w", ".", ".", ".", "."],
        ["w", "w", "w", "w", "w", "w", ".", "w", "w", "w", "w", "w", "."],
        [".", ".", ".", ".", "w", ".", ".", "w", ".", ".", ".", ".", "."],
        [".", "w", "w", ".", ".", ".", "w", "w", ".", "w", "w", "w", "w"],
        [".", ".", "w", ".", "w", ".", ".", "w", ".", ".", ".", "w", "."],
        ["w", ".", "w", ".", "w", "w", ".", "w", "w", "w", ".", "w", "."],
        ["w", ".", "w", ".", ".", "w", ".", ".", ".", "w", ".", ".", "."],
        ["w", ".", "w", "w", ".", "w", "w", "w", ".", "w", "w", "w", "."],
        [".", ".", ".", "w", ".", ".", "a1", "w", ".", "w", ".", ".", "."],
    ]
)

maze2 = np.array(
    [
        [".", ".", ".", "w", ".", "w", ".", ".", ".", ".", "w", ".", "."],
        [".", "w", ".", "w", ".", "w", "w", "w", "w", ".", ".", ".", "w"],
        [".", "w", ".", ".", ".", ".", ".", ".", ".", ".", "w", ".", "."],
        [".", "w", "w", "w", "w", "w", "w", "w", "w", ".", "w", "w", "w"],
        [".", ".", ".", "w", ".", ".", "w", ".", "w", ".", "w", ".", "a2"],
        ["w", "w", ".", "w", ".", "w", "w", ".", "w", ".", "w", ".", "."],
        ["a1", "w", ".", "w", ".", ".", ".", ".", "w", ".", "w", "w", "."],
        [".", "w", ".", "w", "w", ".", "w", "w", "w", ".", ".", "w", "."],
        [".", "w", ".", ".", "w", ".", ".", "w", "w", "w", ".", "w", "."],
        [".", "w", "w", ".", "w", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", "w", ".", ".", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", ".", ".", "w", ".", ".", ".", "w", ".", ".", ".", ".", "."],
    ]
)

ruins = np.array(
    [
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        ["a1[R]", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "a2[L]"],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    ]
)

ruins_2 = np.array(
    [
        ["a1[D]", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "w", "w", ".", ".", "w", "w", ".", ".", "w", "w", "."],
        [".", "w", "w", ".", ".", "w", "w", ".", ".", "w", "w", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "w", "w", ".", ".", "w", "w", ".", ".", "w", "w", "."],
        [".", "w", "w", ".", ".", "w", "w", ".", ".", "w", "w", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "w", "w", ".", ".", "w", "w", ".", ".", "w", "w", "."],
        [".", "w", "w", ".", ".", "w", "w", ".", ".", "w", "w", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "a2[U]"],
    ]
)

star = np.array(
    [
        ["w", "w", ".", "w", "w", "w", ".", "w", "w", "w", ".", "w", "w"],
        ["w", ".", ".", ".", "w", ".", ".", ".", "w", ".", ".", ".", "w"],
        [".", ".", "w", ".", ".", ".", "w", ".", ".", ".", "w", ".", "a2[D]"],
        [".", "w", "w", "w", ".", "w", "w", "w", ".", "w", "w", "w", "."],
        [".", ".", "w", ".", ".", ".", "w", ".", ".", ".", "w", ".", "."],
        ["w", ".", ".", ".", "w", ".", ".", ".", "w", ".", ".", ".", "w"],
        ["w", "w", ".", "w", "w", "w", ".", "w", "w", "w", ".", "w", "w"],
        ["w", ".", ".", ".", "w", ".", ".", ".", "w", ".", ".", ".", "w"],
        [".", ".", "w", ".", ".", ".", "w", ".", ".", ".", "w", ".", "."],
        [".", "w", "w", "w", ".", "w", "w", "w", ".", "w", "w", "w", "."],
        ["a1[U]", ".", "w", ".", ".", ".", "w", ".", ".", ".", "w", ".", "."],
        ["w", ".", ".", ".", "w", ".", ".", ".", "w", ".", ".", ".", "w"],
        ["w", "w", ".", "w", "w", "w", ".", "w", "w", "w", ".", "w", "w"],
    ]
)

corridor_large = np.array(
    [
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", "w", "a1[U]", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", "w", "w", "w", "w", "w", "w", "w", "w", "w", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", "a2[D]", "w", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", "w", ".", "w", ".", "w", ".", "w", ".", "w", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    ]
)

cross = np.array(
    [
        ["a1[R]", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", "w", ".", "w", ".", ".", ".", "."],
        [".", ".", ".", ".", "w", ".", "w", ".", ".", ".", "."],
        [".", ".", ".", ".", "w", ".", "w", ".", ".", ".", "."],
        [".", "w", "w", "w", "w", ".", "w", "w", "w", "w", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "w", "w", "w", "w", ".", "w", "w", "w", "w", "."],
        [".", ".", ".", ".", "w", ".", "w", ".", ".", ".", "."],
        [".", ".", ".", ".", "w", ".", "w", ".", ".", ".", "."],
        [".", ".", ".", ".", "w", ".", "w", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "a2[L]"],
    ]
)


fourrooms_2 = np.array(
    [
        [".", ".", ".", ".", ".", ".", "w", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "w", ".", ".", ".", ".", ".", "."],
        [".", ".", "a1[D]", ".", ".", ".", "w", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "w", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "w", ".", ".", ".", ".", ".", "."],
        ["w", "w", "w", ".", "w", "w", "w", "w", "w", ".", "w", "w", "w"],
        [".", ".", ".", ".", ".", ".", "w", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "w", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "w", ".", ".", ".", "a2[U]", ".", "."],
        [".", ".", ".", ".", ".", ".", "w", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "w", ".", ".", ".", ".", ".", "."],
    ]
)

sixteenrooms_2 = np.array(
    [
        [".", ".", ".", "w", ".", ".", "w", ".", ".", "w", ".", ".", "."],
        [".", "a1[D]", ".", ".", ".", ".", ".", ".", ".", "w", ".", ".", "."],
        [".", ".", ".", "w", ".", ".", "w", ".", ".", ".", ".", ".", "."],
        ["w", ".", "w", "w", "w", ".", "w", "w", ".", "w", "w", "w", "."],
        [".", ".", ".", "w", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "w", ".", ".", "w", ".", ".", "."],
        ["w", "w", ".", "w", ".", "w", "w", ".", "w", "w", "w", ".", "w"],
        [".", ".", ".", "w", ".", ".", ".", ".", ".", "w", ".", ".", "."],
        [".", ".", ".", "w", ".", ".", "w", ".", ".", ".", ".", ".", "."],
        [".", "w", "w", "w", "w", ".", "w", "w", ".", "w", ".", "w", "w"],
        [".", ".", ".", "w", ".", ".", "w", ".", ".", "w", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "w", ".", ".", ".", ".", "a2[L]", "."],
        [".", ".", ".", "w", ".", ".", ".", ".", ".", "w", ".", ".", "."],
    ]
)


class LasertagArena1(LasertagFixed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_agents=2, level=arena_1, **kwargs)


class LasertagArena2(LasertagFixed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_agents=2, level=arena_2, **kwargs)


class LasertagCorridor1(LasertagFixed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_agents=2, level=corridor_1, **kwargs)


class LasertagCorridor2(LasertagFixed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_agents=2, level=corridor_2, **kwargs)


class LasertagMaze1(LasertagFixed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_agents=2, level=maze1, max_steps=350, **kwargs)


class LasertagMaze2(LasertagFixed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_agents=2, level=maze2, max_steps=350, **kwargs)


class LasertagRuins(LasertagFixed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_agents=2, level=ruins, max_steps=250, **kwargs)


class LasertagRuins2(LasertagFixed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_agents=2, level=ruins_2, max_steps=250, **kwargs)


class LasertagStar(LasertagFixed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_agents=2, level=star, max_steps=350, **kwargs)


class LasertagCross(LasertagFixed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_agents=2, level=cross, max_steps=350, **kwargs)


class LasertagLargeCorridor(LasertagFixed):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, n_agents=2, level=corridor_large, max_steps=350, **kwargs
        )


class LasertagFourRoomsN2(LasertagFixed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_agents=2, level=fourrooms_2, max_steps=250, **kwargs)


class LasertagSixteenRoomsN2(LasertagFixed):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, n_agents=2, level=sixteenrooms_2, max_steps=250, **kwargs
        )


def set_global(name, value):
    globals()[name] = value


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


register.register(
    id="Lasertag-Arena1-N2-v0",
    entry_point=module_path + ":LasertagArena1",
)


register.register(
    id="Lasertag-Arena2-N2-v0",
    entry_point=module_path + ":LasertagArena2",
)

register.register(
    id="Lasertag-Corridor1-N2-v0",
    entry_point=module_path + ":LasertagCorridor1",
)
register.register(
    id="Lasertag-Corridor2-N2-v0",
    entry_point=module_path + ":LasertagCorridor2",
)
register.register(
    id="Lasertag-Maze1-N2-v0",
    entry_point=module_path + ":LasertagMaze1",
)
register.register(
    id="Lasertag-Maze2-N2-v0",
    entry_point=module_path + ":LasertagMaze2",
)
register.register(
    id="Lasertag-Ruins-N2-v0",
    entry_point=module_path + ":LasertagRuins",
)
register.register(
    id="Lasertag-Ruins2-N2-v0",
    entry_point=module_path + ":LasertagRuins2",
)
register.register(
    id="Lasertag-Star-N2-v0",
    entry_point=module_path + ":LasertagStar",
)
register.register(
    id="Lasertag-Cross-N2-v0",
    entry_point=module_path + ":LasertagCross",
)
register.register(
    id="Lasertag-FourRooms-N2-v0",
    entry_point=module_path + ":LasertagFourRoomsN2",
)
register.register(
    id="Lasertag-SixteenRooms-N2-v0",
    entry_point=module_path + ":LasertagSixteenRoomsN2",
)
register.register(
    id="Lasertag-LargeCorridor-N2-v0",
    entry_point=module_path + ":LasertagLargeCorridor",
)
