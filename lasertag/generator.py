import networkx as nx
import numpy as np
from networkx import grid_graph

from . import custom_test, grid, lasertag, register
class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class LasertagAdversarial(lasertag.Lasertag):
    """Adversarial environment for multi-agent laser tag."""

    def __init__(
        self,
        *args,
        n_agents=2,  # (maximum) number of agents
        min_agents=2,  # minimum number of agents
        min_size=4,  # minimum size of the square grid (excluding outer walls)
        max_size=10,  # maximum size of the square grid (excludign outer walls)
        max_clutter_rate=0.5,  # How much of the area can be clutter (e.g. walls)
        min_clutter_rate=0,  # How much of the area can be clutter (e.g. walls)
        agent_view_size=5,  # agent view size
        max_steps=200,
        seed=0,
        fixed_environment=None,
        **kwargs,
    ):
        """Initializes environment in which adversary places goal, agent, obstacles.

        Args:
           n_clutter: The maximum number of obstacles the adversary can place.
           size: The number of tiles across one side of the grid; i.e. make a
             size x size grid.
           agent_view_size: The number of tiles in one side of the agent's partially
             observed view of the grid.
           max_steps: The maximum number of steps that can be taken before the
             episode terminates.
        """
        self.n_agents = n_agents
        self.min_agents = min_agents

        self.max_size = max_size
        self.min_size = min_size

        self.max_clutter_rate = max_clutter_rate
        self.min_clutter_rate = min_clutter_rate

        self.fixed_environment = fixed_environment

        # Seed
        self.seed(seed)

        super().__init__(
            *args,
            n_agents=n_agents,
            agent_view_size=agent_view_size,
            max_steps=max_steps,
            **kwargs,
        )

        # Metrics
        self.reset_metrics()

        # Generate the grid.
        self.grid = grid.Grid(
            n_agents=self.n_agents, width=self.max_size, height=self.max_size
        )

    def reset(self):
        self.step_count = 0

        # Extra metrics
        self.reset_metrics()

        # Generate the grid.
        self.grid = grid.Grid(
            n_agents=self.n_agents, width=self.max_size, height=self.max_size
        )

        return super().reset(level=self.grid.level)

    def seed(self, seed=None):
        np.random.seed(seed)
        return seed

    @property
    def processed_action_dim(self):
        return 1

    def reset_metrics(self):
        self.n_clutter_placed = 0
        self.clutter_rate_selected = 0
        self.n_agents_selected = 0
        self.grid_size_selected = 0
        self.solvable = -1
        self.shortest_path_length = 0

    def get_metrics(self):
        metrics = hashabledict()
        metrics["n_clutter_placed"] = self.n_clutter_placed
        metrics["clutter_rate_selected"] = self.clutter_rate_selected
        metrics["n_agents_selected"] = self.n_agents_selected
        metrics["grid_size_selected"] = self.grid_size_selected
        metrics["solvable"] = self.solvable
        metrics["shortest_path_length"] = self.shortest_path_length
        return metrics

    def set_metrics(self, **kwargs):
        self.n_clutter_placed = kwargs["n_clutter_placed"]
        self.clutter_rate_selected = kwargs["clutter_rate_selected"]
        self.n_agents_selected = kwargs["n_agents_selected"]
        self.grid_size_selected = kwargs["grid_size_selected"]
        self.solvable = kwargs["solvable"]
        self.shortest_path_length = kwargs["shortest_path_length"]

    def reset_agent(self, *args, **kwargs):
        # Step count since episode start
        self.step_count = 0

        # Return first observation
        return super().reset(*args, level=self.grid.level, **kwargs)

    @property
    def level(self):
        return self.grid.encode(self.get_metrics())

    def reset_to_level(self, level):
        metrics = self.grid.decode(level)
        self.set_metrics(**metrics)

        return self.reset_agent()

    def reset_random(self):
        """Use domain randomization to create the environment."""
        if self.fixed_environment is not None:
            if self.fixed_environment == "easy":
                self.grid.level = custom_test.corridor_easy
            elif self.fixed_environment == "four_rooms":
                self.grid.level = custom_test.fourrooms_2
            elif self.fixed_environment == "empty":
                self.grid.level = custom_test.empty_2
            elif self.fixed_environment == "ruins":
                self.grid.level = custom_test.ruins
            else:
                raise AttributeError("Wrong fixed env name")

            return self.reset_agent()

        self.step_count = 0
        self.adversary_step_count = 0

        # Choose grid size and create an empty grid
        size = np.random.randint(self.min_size, self.max_size + 1)
        self.grid = grid.Grid(n_agents=self.n_agents, width=size, height=size)

        # Choose number of agents
        n_agents = np.random.randint(self.min_agents, self.n_agents + 1)
        # Randomly place agents
        agent_ids = np.random.choice(
            range(self.n_agents), n_agents, replace=False
        )
        agent_dirs = np.random.choice(["U", "D", "L", "R"], n_agents)
        agent_strs = [
            f"a{id+1}[{dir}]" for id, dir in zip(agent_ids, agent_dirs)
        ]
        agent_locs = np.random.choice(
            range(size * size), n_agents, replace=False
        )
        self.grid.put(agent_locs, agent_strs)

        # Randomly place walls
        space = size * size - n_agents
        clutter_rate = np.random.uniform(
            low=self.min_clutter_rate, high=self.max_clutter_rate
        )
        n_clutter = int(space * clutter_rate)
        possible_locs = [x for x in range(size * size) if x not in agent_locs]
        walls = np.random.choice(possible_locs, n_clutter, replace=False)
        self.grid.put(walls, ["w"])

        # Compute metrics
        self.reset_metrics()
        self.n_clutter_placed = n_clutter
        self.clutter_rate_selected = clutter_rate
        self.n_agents_selected = n_agents
        self.grid_size_selected = size

        # Construct a graph of the grid
        self.construct_graph(size=size, walls=walls, agent_locs=agent_locs)
        self.solvable = self.is_solvable(agent_locs=agent_locs)
        if self.solvable:
            self.shortest_path_length = self.get_shortest_path(
                agent_locs, n_agents
            )
        else:
            self.shortest_path_length = 0

        return self.reset_agent()

    def construct_graph(self, size, walls, agent_locs):
        """Construct the graph of the generated grid."""
        self.graph = grid_graph(dim=[size, size])

        for w in walls:
            if w not in agent_locs:
                self.graph.remove_node(self.get_coord(w))

    def get_coord(self, loc):
        """Get coordinate from int location."""
        x = int(loc % (self.grid_size_selected))
        y = int(loc // (self.grid_size_selected))
        return x, y

    def is_solvable(self, agent_locs):
        # Check if there is a path between agent 1 and other agents.
        solvable = all(
            nx.has_path(
                self.graph,
                source=self.get_coord(agent_locs[0]),
                target=self.get_coord(next_agent),
            )
            for next_agent in agent_locs[1:]
        )
        return int(solvable)

    def get_shortest_path(self, agent_locs, n):
        """Get the shortest path between all agents. Assumes a solvable grid"""
        assert self.solvable
        shortest_path = 0
        for a, b in zip(agent_locs, agent_locs[1:]):
            shortest_path += nx.shortest_path_length(
                self.graph, source=self.get_coord(a), target=self.get_coord(b)
            )
        shortest_path = shortest_path / (n * (n - 1) / 2)

        return shortest_path


# 2 Agent - 5x5-15x15 - Walls [0, 50%]
class LasertagAdversarialN2SizeWalls(LasertagAdversarial):
    def __init__(
        self,
        seed=0,
        record_video=False,
        video_filename="",
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            n_agents=2,
            min_agents=2,
            max_size=15,
            min_size=5,
            min_clutter_rate=0,
            max_clutter_rate=0.5,
            max_steps=300,
            record_video=record_video,
            video_filename=video_filename,
            **kwargs,
        )


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


register.register(
    env_id="Lasertag-Adversarial-N2-Size-Walls-v0",
    entry_point=module_path + ":LasertagAdversarialN2SizeWalls",
    max_episode_steps=300,
)


# Fixed environment (four rooms)
class LasertagFixedFourRooms(LasertagAdversarial):
    def __init__(
        self,
        seed=0,
        record_video=False,
        video_filename="",
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            n_agents=2,
            min_agents=2,
            max_steps=200,
            fixed_environment="four_rooms",
            record_video=record_video,
            video_filename=video_filename,
            **kwargs,
        )


class LasertagFixedEasy(LasertagAdversarial):
    def __init__(
        self,
        seed=0,
        record_video=False,
        video_filename="",
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            n_agents=2,
            min_agents=2,
            max_steps=200,
            fixed_environment="easy",
            record_video=record_video,
            video_filename=video_filename,
            **kwargs,
        )


class LasertagFixedRuins(LasertagAdversarial):
    def __init__(
        self,
        seed=0,
        record_video=False,
        video_filename="",
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            n_agents=2,
            min_agents=2,
            max_steps=200,
            fixed_environment="ruins",
            record_video=record_video,
            video_filename=video_filename,
            **kwargs,
        )


class LasertagFixedEmpty(LasertagAdversarial):
    def __init__(
        self,
        seed=0,
        record_video=False,
        video_filename="",
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            n_agents=2,
            min_agents=2,
            max_steps=200,
            fixed_environment="empty",
            record_video=record_video,
            video_filename=video_filename,
            **kwargs,
        )


register.register(
    env_id="Lasertag-Adversarial-Fixed-Easy-v0",
    entry_point=module_path + ":LasertagFixedEasy",
    max_episode_steps=200,
)
register.register(
    env_id="Lasertag-Adversarial-Fixed-Empty-v0",
    entry_point=module_path + ":LasertagFixedEmpty",
    max_episode_steps=200,
)
register.register(
    env_id="Lasertag-Adversarial-Fixed-Ruins-v0",
    entry_point=module_path + ":LasertagFixedRuins",
    max_episode_steps=200,
)
register.register(
    env_id="Lasertag-Adversarial-Fixed-FourRooms-v0",
    entry_point=module_path + ":LasertagFixedFourRooms",
    max_episode_steps=200,
)
