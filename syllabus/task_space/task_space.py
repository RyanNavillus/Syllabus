import itertools
from typing import Any, List, Union

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple


class TaskSpace():
    def __init__(self, gym_space: Union[Space, int], tasks=None):
        if isinstance(gym_space, int):
            # Syntactic sugar for discrete space
            gym_space = Discrete(gym_space)

        if isinstance(gym_space, tuple):
            # Syntactic sugar for discrete space
            gym_space = MultiDiscrete(gym_space)

        if isinstance(gym_space, dict):
            # Syntactic sugar for Dict
            gym_space = Dict(gym_space)
            


        self.gym_space = gym_space

        # Autogenerate task names for discrete spaces
        if isinstance(gym_space, Discrete):
            if tasks is None:
                tasks = range(gym_space.n)

          # Autogenerate task names for multidiscrete spaces
        if isinstance(gym_space, MultiDiscrete):
            if tasks is None:
                tasks = [[] for _ in range(len(gym_space.nvec))]

        if isinstance(gym_space, Dict):
            if tasks is None:
                tasks = [[] for _ in range(len(gym_space.spaces))]

        self._tasks = set(tasks) if tasks is not None else None
        self._encoder, self._decoder = self._make_task_encoder(gym_space, tasks)

    def _make_task_encoder(self, space, tasks):
        if isinstance(space, Discrete):
            assert space.n == len(tasks), f"Number of tasks ({space.n}) must match number of discrete options ({len(tasks)})"
            self._encode_map = {task: i for i, task in enumerate(tasks)}
            self._decode_map = {i: task for i, task in enumerate(tasks)}
            encoder = lambda task: self._encode_map[task] if task in self._encode_map else None
            decoder = lambda task: self._decode_map[task] if task in self._decode_map else None

        elif isinstance(space, Box):
            encoder = lambda task: task if space.contains(np.asarray(task, dtype=space.dtype)) else None
            decoder = lambda task: task if space.contains(np.asarray(task, dtype=space.dtype)) else None
        elif isinstance(space, Tuple):
           
            assert len(space.spaces) == len(tasks), f"Number of task ({len(space.spaces)})must match options in Tuple ({len(tasks)})"
            results = [list(self._make_task_encoder(s, t)) for (s, t) in zip(space.spaces, tasks)]
            encoders = [r[0] for r in results]
            decoders = [r[1] for r in results]
            encoder = lambda task: [e(t) for e, t in zip(encoders, task)]
            decoder = lambda task: [d(t) for d, t in zip(decoders, task)]

        elif isinstance(space, MultiDiscrete):
            assert len(space.nvec) == len(tasks), f"Number of steps in a tasks ({len(space.nvec)}) must match number of discrete options ({len(tasks)})"
            
            combinations = [p for p in itertools.product(*tasks)]
            encode_map = {task: i for i, task in enumerate(combinations)}
            decode_map = {i: task for i, task in enumerate(combinations)}
    
            encoder = lambda task: encode_map[task] if task in encode_map else None
            decoder = lambda task: decode_map[task] if task in decode_map else None

        elif isinstance(space, Dict):

            def helper(input, spaces, tasks, type):
                # This helper method encodes ("type" = 0) or decodes ("type" = 1) the "input" dictionary
                # "spaces" and "tasks" contains the info of the task space
                map = {}
                if (isinstance(spaces, dict) or isinstance(spaces, Dict)):
                    for key,value in spaces.items() :
                        if (isinstance(value, dict) or isinstance(value, Dict)):
                            temp = helper(input[key], value, tasks[key], type)
                            map.update({key : temp})
                        else :
                            coders = list(self._make_task_encoder(value,tasks[key]))
                            map[key] = coders[type](input[key])
                return map

            
            encoder = lambda task: helper(task, space.spaces,tasks, 0)
            decoder = lambda task: helper(task, space.spaces,tasks, 1)


        else:
            encoder = lambda task: task
            decoder = lambda task: task
        return encoder, decoder

    def decode(self, encoding):
        """Convert the efficient task encoding to a task that can be used by the environment."""
        return self._decoder(encoding)

    def encode(self, task):
        """Convert the task to an efficient encoding to speed up multiprocessing."""
        return self._encoder(task)

    def add_task(self, task):
        """Add a task to the task space. Only implemented for discrete spaces."""
        if task not in self._tasks:
            self._tasks.add(task)
            # TODO: Increment task space size
            self.gym_space = self.increase_space()
            # TODO: Optimize adding tasks
            self._encoder, self._decoder = self._make_task_encoder(self.gym_space, self._tasks)

    def _sum_axes(list_or_size: Union[list, int]):
        if isinstance(list_or_size, int) or isinstance(list_or_size, np.int64):
            return list_or_size
        elif isinstance(list_or_size, list) or isinstance(list_or_size, np.ndarray):
            return np.prod([TaskSpace._sum_axes(x) for x in list_or_size])
        else:
            raise NotImplementedError(f"{type(list_or_size)}")

    def _enumerate_axes(self, list_or_size: Union[np.ndarray, int]):
        if isinstance(list_or_size, int) or isinstance(list_or_size, np.int64):
            return tuple(range(list_or_size))
        elif isinstance(list_or_size, list) or isinstance(list_or_size, np.ndarray):
            return tuple(itertools.product(*[self._enumerate_axes(x) for x in list_or_size]))
        else:
            raise NotImplementedError(f"{type(list_or_size)}")

    def seed(self, seed):
        self.gym_space.seed(seed)

    @property
    def tasks(self) -> List[Any]:
        # TODO: Can I just use _tasks?
        return self._tasks

    def get_tasks(self, gym_space: Space = None, sample_interval: float = None) -> List[tuple]:
        """
        Return the full list of discrete tasks in the task_space.
        Return a sample of the tasks for continuous spaces if sample_interval is specified.
        Can be overridden to exclude invalid tasks within the space.
        """
        if gym_space is None:
            gym_space = self.gym_space

        if isinstance(gym_space, Discrete):
            return list(range(gym_space.n))
        elif isinstance(gym_space, Box):
            raise NotImplementedError
        elif isinstance(gym_space, Tuple):
            return list(itertools.product([self.get_tasks(task_space=s) for s in gym_space.spaces]))
        elif isinstance(gym_space, Dict):
            return itertools.product([self.get_tasks(task_space=s) for s in gym_space.spaces.values()])
        elif isinstance(gym_space, MultiBinary):
            return list(self._enumerate_axes(gym_space.nvec))
        elif isinstance(gym_space, MultiDiscrete):
            return list(self._enumerate_axes(gym_space.nvec))
        elif gym_space is None:
            return []
        else:
            raise NotImplementedError

    @property
    def num_tasks(self) -> int:
        # TODO: Cache results
        return self.count_tasks()

    def count_tasks(self, gym_space: Space = None) -> int:
        """
        Return the number of discrete tasks in the task_space.
        Returns None for continuous spaces.
        Graph space not implemented.
        """
        # TODO: Test these implementations
        if gym_space is None:
            gym_space = self.gym_space

        if isinstance(gym_space, Discrete):
            return gym_space.n
        elif isinstance(gym_space, Box):
            return None
        elif isinstance(gym_space, Tuple):
            return sum([self.count_tasks(gym_space=s) for s in gym_space.spaces])
        elif isinstance(gym_space, Dict):
            return sum([self.count_tasks(gym_space=s) for s in gym_space.spaces.values()])
        elif isinstance(gym_space, MultiBinary):
            return TaskSpace._sum_axes(gym_space.nvec)
        elif isinstance(gym_space, MultiDiscrete):
            return TaskSpace._sum_axes(gym_space.nvec)
        elif gym_space is None:
            return 0
        else:
            raise NotImplementedError(f"Unsupported task space type: {type(gym_space)}")

    def task_name(self, task):
        return repr(self.decode(task))

    def contains(self, task):
        return task in self._tasks or self.decode(task) in self._tasks

    def increase_space(self, amount: Union[int, float] = 1):
        if isinstance(self.gym_space, Discrete):
            assert isinstance(amount, int), f"Discrete task space can only be increased by integer amount. Got {amount} instead."
            return Discrete(self.gym_space.n + amount)

    def sample(self):
        assert isinstance(self.gym_space, Discrete) or isinstance(self.gym_space, Box) or isinstance(self.gym_space, Dict) 
        return self.decode(self.gym_space.sample())

    def list_tasks(self):
        return list(self._tasks)

    def box_contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not isinstance(x, np.ndarray):
            try:
                x = np.asarray(x, dtype=self.gym_space.dtype)
            except (ValueError, TypeError):
                return False

        return not bool(x.shape == self.gym_space.shape and np.any((x < self.gym_space.low) | (x > self.gym_space.high)))
