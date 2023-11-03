import itertools
from typing import Any, List, Union

import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple

#from syllabus.core import enumerate_axes


class TaskSpace():
    def __init__(self, gym_space, tasks):
        self.gym_space = gym_space
        self._encoder, self._decoder = self._make_task_encoder(gym_space, tasks)
        self._tasks = set(tasks)

    def _make_task_encoder(self, space, tasks):
        if isinstance(space, Discrete):
            assert space.n == len(tasks), f"Number of tasks ({space.n}) must match number of discrete options ({len(tasks)})"
            self._encode_map = {task: i for i, task in enumerate(tasks)}
            self._decode_map = {i: task for i, task in enumerate(tasks)}
            encoder = lambda task: self._encode_map[task]
            decoder = lambda task: self._decode_map[task]
        elif isinstance(space, Box):
            encoder = lambda task: task if space.contains(np.asarray(task, dtype=space.dtype)) else None
            decoder = lambda task: task if space.contains(np.asarray(task, dtype=space.dtype)) else None
        elif isinstance(space, Tuple):
            for i, task in enumerate(tasks):
                assert self.count_tasks(space.spaces[i]) == len(task), "Each task must have number of components equal to Tuple space length. Got {len(task)} components and space length {self.count_tasks(space.spaces[i])}."
            results = [list(self._make_task_encoder(s, t)) for (s, t) in zip(space.spaces, tasks)]
            encoders = [r[0] for r in results]
            decoders = [r[1] for r in results]
            encoder = lambda task: [e(t) for e, t in zip(encoders, task)]
            decoder = lambda task: [d(t) for d, t in zip(decoders, task)]
            #raise NotImplementedError(f"Task space not implemented for this gym space: {self.gym_space}")
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
        assert isinstance(self.gym_space, Discrete) or isinstance(self.gym_space, Box)
        return self.decode(self.gym_space.sample())