import itertools
import typing
import warnings
from typing import Any, List, Union

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Space

from syllabus.utils import UsageError


class TaskSpace:
    def __init__(self, space_or_value, tasks=None):
        # Syntactic sugar for creating gym spaces
        if isinstance(space_or_value, Space):
            self.gym_space = space_or_value
        else:
            self.gym_space = self._create_gym_space(space_or_value)

        # Autogenerate task names
        if tasks is None:
            tasks = self._generate_task_names(self.gym_space)

        self._task_set = set(tasks)
        self._task_list = tasks

    def _create_gym_space(self, gym_space):
        if isinstance(gym_space, int):
            gym_space = Discrete(gym_space)
        elif isinstance(gym_space, (tuple, list)):
            gym_space = MultiDiscrete(gym_space)
        return gym_space

    def _generate_task_names(self, gym_space):
        if isinstance(gym_space, Discrete):
            tasks = list(range(gym_space.n))
        elif isinstance(gym_space, MultiDiscrete):
            tasks = [tuple(range(dim)) for dim in gym_space.nvec]
        else:
            tasks = []
        return tasks

    def decode(self, encoding):
        """Convert the efficient task encoding to a task that can be used by the environment."""
        try:
            return self._decode(encoding)
        except KeyError as e:
            # Check if task is already in decoded form
            try:
                self._encode(encoding)
                warnings.warn(f"Task encoding already decoded: {encoding}", stacklevel=2)
                return encoding
            except (KeyError, TypeError):
                raise UsageError(f"Failed to decode task encoding: {encoding}") from e
        except ValueError as e:
            raise UsageError(f"Failed to decode task encoding: {encoding}") from e

    def encode(self, task):
        """Convert the task to an efficient encoding to speed up multiprocessing."""
        try:
            return self._encode(task)
        except KeyError as e:
            try:
                self._decode(task)
                warnings.warn(f"Task already encoded: {task}", stacklevel=2)
                return task
            except (KeyError, TypeError):
                raise UsageError(f"Failed to encode task: {task}") from e
        except ValueError as e:
            raise UsageError(f"Failed to encode task: {task}") from e

    def _decode(self, encoding):
        raise NotImplementedError

    def _encode(self, task):
        raise NotImplementedError

    def contains(self, encoding):
        return encoding in self._task_set or self._decode(encoding) in self._task_set


class DiscreteTaskSpace(TaskSpace):
    def __init__(self, space_or_value: Union[Space, int], tasks=None):
        super().__init__(space_or_value, tasks)

        # Use space efficient implementation for sequential task spaces
        self._sequential = self._is_sequential(self.tasks)
        if self._sequential:
            self._first_task = self.tasks[0]     # First and smallest task
            self._last_task = self.tasks[-1]     # Last and largest task
        else:
            self._encode_map = {task: i for i, task in enumerate(self.tasks)}
            self._decode_map = {i: task for i, task in enumerate(self.tasks)}

    def _is_sequential(self, tasks):
        """Check if the tasks are sequential integers."""
        return isinstance(tasks[0], int) and tuple(tasks) == tuple(range(tasks[0], tasks[-1] + 1))

    def _decode(self, encoding):
        if self._sequential:
            task = encoding + self._first_task
            if task < self._first_task or task > self._last_task:
                raise UsageError(f"Encoding {encoding} does not map to a task in the task space")
            return task
        else:
            return self._decode_map[encoding]

    def _encode(self, task):
        if self._sequential:
            if task < self._first_task or task > self._last_task:
                raise UsageError(f"Task {task} is not in the task space")
            return task - self._first_task
        else:
            return self._encode_map[task]

    def sample(self):
        sample = self.gym_space.sample()
        return self._decode(sample)

    def seed(self, seed):
        self.gym_space.seed(seed)

    @property
    def tasks(self) -> List[Any]:
        return self._task_list

    @property
    def num_tasks(self) -> int:
        return len(self._task_list)

    def task_name(self, task):
        return repr(self._decode(task))


class BoxTaskSpace(TaskSpace):
    def _decode(self, encoding):
        if not self.contains(encoding):
            raise UsageError(f"Encoding {encoding} does not map to a task in the task space")
        return encoding

    def _encode(self, task):
        return task

    def sample(self):
        sample = self.gym_space.sample()
        return self._decode(sample)

    def seed(self, seed):
        self.gym_space.seed(seed)

    @property
    def tasks(self) -> List[Any]:
        return None

    @property
    def num_tasks(self) -> int:
        return -1

    def task_name(self, task):
        return repr(self._decode(task))

    def contains(self, encoding):
        """Return boolean specifying if x is a valid member of this space.
        Replaces slow gymnasium implementation.
        """
        if not isinstance(encoding, np.ndarray):
            try:
                encoding = np.asarray(encoding, dtype=self.gym_space.dtype)
            except (ValueError, TypeError):
                return False

        shape_check = encoding.shape == self.gym_space.shape
        bounds_check = np.all((encoding >= self.gym_space.low) & (encoding <= self.gym_space.high))
        return shape_check and bounds_check

    def to_multidiscrete(self, grid_points: Union[int, List[int]]):
        # Convert to Box Task Space to MultiDiscrete Task Space
        if isinstance(self.gym_space, Box):
            elements = self.gym_space.shape[0]
            print(self.gym_space.shape)


class MultiDiscreteTaskSpace(TaskSpace):
    def __init__(self, space_or_value: Union[MultiDiscrete, int], tasks=None, flatten=False):
        super().__init__(space_or_value, tasks)

        self.flatten = flatten
        self._all_tasks = list(itertools.product(*self._task_list))
        self._encode_maps = [{task: i for i, task in enumerate(tasks)} for tasks in self._task_list]
        self._decode_maps = [{i: task for i, task in enumerate(tasks)} for tasks in self._task_list]

    def _is_sequential(self, tasks):
        """Check if the tasks are sequential integers."""
        return isinstance(tasks[0], int) and tuple(tasks) == tuple(range(tasks[0], tasks[-1] + 1))

    def _decode(self, encoding):
        if self.flatten:
            # Convert single index into tuple of indices, where each component has a different number of options listed in nvec
            encoding = np.unravel_index(encoding, self.gym_space.nvec)
        if len(encoding) != len(self._decode_maps):
            raise UsageError(
                f"Encoding length ({len(encoding)}) must match number of discrete spaces ({len(self._decode_maps)})")
        return tuple(decode_map[t] for decode_map, t in zip(self._decode_maps, encoding))

    def _encode(self, task):
        if len(task) != len(self._encode_maps):
            raise UsageError(
                f"Task length ({len(task)}) must match number of discrete spaces ({len(self._encode_maps)})")
        encoding = tuple(encode_map[t] for encode_map, t in zip(self._encode_maps, task))
        if self.flatten:
            # Convert tuple of indices into a single index, where each component has a different number of options listed in nvec
            encoding = np.ravel_multi_index(encoding, self.gym_space.nvec)
        return encoding

    def sample(self):
        sample = self.gym_space.sample()
        if self.flatten:
            sample = np.ravel_multi_index(sample, self.gym_space.nvec)
        return self._decode(sample)

    def seed(self, seed):
        self.gym_space.seed(seed)

    @ property
    def tasks(self) -> List[Any]:
        return self._all_tasks

    @ property
    def num_tasks(self) -> int:
        return int(np.prod(self.gym_space.nvec))

    def task_name(self, task):
        return repr(self._decode(task))


class TupleTaskSpace(TaskSpace):
    def __init__(self, task_spaces: typing.Tuple[TaskSpace], space_names=None, flatten=False):
        super().__init__(None, None)
        self.task_spaces = task_spaces
        self.space_names = space_names
        self.flatten = flatten

        # Force all subspaces to match flatten setting
        if self.flatten:
            for space in self.task_spaces:
                if hasattr(space, "flatten"):
                    space.flatten = self.flatten

        self._all_tasks = None
        self._task_nums = tuple(space.num_tasks for space in self.task_spaces)

    def _is_sequential(self, tasks):
        """Check if the tasks are sequential integers."""
        return isinstance(tasks[0], int) and tuple(tasks) == tuple(range(tasks[0], tasks[-1] + 1))

    def _decode(self, encoding):
        if self.flatten:
            # Convert single flat index into tuple of indices
            encoding = np.unravel_index(encoding, self._task_nums)
        if len(encoding) != len(self.task_spaces):
            raise UsageError(
                f"Encoding length ({len(encoding)}) must match number of task spaces ({len(self.task_spaces)})")
        return tuple(space.decode(t) for space, t in zip(self.task_spaces, encoding))

    def _encode(self, task):
        if len(task) != len(self.task_spaces):
            raise UsageError(
                f"Task length ({len(task)}) must match number of task spaces ({len(self.task_spaces)})")
        encoding = tuple(space.encode(t) for space, t in zip(self.task_spaces, task))
        if self.flatten:
            # Convert tuple of indices into single flat index
            encoding = np.ravel_multi_index(encoding, self._task_nums)
        return encoding

    def contains(self, encoding):
        for element, space in zip(encoding, self.task_spaces):
            if not space.contains(element):
                return False
        return True

    def sample(self):
        return [space.sample() for space in self.task_spaces]

    def seed(self, seed):
        for space in self.task_spaces:
            space.seed(seed)

    @ property
    def tasks(self) -> List[Any]:
        if self._all_tasks is None:
            # Expand all task_spaces
            task_lists = [space.tasks for space in self.task_spaces]
            self._all_tasks = list(itertools.product(*task_lists))
        return self._all_tasks

    @ property
    def num_tasks(self) -> int:
        return int(np.prod(self._task_nums))

    def task_name(self, task):
        # TODO: Add space_names here
        return repr(self._decode(task))
