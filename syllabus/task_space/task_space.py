import itertools
import typing
import warnings
from typing import Any, List, Tuple, Union

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Space

from syllabus.utils import UsageError


class TaskSpace:
    """ TaskSpace is an extension of gym spaces that allows for efficient encoding and decoding of tasks.
    This is useful for environments that have a large number of tasks or require complex task representations.

    Encoding tasks provides several advantages:
    1. Minimizing the bandwidth required to transfer tasks between processes
    2. Simplifying the task formats that curricula need to support
    3. Allowing the environment to use a convenient and interpretable task format, with no impact on performance
    """

    def __init__(self, space_or_value: Union[Space, int, List, Tuple], tasks: List[Any] = None):
        """ Generic TaskSpace initialization. Provides syntactic sugar for creating gym spaces.
        Args:
            space_or_value: gym space or value that can be parsed into a gym space
            tasks: The corresponding task representations
        """
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

    def _create_gym_space(self, gym_space: Space):
        """Create a gym space from a simple value."""
        if isinstance(gym_space, int):
            gym_space = Discrete(gym_space)
        elif isinstance(gym_space, (tuple, list)):
            gym_space = MultiDiscrete(gym_space)
        return gym_space

    def _generate_task_names(self, gym_space: Space):
        """Generate basic task names for a gym space.

        Args:
            gym_space: A gymnasium space
        """
        if isinstance(gym_space, Discrete):
            tasks = list(range(gym_space.n))
        elif isinstance(gym_space, MultiDiscrete):
            tasks = [tuple(range(dim)) for dim in gym_space.nvec]
        else:
            tasks = []
        return tasks

    def decode(self, encoding: Any) -> Any:
        """Convert the task encoding to the original task representation.
        This method provides generic decoding safety checks all task spaces, and
        calls the specific _encode method for each task space. It will throw a
        UsageError if the encoding cannot be decoded into the task space.

        Args:
            encoding: Encoding of the task
        Returns:
            Decoded task that can be used by the environment
        """
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

    def encode(self, task: Any) -> Any:
        """Convert the task to an efficient encoding to speed up multiprocessing.
        This method provides generic encoding safety checks for all task spaces,
        and calls the specific _encode method for each task space. It will throw a
        UsageError if the task is not in the task space or cannot be encoded.

        Args:
            task: Task to encode
        Returns:
            Encoded task
        """
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

    def _decode(self, encoding: Any) -> Any:
        """ Convert the task encoding to the original task representation.
        Subclasses should implement this method for their decoding logic.

        Args:
            encoding: Encoding of the task
        Returns:
            Decoded task representation
        """
        raise NotImplementedError

    def _encode(self, task: Any) -> Any:
        """ Convert the task to an efficient encoding to speed up multiprocessing.
        Subclasses should implement this method for their encoding logic.

        Args:
            task: Task to encode
        Returns:
            Encoded task
        """
        raise NotImplementedError

    def contains(self, encoding: Any) -> bool:
        """Check if the encoding is a valid task in the task space.

        Args:
            encoding: Encoding of the task
        Returns:
            Boolean specifying if the encoding is a valid task
        """
        return encoding in self._task_set or self._decode(encoding) in self._task_set

    @property
    def tasks(self) -> List[Any]:
        """ Return the list of all tasks in the task space.

        Returns:
            List of all tasks
        """
        return self._task_list

    @property
    def num_tasks(self) -> int:
        """ Return the number of tasks in the task space.

        Returns:
            Number of tasks
        """
        return len(self._task_list)

    def task_name(self, task: int) -> str:
        """ Return the name of the task.

        Args:
            task: Task to get the name of

        Returns:
            Name of the task
        """
        return repr(self._decode(task))


class DiscreteTaskSpace(TaskSpace):
    """Task space for discrete tasks."""

    def __init__(self, space_or_value: Union[Space, int], tasks=None):
        """Initialize a discrete task space.

        Args:
            space_or_value: gym space or value that can be parsed into a gym space
            tasks: The corresponding tasks representations
        """
        super().__init__(space_or_value, tasks)

        # Use space efficient implementation for sequential task spaces
        self._sequential = self._is_sequential(self.tasks)
        if self._sequential:
            self._first_task = self.tasks[0]     # First and smallest task
            self._last_task = self.tasks[-1]     # Last and largest task
        else:
            self._encode_map = {task: i for i, task in enumerate(self.tasks)}
            self._decode_map = {i: task for i, task in enumerate(self.tasks)}

    def _is_sequential(self, tasks: List[int]):
        """Check if the tasks are sequential integers.

        Args:
            tasks: List of tasks
        Returns:
            Boolean specifying if the tasks are sequential integers
        """
        return isinstance(tasks[0], int) and tuple(tasks) == tuple(range(tasks[0], tasks[-1] + 1))

    def _decode(self, encoding: int) -> int:
        """ Convert the task encoding to the original task representation."""
        if self._sequential:
            task = encoding + self._first_task
            if task < self._first_task or task > self._last_task:
                raise UsageError(f"Encoding {encoding} does not map to a task in the task space")
            return task
        else:
            return self._decode_map[encoding]

    def _encode(self, task: int) -> int:
        """ Convert the task to an efficient encoding."""
        if self._sequential:
            if task < self._first_task or task > self._last_task:
                raise UsageError(f"Task {task} is not in the task space")
            return task - self._first_task
        else:
            return self._encode_map[task]

    def sample(self) -> int:
        """ Sample a task from the task space.

        Returns:
            Sampled task
        """
        sample = self.gym_space.sample()
        return self._decode(sample)

    def seed(self, seed: int):
        """ Seed the task space.

        Args:
            seed: Seed value
        """
        self.gym_space.seed(seed)


class BoxTaskSpace(TaskSpace):
    """Task space for continuous tasks."""

    def _decode(self, encoding: np.ndarray) -> np.ndarray:
        """ Convert the task encoding to the original task representation."""
        if not self.contains(encoding):
            raise UsageError(f"Encoding {encoding} does not map to a task in the task space")
        return encoding

    def _encode(self, task: np.ndarray) -> np.ndarray:
        """ Convert the task to an efficient encoding."""
        return task

    def sample(self):
        """ Sample a task from the task space.

        Returns:
            Sampled task
        """
        sample = self.gym_space.sample()
        return self._decode(sample)

    def seed(self, seed: int):
        """ Seed the task space.

        Args:
            seed: Seed value
        """
        self.gym_space.seed(seed)

    @property
    def tasks(self) -> List[Any]:
        """ Return the list of all tasks in the task space.

        Returns:
            List of all tasks
        """
        return None

    @property
    def num_tasks(self) -> int:
        """ Return the number of tasks in the task space.

        Returns:
            Number of tasks
        """
        return -1

    def task_name(self, task: np.ndarray) -> str:
        """ Return the name of the task.

        Args:
            task: Task to get the name of
        Returns:
            Name of the task
        """
        return repr(self._decode(task))

    def contains(self, encoding: np.ndarray) -> bool:
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

    # def to_multidiscrete(self, grid_points: Union[int, List[int]]):
    #     # Convert to Box Task Space to MultiDiscrete Task Space
    #     if isinstance(self.gym_space, Box):
    #         elements = self.gym_space.shape[0]
    #         print(self.gym_space.shape)


class MultiDiscreteTaskSpace(TaskSpace):
    """Task space for multi-discrete tasks."""

    def __init__(self, space_or_value: Union[MultiDiscrete, int], tasks: Union[List[Any], typing.Tuple[Any]] = None, flatten: bool = False):
        """Initialize a multi-discrete task space.

        Args:
            space_or_value: gym space or value that can be parsed into a gym space
            tasks: The corresponding tasks representations
            flatten: Whether to flatten the encoding into a discrete list
        """
        super().__init__(space_or_value, tasks)

        self.flatten = flatten
        self._all_tasks = list(itertools.product(*self._task_list))
        self._encode_maps = [{task: i for i, task in enumerate(tasks)} for tasks in self._task_list]
        self._decode_maps = [{i: task for i, task in enumerate(tasks)} for tasks in self._task_list]

    def _is_sequential(self, tasks: List[int]) -> bool:
        """Check if the tasks are sequential integers.

        Args:
            tasks: List of tasks
        Returns:
            Boolean specifying if the tasks are sequential integers
        """
        return isinstance(tasks[0], int) and tuple(tasks) == tuple(range(tasks[0], tasks[-1] + 1))

    def _decode(self, encoding: Union[int, typing.Tuple[int]]) -> Tuple[int]:
        """ Convert the task encoding to the original task representation."""
        if self.flatten:
            # Convert single index into tuple of indices, where each component has a different number of options listed in nvec
            encoding = np.unravel_index(encoding, self.gym_space.nvec)
        if len(encoding) != len(self._decode_maps):
            raise UsageError(
                f"Encoding length ({len(encoding)}) must match number of discrete spaces ({len(self._decode_maps)})")
        return tuple(decode_map[t] for decode_map, t in zip(self._decode_maps, encoding))

    def _encode(self, task: typing.Tuple[Any]) -> int:
        """ Convert the task to an efficient encoding."""
        if len(task) != len(self._encode_maps):
            raise UsageError(
                f"Task length ({len(task)}) must match number of discrete spaces ({len(self._encode_maps)})")
        encoding = tuple(encode_map[t] for encode_map, t in zip(self._encode_maps, task))
        if self.flatten:
            # Convert tuple of indices into a single index, where each component has a different number of options listed in nvec
            encoding = np.ravel_multi_index(encoding, self.gym_space.nvec)
        return encoding

    def sample(self):
        """ Sample a task from the task space."""
        sample = self.gym_space.sample()
        if self.flatten:
            sample = np.ravel_multi_index(sample, self.gym_space.nvec)
        return self._decode(sample)

    def seed(self, seed: int):
        """ Seed the task space."""
        self.gym_space.seed(seed)

    @ property
    def tasks(self) -> List[Any]:
        """ Return the list of all tasks in the task space."""
        return self._all_tasks

    @ property
    def num_tasks(self) -> int:
        """ Return the number of tasks in the task space."""
        return int(np.prod(self.gym_space.nvec))


class TupleTaskSpace(TaskSpace):
    """Task space for tuple tasks. Can be used to combine multiple task spaces into a single task space."""

    def __init__(self, task_spaces: typing.Tuple[TaskSpace], space_names: typing.Tuple = None, flatten: bool = False):
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

    def _is_sequential(self, tasks: typing.Tuple[int]):
        """Check if the tasks are sequential integers."""
        return isinstance(tasks[0], int) and tuple(tasks) == tuple(range(tasks[0], tasks[-1] + 1))

    def _decode(self, encoding: int):
        """ Convert the task encoding to the original task representation."""
        if self.flatten:
            # Convert single flat index into tuple of indices
            encoding = np.unravel_index(encoding, self._task_nums)
        if len(encoding) != len(self.task_spaces):
            raise UsageError(
                f"Encoding length ({len(encoding)}) must match number of task spaces ({len(self.task_spaces)})")
        return tuple(space.decode(t) for space, t in zip(self.task_spaces, encoding))

    def _encode(self, task: typing.Tuple[Any]):
        """ Convert the task to an efficient encoding."""
        if len(task) != len(self.task_spaces):
            raise UsageError(
                f"Task length ({len(task)}) must match number of task spaces ({len(self.task_spaces)})")
        encoding = tuple(space.encode(t) for space, t in zip(self.task_spaces, task))
        if self.flatten:
            # Convert tuple of indices into single flat index
            encoding = np.ravel_multi_index(encoding, self._task_nums)
        return encoding

    def contains(self, encoding: int):
        """Check if the encoding is a valid task in the task space."""
        for element, space in zip(encoding, self.task_spaces):
            if not space.contains(element):
                return False
        return True

    def sample(self) -> typing.Tuple[Any]:
        """ Sample a task from the task space."""
        return [space.sample() for space in self.task_spaces]

    def seed(self, seed: int):
        """ Seed all subspaces."""
        for space in self.task_spaces:
            space.seed(seed)

    @ property
    def tasks(self) -> List[Any]:
        """ Return the list of all tasks in the task space."""
        if self._all_tasks is None:
            # Expand all task_spaces
            task_lists = [space.tasks for space in self.task_spaces]
            self._all_tasks = list(itertools.product(*task_lists))
        return self._all_tasks

    @ property
    def num_tasks(self) -> int:
        """ Return the number of tasks in the task space."""
        return int(np.prod(self._task_nums))

    def task_name(self, task: typing.Tuple[int]) -> str:
        """ Return the name of the task."""
        # TODO: Add space_names here
        return repr(self._decode(task))
