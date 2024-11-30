import itertools
import typing
import warnings
from typing import Any, List, Tuple, Union

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Space

from syllabus.utils import UsageError


class TaskSpace:
    """
    TaskSpace is an extension of gym spaces that allows for efficient encoding and decoding of tasks.
    This is useful for environments that have a large number of tasks or require complex task representations.

    Encoding tasks provides several advantages:
    1. Minimizing the bandwidth required to transfer tasks between processes
    2. Simplifying the task formats that curricula need to support
    3. Allowing the environment to use a convenient and interpretable task format, with no impact on performance
    """

    def __init__(self, space_or_value: Union[Space, int, List, Tuple], tasks: List[Any] = None):
        """
        Generic TaskSpace initialization. Provides syntactic sugar for creating gym spaces.

        :param space_or_value: gym space or value that can be parsed into a gym space
        :type space_or_value: Union[Space, int, List, Tuple]
        :param tasks: The corresponding task representations
        :type tasks: List[Any], optional
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
        """
        Create a gym space from a simple value.

        :param gym_space: A simple value to create a gym space from
        :type gym_space: Space
        :return: Created gym space
        :rtype: Space
        """
        if isinstance(gym_space, int):
            gym_space = Discrete(gym_space)
        elif isinstance(gym_space, (tuple, list)):
            gym_space = MultiDiscrete(gym_space)
        return gym_space

    def _generate_task_names(self, gym_space: Space):
        """
        Generate basic task names for a gym space.

        :param gym_space: A gymnasium space
        :type gym_space: Space
        :return: List of task names
        :rtype: List
        """
        if isinstance(gym_space, Discrete):
            tasks = list(range(gym_space.n))
        elif isinstance(gym_space, MultiDiscrete):
            tasks = [tuple(range(dim)) for dim in gym_space.nvec]
        else:
            tasks = []
        return tasks

    def decode(self, encoding: Any) -> Any:
        """
        Convert the task encoding to the original task representation.
        This method provides generic decoding safety checks for all task spaces, and
        calls the specific _decode method for each task space. It will throw a
        UsageError if the encoding cannot be decoded into the task space.

        :param encoding: Encoding of the task
        :type encoding: Any
        :return: Decoded task that can be used by the environment
        :rtype: Any
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
        """
        Convert the task to an efficient encoding to speed up multiprocessing.
        This method provides generic encoding safety checks for all task spaces,
        and calls the specific _encode method for each task space. It will throw a
        UsageError if the task is not in the task space or cannot be encoded.

        :param task: Task to encode
        :type task: Any
        :return: Encoded task
        :rtype: Any
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
        """
        Convert the task encoding to the original task representation.
        Subclasses should implement this method for their decoding logic.

        :param encoding: Encoding of the task
        :type encoding: Any
        :return: Decoded task representation
        :rtype: Any
        """
        raise NotImplementedError

    def _encode(self, task: Any) -> Any:
        """
        Convert the task to an efficient encoding to speed up multiprocessing.
        Subclasses should implement this method for their encoding logic.

        :param task: Task to encode
        :type task: Any
        :return: Encoded task
        :rtype: Any
        """
        raise NotImplementedError

    def contains(self, encoding: Any) -> bool:
        """
        Check if the encoding is a valid task in the task space.

        :param encoding: Encoding of the task
        :type encoding: Any
        :return: Boolean specifying if the encoding is a valid task
        :rtype: bool
        """
        return encoding in self._task_set or self._decode(encoding) in self._task_set

    @property
    def tasks(self) -> List[Any]:
        """
        Return the list of all tasks in the task space.

        :return: List of all tasks
        :rtype: List[Any]
        """
        return self._task_list

    @property
    def num_tasks(self) -> int:
        """
        Return the number of tasks in the task space.

        :return: Number of tasks
        :rtype: int
        """
        return len(self._task_list)

    def task_name(self, task: int) -> str:
        """
        Return the name of the task.

        :param task: Task to get the name of
        :type task: int
        :return: Name of the task
        :rtype: str
        """
        return repr(self._decode(task))


class DiscreteTaskSpace(TaskSpace):
    """Task space for discrete tasks."""

    def __init__(self, space_or_value: Union[Space, int], tasks=None):
        """
        Initialize a discrete task space.

        :param space_or_value: gym space or value that can be parsed into a gym space
        :type space_or_value: Union[Space, int]
        :param tasks: The corresponding tasks representations
        :type tasks: List[Any], optional
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
        """
        Check if the tasks are sequential integers.

        :param tasks: List of tasks
        :type tasks: List[int]
        :return: Boolean specifying if the tasks are sequential integers
        :rtype: bool
        """
        return isinstance(tasks[0], (int, np.integer)) and tuple(tasks) == tuple(range(tasks[0], tasks[-1] + 1))

    def _decode(self, encoding: int) -> int:
        """
        Convert the task encoding to the original task representation.

        :param encoding: Encoding of the task
        :type encoding: int
        :return: Decoded task representation
        :rtype: int
        """
        assert isinstance(encoding, (int, np.integer)), f"Encoding must be an integer. Got {type(encoding)} instead."
        if self._sequential:
            task = encoding + self._first_task
            if task < self._first_task or task > self._last_task:
                raise UsageError(f"Encoding {encoding} does not map to a task in the task space")
            return task
        else:
            return self._decode_map[encoding]

    def _encode(self, task: int) -> int:
        """
        Convert the task to an efficient encoding.

        :param task: Task to encode
        :type task: int
        :return: Encoded task
        :rtype: int
        """
        if self._sequential:
            assert isinstance(task, (int, np.integer)), f"Task must be an integer. Got {type(task)} instead."
            if task < self._first_task or task > self._last_task:
                raise UsageError(f"Task {task} is not in the task space")
            return task - self._first_task
        else:
            return self._encode_map[task]

    def sample(self) -> int:
        """
        Sample a task from the task space.

        :return: Sampled task
        :rtype: int
        """
        sample = self.gym_space.sample()
        return self._decode(sample)

    def seed(self, seed: int):
        """
        Seed the task space.

        :param seed: Seed value
        :type seed: int
        """
        self.gym_space.seed(seed)


class BoxTaskSpace(TaskSpace):
    """Task space for continuous tasks."""

    def _decode(self, encoding: np.ndarray) -> np.ndarray:
        """
        Convert the task encoding to the original task representation.

        :param encoding: Encoding of the task
        :type encoding: np.ndarray
        :return: Decoded task representation
        :rtype: np.ndarray
        :raises UsageError: If encoding does not map to a task in the task space
        """
        assert isinstance(encoding, np.ndarray), f"Encoding must be a numpy array. Got {type(encoding)} instead."
        if not self.contains(encoding):
            raise UsageError(f"Encoding {encoding} does not map to a task in the task space")
        return encoding

    def _encode(self, task: np.ndarray) -> np.ndarray:
        """
        Convert the task to an efficient encoding.

        :param task: Task to encode
        :type task: np.ndarray
        :return: Encoded task
        :rtype: np.ndarray
        """
        return task

    def sample(self) -> np.ndarray:
        """
        Sample a task from the task space.

        :return: Sampled task
        :rtype: np.ndarray
        """
        sample = self.gym_space.sample()
        return self._decode(sample)

    def seed(self, seed: int):
        """
        Seed the task space.

        :param seed: Seed value
        :type seed: int
        """
        self.gym_space.seed(seed)

    @property
    def tasks(self) -> List[Any]:
        """
        Return the list of all tasks in the task space.

        :return: List of all tasks
        :rtype: List[Any]
        """
        return None

    @property
    def num_tasks(self) -> int:
        """
        Return the number of tasks in the task space.

        :return: Number of tasks
        :rtype: int
        """
        return -1

    def task_name(self, task: np.ndarray) -> str:
        """
        Return the name of the task.

        :param task: Task to get the name of
        :type task: np.ndarray
        :return: Name of the task
        :rtype: str
        """
        return repr(self._decode(task))

    def contains(self, encoding: np.ndarray) -> bool:
        """
        Return boolean specifying if encoding is a valid member of this space.

        :param encoding: Encoding of the task
        :type encoding: np.ndarray
        :return: Boolean specifying if encoding is a valid task
        :rtype: bool
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

    def __init__(self, space_or_value: Union[MultiDiscrete, int], tasks: Union[List[Any], Tuple[Any]] = None, flatten: bool = False):
        """
        Initialize a multi-discrete task space.

        :param space_or_value: gym space or value that can be parsed into a gym space
        :type space_or_value: Union[MultiDiscrete, int]
        :param tasks: The corresponding tasks representations
        :type tasks: Union[List[Any], Tuple[Any]], optional
        :param flatten: Whether to flatten the encoding into a discrete list
        :type flatten: bool
        """
        super().__init__(space_or_value, tasks)

        self.flatten = flatten
        self._all_tasks = list(itertools.product(*self._task_list))
        self._encode_maps = [{task: i for i, task in enumerate(tasks)} for tasks in self._task_list]
        self._decode_maps = [{i: task for i, task in enumerate(tasks)} for tasks in self._task_list]

    def _is_sequential(self, tasks: List[int]) -> bool:
        """
        Check if the tasks are sequential integers.

        :param tasks: List of tasks
        :type tasks: List[int]
        :return: Boolean specifying if the tasks are sequential integers
        :rtype: bool
        """
        return isinstance(tasks[0], (int, np.integer)) and tuple(tasks) == tuple(range(tasks[0], tasks[-1] + 1))

    def _decode(self, encoding: Union[int, Tuple[int]]) -> Tuple[int]:
        """
        Convert the task encoding to the original task representation.

        :param encoding: Encoding of the task
        :type encoding: Union[int, Tuple[int]]
        :return: Decoded task representation
        :rtype: Tuple[int]
        """
        assert isinstance(encoding, (int, np.integer, tuple)
                          ), f"Encoding must be an integer or tuple. Got {type(encoding)} instead."
        if self.flatten:
            assert isinstance(encoding, (int, np.integer)
                              ), f"Encoding must be an integer. Got {type(encoding)} instead."
            encoding = np.unravel_index(encoding, self.gym_space.nvec)
        if len(encoding) != len(self._decode_maps):
            raise UsageError(
                f"Encoding length ({len(encoding)}) must match number of discrete spaces ({len(self._decode_maps)})")
        return tuple(decode_map[t] for decode_map, t in zip(self._decode_maps, encoding))

    def _encode(self, task: Tuple[Any]) -> int:
        """
        Convert the task to an efficient encoding.

        :param task: Task to encode
        :type task: Tuple[Any]
        :return: Encoded task
        :rtype: int
        """
        if len(task) != len(self._encode_maps):
            raise UsageError(
                f"Task length ({len(task)}) must match number of discrete spaces ({len(self._encode_maps)})")
        encoding = tuple(encode_map[t] for encode_map, t in zip(self._encode_maps, task))
        if self.flatten:
            encoding = np.ravel_multi_index(encoding, self.gym_space.nvec)
        return encoding

    def sample(self):
        """
        Sample a task from the task space.

        :return: Sampled task
        :rtype: int
        """
        sample = self.gym_space.sample()
        if self.flatten:
            sample = np.ravel_multi_index(sample, self.gym_space.nvec)
        return self._decode(sample)

    def seed(self, seed: int):
        """
        Seed the task space.

        :param seed: Seed value
        :type seed: int
        """
        self.gym_space.seed(seed)

    @property
    def tasks(self) -> List[Any]:
        """
        Return the list of all tasks in the task space.

        :return: List of all tasks
        :rtype: List[Any]
        """
        return self._all_tasks

    @property
    def num_tasks(self) -> int:
        """
        Return the number of tasks in the task space.

        :return: Number of tasks
        :rtype: int
        """
        return int(np.prod(self.gym_space.nvec))


class TupleTaskSpace(TaskSpace):
    """Task space for tuple tasks. Can be used to combine multiple task spaces into a single task space."""

    def __init__(self, task_spaces: Tuple[TaskSpace], space_names: Tuple = None, flatten: bool = False):
        """
        Initialize a tuple task space.

        :param task_spaces: Tuple of task spaces
        :type task_spaces: Tuple[TaskSpace]
        :param space_names: Names of the spaces
        :type space_names: Tuple, optional
        :param flatten: Whether to flatten the encoding into a discrete list
        :type flatten: bool
        """
        super().__init__(None, None)
        self.task_spaces = task_spaces
        self.space_names = space_names
        self.flatten = flatten

        if self.flatten:
            for space in self.task_spaces:
                if hasattr(space, "flatten"):
                    space.flatten = self.flatten

        self._all_tasks = None
        self._task_nums = tuple(space.num_tasks for space in self.task_spaces)

    def _is_sequential(self, tasks: Tuple[int]) -> bool:
        """
        Check if the tasks are sequential integers.

        :param tasks: List of tasks
        :type tasks: Tuple[int]
        :return: Boolean specifying if the tasks are sequential integers
        :rtype: bool
        """
        return isinstance(tasks[0], (int, np.integer)) and tuple(tasks) == tuple(range(tasks[0], tasks[-1] + 1))

    def _decode(self, encoding: Union[int, Tuple[Any]]) -> Tuple[int]:
        """
        Convert the task encoding to the original task representation.

        :param encoding: Encoding of the task
        :type encoding: int
        :return: Decoded task representation
        :rtype: Tuple[Any]
        """
        assert isinstance(encoding, (int, np.integer, tuple)
                          ), f"Encoding must be an integer or tuple. Got {type(encoding)} instead."
        if self.flatten:
            assert isinstance(encoding, (int, np.integer)
                              ), f"Encoding must be an integer. Got {type(encoding)} instead."
            encoding = np.unravel_index(encoding, self._task_nums)
        if len(encoding) != len(self.task_spaces):
            raise UsageError(
                f"Encoding length ({len(encoding)}) must match number of task spaces ({len(self.task_spaces)})")
        return tuple(space.decode(t) for space, t in zip(self.task_spaces, encoding))

    def _encode(self, task: Tuple[Any]):
        """
        Convert the task to an efficient encoding.

        :param task: Task to encode
        :type task: Tuple[Any]
        :return: Encoded task
        :rtype: int
        """
        if len(task) != len(self.task_spaces):
            raise UsageError(
                f"Task length ({len(task)}) must match number of task spaces ({len(self.task_spaces)})")
        encoding = tuple(space.encode(t) for space, t in zip(self.task_spaces, task))
        if self.flatten:
            encoding = np.ravel_multi_index(encoding, self._task_nums)
        return encoding

    def contains(self, encoding: int) -> bool:
        """
        Check if the encoding is a valid task in the task space.

        :param encoding: Encoding of the task
        :type encoding: int
        :return: Boolean specifying if the encoding is a valid task
        :rtype: bool
        """
        for element, space in zip(encoding, self.task_spaces):
            if not space.contains(element):
                return False
        return True

    def sample(self) -> Tuple[Any]:
        """
        Sample a task from the task space.

        :return: Sampled task
        :rtype: Tuple[Any]
        """
        return [space.sample() for space in self.task_spaces]

    def seed(self, seed: int):
        """
        Seed all subspaces.

        :param seed: Seed value
        :type seed: int
        """
        for space in self.task_spaces:
            space.seed(seed)

    @property
    def tasks(self) -> List[Any]:
        """
        Return the list of all tasks in the task space.

        :return: List of all tasks
        :rtype: List[Any]
        """
        if self._all_tasks is None:
            task_lists = [space.tasks for space in self.task_spaces]
            self._all_tasks = list(itertools.product(*task_lists))
        return self._all_tasks

    @property
    def num_tasks(self) -> int:
        """
        Return the number of tasks in the task space.

        :return: Number of tasks
        :rtype: int
        """
        return int(np.prod(self._task_nums))

    def task_name(self, task: Tuple[int]) -> str:
        """
        Return the name of the task.

        :param task: Task to get the name of
        :type task: Tuple[int]
        :return: Name of the task
        :rtype: str
        """
        return repr(self._decode(task))
