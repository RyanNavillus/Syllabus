import itertools
from typing import Any, List, Union
import typing
import warnings

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
from syllabus.utils import UsageError


class TaskSpace():
    def __init__(self, gym_space: Union[Space, int], tasks=None):

        if not isinstance(gym_space, Space):
            gym_space = self._create_gym_space(gym_space)

        if isinstance(gym_space, tuple) and all([isinstance(x, int) for x in gym_space]):
            # Syntactic sugar for multidiscrete space
            gym_space = MultiDiscrete(gym_space)

        # Expand Multidiscrete tasks
        task_list = tasks
        if isinstance(gym_space, MultiDiscrete) and tasks is not None:
            # TODO: Add some check that task names are correct shape for gym space
            task_list = list(itertools.product(*tasks))

        self.gym_space = gym_space

        # Autogenerate task names
        if tasks is None:
            tasks = self._generate_task_names(gym_space)

        self._task_set = set(task_list) if task_list is not None else None
        self._task_list = task_list
        self._encoder, self._decoder = self._make_task_encoder(gym_space, tasks)
        sample = self.sample()
        self.task_shape = np.array(self.encode(sample)).shape

    def _create_gym_space(self, gym_space):
        if isinstance(gym_space, int):
            # Syntactic sugar for discrete space
            gym_space = Discrete(gym_space)
        elif isinstance(gym_space, tuple):
            # Syntactic sugar for discrete space
            gym_space = MultiDiscrete(gym_space)
        elif isinstance(gym_space, list):
            # Syntactic sugar for tuple space
            spaces = []
            for i, value in enumerate(gym_space):
                spaces[i] = self._create_gym_space(value)
            gym_space = Tuple(spaces)
        elif isinstance(gym_space, dict):
            # Syntactic sugar for dict space
            spaces = {}
            for key, value in gym_space.items():
                spaces[key] = self._create_gym_space(value)
            gym_space = Dict(spaces)
        return gym_space

    def _generate_task_names(self, gym_space):
        if isinstance(gym_space, Discrete):
            tasks = tuple(range(gym_space.n))
        elif isinstance(gym_space, MultiDiscrete):
            tasks = [tuple(range(dim)) for dim in gym_space.nvec]
        elif isinstance(gym_space, Tuple):
            tasks = [self._generate_task_names(value) for value in gym_space.spaces]
        elif isinstance(gym_space, Dict):
            tasks = {key: tuple(self._generate_task_names(value)) for key, value in gym_space.spaces.items()}
        else:
            tasks = None
        return tasks

    def _encode_box(self, task):
        if not self.gym_space.contains(task):
            raise UsageError(f"Task {task} is not in the task space")
        return task

    def _make_task_encoder(self, space, tasks):
        if isinstance(space, Discrete):
            assert space.n == len(
                tasks), f"Number of tasks ({space.n}) must match number of discrete options ({len(tasks)})"
            self._encode_map = {task: i for i, task in enumerate(tasks)}
            self._decode_map = {i: task for i, task in enumerate(tasks)}
            def encoder(task): return self._encode_map[task]
            def decoder(task): return self._decode_map[task]
        elif isinstance(space, Box):
            encoder = self._encode_box
            decoder = self._encode_box
        elif isinstance(space, Tuple):

            assert len(space.spaces) == len(
                tasks), f"Number of task ({len(space.spaces)}) must match options in Tuple ({len(tasks)})"
            results = [list(self._make_task_encoder(s, t)) for (s, t) in zip(space.spaces, tasks)]
            encoders = [r[0] for r in results]
            decoders = [r[1] for r in results]
            def encoder(task): return [e(t) for e, t in zip(encoders, task)]
            def decoder(task): return [d(t) for d, t in zip(decoders, task)]

        elif isinstance(space, MultiDiscrete):
            assert len(space.nvec) == len(
                tasks), f"Number of steps in a tasks ({len(space.nvec)}) must match number of discrete options ({len(tasks)})"

            encode_maps = []
            decode_maps = []
            for index, size in enumerate(space.nvec):
                # task_index = np.prod(space.nvec[:index]) - 1
                encode_maps.append({task: i for i, task in enumerate(tasks[index])})
                decode_maps.append({i: task for i, task in enumerate(tasks[index])})

            def encoder(task): return tuple([encode_maps[i][t] for i, t in enumerate(task)])
            def decoder(task): return tuple([decode_maps[i][t] for i, t in enumerate(task)])

        elif isinstance(space, Dict):

            def helper(task, spaces, tasks, action="encode"):
                # Iteratively encodes or decodes each space in the dictionary
                output = {}
                if (isinstance(spaces, dict) or isinstance(spaces, Dict)):
                    for key, value in spaces.items():
                        if (isinstance(value, dict) or isinstance(value, Dict)):
                            temp = helper(task[key], value, tasks[key], action)
                            output.update({key: temp})
                        else:
                            encoder, decoder = self._make_task_encoder(value, tasks[key])
                            output[key] = encoder(task[key]) if action == "encode" else decoder(task[key])
                return output

            def encoder(task): return helper(task, space.spaces, tasks, "encode")
            def decoder(task): return helper(task, space.spaces, tasks, "decode")
        else:
            def encoder(task): return task
            def decoder(task): return task
        return encoder, decoder

    def decode(self, encoding):
        """Convert the efficient task encoding to a task that can be used by the environment."""
        try:
            return self._decoder(encoding)
        except KeyError as e:
            # Check if task is already in decoded form
            try:
                self._encoder(encoding)
                warnings.warn(f"Task encoding already decoded: {encoding}")
                return encoding
            except KeyError:
                raise UsageError(f"Failed to decode task encoding: {encoding}") from e

    def encode(self, task):
        """Convert the task to an efficient encoding to speed up multiprocessing."""
        try:
            return self._encoder(task)
        except KeyError as e:
            try:
                self._decoder(task)
                warnings.warn(f"Task already encoded: {task}")
                return task
            except KeyError:
                raise UsageError(f"Failed to encode task: {task}") from e

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
        if isinstance(self.gym_space, MultiDiscrete):
            return list(range(len(self._task_list)))
        return self._task_list

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
        return task in self._task_set or self.decode(task) in self._task_set

    def increase_space(self, amount: Union[int, float] = 1):
        if isinstance(self.gym_space, Discrete):
            assert isinstance(
                amount, int), f"Discrete task space can only be increased by integer amount. Got {amount} instead."
            return Discrete(self.gym_space.n + amount)

    def sample(self):
        sample = self.gym_space.sample()
        return self.decode(sample)

    def list_tasks(self):
        return self._task_list

    def box_contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not isinstance(x, np.ndarray):
            try:
                x = np.asarray(x, dtype=self.gym_space.dtype)
            except (ValueError, TypeError):
                return False

        return not bool(x.shape == self.gym_space.shape and np.any((x < self.gym_space.low) | (x > self.gym_space.high)))


class FlatTaskSpace():
    def __init__(self, gym_space: Union[Space, int], tasks=None):

        if not isinstance(gym_space, Space):
            gym_space = self._create_gym_space(gym_space)

        if isinstance(gym_space, tuple) and all([isinstance(x, int) for x in gym_space]):
            # Syntactic sugar for multidiscrete space
            gym_space = MultiDiscrete(gym_space)

        # Expand Multidiscrete tasks
        task_list = tasks
        if isinstance(gym_space, (MultiDiscrete, Tuple)) and tasks is not None:
            # TODO: Add some check that task names are correct shape for gym space
            task_list = list(itertools.product(*tasks))

        self.gym_space = gym_space

        # Autogenerate task names
        if tasks is None:
            tasks = self._generate_task_names(gym_space)

        self._task_set = set(task_list) if task_list is not None else None
        self._task_list = task_list
        self._encoder, self._decoder = self._make_task_encoder(gym_space, tasks)
        sample = self.sample()
        print(sample)
        self.task_shape = np.array(self.encode(sample)).shape

    def _create_gym_space(self, gym_space):
        """Syntactic sugar for creating gym spaces."""
        if isinstance(gym_space, int):
            gym_space = Discrete(gym_space)
        elif isinstance(gym_space, tuple):
            gym_space = MultiDiscrete(gym_space)
        elif isinstance(gym_space, list):
            spaces = []
            for i, value in enumerate(gym_space):
                spaces[i] = self._create_gym_space(value)
            gym_space = Tuple(spaces)
        elif isinstance(gym_space, dict):
            spaces = {}
            for key, value in gym_space.items():
                spaces[key] = self._create_gym_space(value)
            gym_space = Dict(spaces)
        return gym_space

    def _generate_task_names(self, gym_space):
        if isinstance(gym_space, Discrete):
            tasks = tuple(range(gym_space.n))
        elif isinstance(gym_space, MultiDiscrete):
            tasks = [tuple(range(dim)) for dim in gym_space.nvec]
        elif isinstance(gym_space, Tuple):
            tasks = [self._generate_task_names(value) for value in gym_space.spaces]
        elif isinstance(gym_space, Dict):
            tasks = {key: tuple(self._generate_task_names(value)) for key, value in gym_space.spaces.items()}
        else:
            tasks = None
        return tasks

    def _is_sequential(self, tasks):
        """Check if the tasks are sequential integers."""
        return isinstance(tasks[0], int) and tuple(tasks) == tuple(range(tasks[0], tasks[-1] + 1))

    def _make_task_encoder(self, space, tasks):
        if isinstance(space, Discrete):
            assert space.n == len(
                tasks), f"Number of tasks ({space.n}) must match number of discrete options ({len(tasks)})"
            if self._is_sequential(tasks):

                # Use less memory for sequential tasks
                def encoder(task):
                    if task < tasks[0] or task > tasks[-1]:
                        raise UsageError(f"Task {task} is not in the task space")
                    return task - tasks[0]

                def decoder(task):
                    if task < 0 or task >= len(tasks):
                        raise UsageError(f"Encoding {task} does not map to a task in the task space")
                    return task + tasks[0]
            else:
                # Map each discrete task to an integer
                self._decode_map = {i: task for i, task in enumerate(tasks)}

                def encoder(task):
                    return self._encode_map[task]

                def decoder(task):
                    return self._decode_map[task]
        elif isinstance(space, Tuple):
            assert len(space.spaces) == len(
                tasks), f"Number of task ({len(space.spaces)}) must match options in Tuple ({len(tasks)})"
            results = [list(self._make_task_encoder(s, t)) for (s, t) in zip(space.spaces, tasks)]
            encoders = [r[0] for r in results]
            decoders = [r[1] for r in results]
            def encoder(task): return [e(t) for e, t in zip(encoders, task)]
            def decoder(task): return [d(t) for d, t in zip(decoders, task)]

        elif isinstance(space, MultiDiscrete):
            task_lens = [len(t) for t in tasks]
            assert np.prod(space.nvec) == np.prod(
                task_lens), f"Number of steps in a tasks ({np.prod(space.nvec)}) must match number of discrete options ({np.prod(task_lens)})"

            combinations = [p for p in itertools.product(*tasks)]
            encode_map = {task: i for i, task in enumerate(combinations)}
            decode_map = {i: task for i, task in enumerate(combinations)}
            def encoder(task): return encode_map[tuple(task)]
            def decoder(task): return decode_map[tuple(task)]

        elif isinstance(space, Dict):

            def helper(task, spaces, tasks, action="encode"):
                # Iteratively encodes or decodes each space in the dictionary
                output = {}
                if (isinstance(spaces, dict) or isinstance(spaces, Dict)):
                    for key, value in spaces.items():
                        if (isinstance(value, dict) or isinstance(value, Dict)):
                            temp = helper(task[key], value, tasks[key], action)
                            output.update({key: temp})
                        else:
                            encoder, decoder = self._make_task_encoder(value, tasks[key])
                            output[key] = encoder(task[key]) if action == "encode" else decoder(task[key])
                return output

            def encoder(task): return helper(task, space.spaces, tasks, "encode")
            def decoder(task): return helper(task, space.spaces, tasks, "decode")
        else:
            def encoder(task): return task
            def decoder(task): return task
        return encoder, decoder

    def decode(self, encoding):
        """Convert the efficient task encoding to a task that can be used by the environment."""
        try:
            return self._decoder(encoding)
        except KeyError as e:
            # Check if task is already in decoded form
            try:
                self._encoder(encoding)
                warnings.warn(f"Task encoding already decoded: {encoding}")
                return encoding
            except KeyError:
                raise UsageError(f"Failed to decode task encoding: {encoding}") from e

    def encode(self, task):
        """Convert the task to an efficient encoding to speed up multiprocessing."""
        try:
            return self._encoder(task)
        except KeyError as e:
            try:
                self._decoder(task)
                warnings.warn(f"Task already encoded: {task}")
                return task
            except KeyError:
                raise UsageError(f"Failed to encode task: {task}") from e

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
        if isinstance(self.gym_space, MultiDiscrete):
            return list(range(len(self._task_list)))
        return self._task_list

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
            return FlatTaskSpace._sum_axes(gym_space.nvec)
        elif isinstance(gym_space, MultiDiscrete):
            return FlatTaskSpace._sum_axes(gym_space.nvec)
        elif gym_space is None:
            return 0
        else:
            raise NotImplementedError(f"Unsupported task space type: {type(gym_space)}")

    def task_name(self, task):
        return repr(self.decode(task))

    def contains(self, task):
        return task in self._task_set or self.decode(task) in self._task_set

    def increase_space(self, amount: Union[int, float] = 1):
        if isinstance(self.gym_space, Discrete):
            assert isinstance(
                amount, int), f"Discrete task space can only be increased by integer amount. Got {amount} instead."
            return Discrete(self.gym_space.n + amount)

    def sample(self):
        sample = self.gym_space.sample()
        print(sample)
        return self.decode(sample)

    def list_tasks(self):
        return self._task_list

    def box_contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not isinstance(x, np.ndarray):
            try:
                x = np.asarray(x, dtype=self.gym_space.dtype)
            except (ValueError, TypeError):
                return False

        return not bool(x.shape == self.gym_space.shape and np.any((x < self.gym_space.low) | (x > self.gym_space.high)))


class BaseTaskSpace:
    def __init__(self, space_or_value, tasks):
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

    def contains(self, task):
        return task in self._task_set or self._decode(task) in self._task_set


class DiscreteTaskSpace(BaseTaskSpace):
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


class MultiDiscreteTaskSpace(BaseTaskSpace):
    def __init__(self, space_or_value: Union[Space, int], tasks=None, flatten=False):
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

    @property
    def tasks(self) -> List[Any]:
        return self._all_tasks

    @property
    def num_tasks(self) -> int:
        return int(np.prod(self.gym_space.nvec))

    def task_name(self, task):
        return repr(self._decode(task))


class TupleTaskSpace(BaseTaskSpace):
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

    def contains(self, task):
        for element, space in zip(task, self.task_spaces):
            if not space.contains(element):
                return False
        return True

    def sample(self):
        return [space.sample() for space in self.task_spaces]

    def seed(self, seed):
        for space in self.task_spaces:
            space.seed(seed)

    @property
    def tasks(self) -> List[Any]:
        if self._all_tasks is None:
            # Expand all task_spaces
            task_lists = [space.tasks for space in self.task_spaces]
            self._all_tasks = list(itertools.product(*task_lists))
        return self._all_tasks

    @property
    def num_tasks(self) -> int:
        return int(np.prod(self._task_nums))

    def task_name(self, task):
        # TODO: Add space_names here
        return repr(self._decode(task))
