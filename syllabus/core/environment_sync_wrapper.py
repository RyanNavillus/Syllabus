from multiprocessing import SimpleQueue
from typing import Any, Callable, Dict

import gymnasium as gym
import numpy as np
import ray
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from syllabus.core import Curriculum
from syllabus.core.task_interface import TaskEnv, TaskWrapper, PettingZooTaskWrapper
from syllabus.task_space import TaskSpace


class MultiProcessingSyncWrapper(gym.Wrapper):
    """
    This wrapper is used to set the task on reset for a Gym environments running
    on parallel processes created using multiprocessing.Process. Meant to be used
    with a QueueLearningProgressCurriculum running on the main process.
    """
    def __init__(self,
                 env,
                 task_queue: SimpleQueue,
                 update_queue: SimpleQueue,
                 update_on_step: bool = True,   # TODO: Fine grained control over which step elements are used. Controlled by curriculum?
                 buffer_size: int = 1,
                 task_space: TaskSpace = None,
                 global_task_completion: Callable[[Curriculum, np.ndarray, float, bool, Dict[str, Any]], bool] = None):
        assert isinstance(task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        super().__init__(env)
        self.env = env
        self.task_queue = task_queue
        self.update_queue = update_queue
        self.task_space = task_space
        self.update_on_step = update_on_step
        self.global_task_completion = global_task_completion
        self.task_progress = 0.0
        self.step_updates = []
        self.warned_once = False
        self._first_episode = True

        # Request initial task
        for _ in range(buffer_size):
            update = {
                "update_type": "noop",
                "metrics": None,
                "request_sample": True,
            }
            self.update_queue.put(update)

    def reset(self, *args, **kwargs):
        self.step_updates = []
        self.task_progress = 0.0

        message = self.task_queue.get()     # Blocks until a task is available
        next_task = self.task_space.decode(message["next_task"])

        # Add any new tasks
        if "added_tasks" in message:
            added_tasks = message["added_tasks"]
            for add_task in added_tasks:
                self.env.add_task(add_task)
        return self.env.reset(*args, new_task=next_task, **kwargs)

    def step(self, action):
        obs, rew, term, trunc, info = step_api_compatibility(self.env.step(action), output_truncation_bool=True)
        if "task_completion" in info:
            if self.global_task_completion is not None:
                self.task_progress = self.global_task_completion(self.curriculum, obs, rew, term, trunc, info)
            else:
                self.task_progress = info["task_completion"]

        # Update curriculum with step info
        if self.update_on_step:
            # Environment outputs
            self.step_updates.append({
                "update_type": "step",
                "metrics": (obs, rew, term, trunc, info),
                "request_sample": False
            })
            # Task progress
            self.step_updates.append({
                "update_type": "task_progress",
                "metrics": ((self.task_space.encode(self.env.task), self.task_progress)),
                "request_sample": term or trunc
            })
            # Send batched updates
            if len(self.step_updates) >= 1000 or term or trunc:
                self.update_queue.put(self.step_updates)
                self.step_updates = []
        elif term or trunc:
            # Task progress
            update = {
                "update_type": "task_progress",
                "metrics": ((self.task_space.encode(self.env.task), self.task_progress)),
                "request_sample": True,
            }
            self.update_queue.put(update)

        return obs, rew, term, trunc, info

    def add_task(self, task):
        update = {
            "update_type": "add_task",
            "metrics": task
        }
        self.update_queue.put(update)

    def __getattr__(self, attr):
        env_attr = getattr(self.env, attr, None)
        if env_attr is not None:
            return env_attr


class PettingZooMultiProcessingSyncWrapper(BaseParallelWrapper):
    """
    This wrapper is used to set the task on reset for a Gym environments running
    on parallel processes created using multiprocessing.Process. Meant to be used
    with a QueueLearningProgressCurriculum running on the main process.
    """
    def __init__(self,
                 env,
                 task_queue: SimpleQueue,
                 update_queue: SimpleQueue,
                 update_on_step: bool = True,   # TODO: Fine grained control over which step elements are used. Controlled by curriculum?
                 buffer_size: int = 1,
                 task_space: TaskSpace = None,
                 global_task_completion: Callable[[Curriculum, np.ndarray, float, bool, Dict[str, Any]], bool] = None):
        assert isinstance(task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        super().__init__(env)
        self.env = env
        self.task_queue = task_queue
        self.update_queue = update_queue
        self.task_space = task_space
        self.update_on_step = update_on_step
        self.global_task_completion = global_task_completion
        self.task_progress = 0.0
        self.step_updates = []
        self.warned_once = False
        self._first_episode = True

        # Request initial task
        for _ in range(buffer_size):
            update = {
                "update_type": "noop",
                "metrics": None,
                "request_sample": True,
            }
            self.update_queue.put(update)

    def reset(self, *args, **kwargs):
        self.step_updates = []
        self.task_progress = 0.0
        message = self.task_queue.get()     # Blocks until a task is available
        next_task = self.task_space.decode(message["next_task"])
        # Add any new tasks
        if "added_tasks" in message:
            added_tasks = message["added_tasks"]
            for add_task in added_tasks:
                self.env.add_task(add_task)
        return self.env.reset(*args, new_task=next_task, **kwargs)

    def step(self, action):
        obs, rews, terms, truncs, infos = self.env.step(action)

        if "task_completion" in list(infos.values())[0]:
            self.task_progress = max([info["task_completion"] for info in infos.values()])

        is_finished = (len(self.env.agents) == 0) or all(terms.values())
        # Update curriculum with step info
        if self.update_on_step:
            # Environment outputs
            # TODO: Create a better system for aggregating step results in different ways. Maybe custom aggregation functions
            self.step_updates.append({
                "update_type": "step",
                "metrics": (obs, sum(rews.values()), all(terms.values()), all(truncs.values()), list(infos.values())[0]),
                "request_sample": False
            })
            # Task progress
            self.step_updates.append({
                "update_type": "task_progress",
                "metrics": ((self.task_space.encode(self.env.task), self.task_progress)),
                "request_sample": is_finished
            })
            # Send batched updates
            if len(self.step_updates) >= 1000 or is_finished:
                self.update_queue.put(self.step_updates)
                self.step_updates = []
        elif is_finished:
            # Task progress
            update = {
                "update_type": "task_progress",
                "metrics": ((self.task_space.encode(self.env.task), self.task_progress)),
                "request_sample": True,
            }
            self.update_queue.put(update)
        return obs, rews, terms, truncs, infos

    def add_task(self, task):
        update = {
            "update_type": "add_task",
            "metrics": task
        }
        self.update_queue.put(update)

    def __getattr__(self, attr):
        env_attr = getattr(self.env, attr, None)
        if env_attr is not None:
            return env_attr


class RaySyncWrapper(gym.Wrapper):
    """
    This wrapper is used to set the task on reset for a Gym environments running
    on parallel processes created using ray. Meant to be used with a
    RayLearningProgressCurriculum running on the main process.
    """
    def __init__(self,
                 env,
                 update_on_step: bool = True,
                 task_space: gym.Space = None,
                 global_task_completion: Callable[[Curriculum, np.ndarray, float, bool, Dict[str, Any]], bool] = None):
        assert isinstance(env, TaskWrapper) or isinstance(env, TaskEnv) or isinstance(env, PettingZooTaskWrapper), "Env must implement the task API"
        super().__init__(env)
        self.env = env
        self.update_on_step = update_on_step    # Disable to improve performance
        self.task_space = task_space
        self.curriculum = ray.get_actor("curriculum")
        self.task_completion = 0.0
        self.global_task_completion = global_task_completion
        self.step_results = []

    def reset(self, *args, **kwargs):
        self.step_results = []

        # Update curriculum
        update = {
            "update_type": "task_progress",
            "metrics": (self.env.task, self.task_completion),
            "request_sample": True
        }
        self.curriculum.update.remote(update)
        self.task_completion = 0.0

        # Sample new task
        sample = ray.get(self.curriculum.sample.remote())
        next_task = sample[0]

        return self.env.reset(*args, new_task=next_task, **kwargs)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)

        if "task_completion" in info:
            if self.global_task_completion is not None:
                # TODO: Hide rllib interface?
                self.task_completion = self.global_task_completion(self.curriculum, obs, rew, term, trunc, info)
            else:
                self.task_completion = info["task_completion"]

        # TODO: Optimize
        if self.update_on_step:
            self.step_results.append((obs, rew, term, trunc, info))
            if len(self.step_results) >= 1000 or term or trunc:
                update = {
                    "update_type": "step_batch",
                    "metrics": (self.step_results,),
                    "request_sample": False
                }
                self.curriculum.update.remote(update)
                self.step_results = []

        return obs, rew, term, trunc, info

    def change_task(self, new_task):
        """
        Changes the task of the existing environment to the new_task.

        Each environment will implement tasks differently. The easiest system would be to call a
        function or set an instance variable to change the task.

        Some environments may need to be reset or even reinitialized to change the task.
        If you need to reset or re-init the environment here, make sure to check
        that it is not in the middle of an episode to avoid unexpected behavior.
        """
        self.env.change_task(new_task)

    def add_task(self, task):
        self.curriculum.add_task.remote(task)

    def __getattr__(self, attr):
        env_attr = getattr(self.env, attr, None)
        if env_attr:
            return env_attr


class PettingZooRaySyncWrapper(BaseParallelWrapper):
    """
    This wrapper is used to set the task on reset for a Gym environments running
    on parallel processes created using ray. Meant to be used with a
    RayLearningProgressCurriculum running on the main process.
    """
    def __init__(self,
                 env,
                 update_on_step: bool = True,
                 task_space: gym.Space = None,
                 global_task_completion: Callable[[Curriculum, np.ndarray, float, bool, Dict[str, Any]], bool] = None):
        assert isinstance(env, TaskWrapper) or isinstance(env, TaskEnv) or isinstance(env, PettingZooTaskWrapper), "Env must implement the task API"
        super().__init__(env)
        self.env = env
        self.update_on_step = update_on_step    # Disable to improve performance
        self.task_space = task_space
        self.curriculum = ray.get_actor("curriculum")
        self.task_completion = 0.0
        self.global_task_completion = global_task_completion
        self.step_results = []

    def reset(self, *args, **kwargs):
        self.step_results = []

        # Update curriculum
        update = {
            "update_type": "task_progress",
            "metrics": (self.env.task, self.task_completion),
            "request_sample": True
        }
        self.curriculum.update.remote(update)
        self.task_completion = 0.0

        # Sample new task
        sample = ray.get(self.curriculum.sample.remote())
        next_task = sample[0]

        return self.env.reset(*args, new_task=next_task, **kwargs)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)

        if "task_completion" in info:
            if self.global_task_completion is not None:
                # TODO: Hide rllib interface?
                self.task_completion = self.global_task_completion(self.curriculum, obs, rew, term, trunc, info)
            else:
                self.task_completion = info["task_completion"]

        # TODO: Optimize
        if self.update_on_step:
            self.step_results.append((obs, rew, term, trunc, info))
            if len(self.step_results) >= 1000 or term or trunc:
                update = {
                    "update_type": "step_batch",
                    "metrics": (self.step_results,),
                    "request_sample": False
                }
                self.curriculum.update.remote(update)
                self.step_results = []

        return obs, rew, term, trunc, info

    def change_task(self, new_task):
        """
        Changes the task of the existing environment to the new_task.

        Each environment will implement tasks differently. The easiest system would be to call a
        function or set an instance variable to change the task.

        Some environments may need to be reset or even reinitialized to change the task.
        If you need to reset or re-init the environment here, make sure to check
        that it is not in the middle of an episode to avoid unexpected behavior.
        """
        self.env.change_task(new_task)

    def add_task(self, task):
        self.curriculum.add_task.remote(task)

    def __getattr__(self, attr):
        env_attr = getattr(self.env, attr, None)
        if env_attr:
            return env_attr
