from multiprocessing import SimpleQueue
from typing import Any, Callable, Dict

import gym
import numpy as np
import ray
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWraper
from syllabus.core import (Curriculum, PettingZooTaskWrapper, TaskEnv,
                           TaskWrapper)
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
        obs, rew, done, info = self.env.step(action)
        if "task_completion" in info:
            if self.global_task_completion is not None:
                self.task_progress = self.global_task_completion(self.curriculum, obs, rew, done, info)
            else:
                self.task_progress = info["task_completion"]

        # Update curriculum with step info
        if self.update_on_step:
            # Environment outputs
            self.step_updates.append({
                "update_type": "step",
                "metrics": (obs, rew, done, info),
                "request_sample": False
            })
            # Task progress
            self.step_updates.append({
                "update_type": "task_progress",
                "metrics": ((self.task_space.encode(self.env.task), self.task_progress)),
                "request_sample": done
            })
            # Send batched updates
            if len(self.step_updates) >= 1000 or done:
                self.update_queue.put(self.step_updates)
                self.step_updates = []
        elif done:
            # Task progress
            update = {
                "update_type": "task_progress",
                "metrics": ((self.task_space.encode(self.env.task), self.task_progress)),
                "request_sample": True,
            }
            self.update_queue.put(update)

        return obs, rew, done, info

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


# TODO: Fix this and refactor
class PettingZooMultiProcessingSyncWrapper(BaseParallelWraper):
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
                 default_task=None,
                 task_space: TaskSpace = None,
                 global_task_completion: Callable[[Curriculum, np.ndarray, float, bool, Dict[str, Any]], bool] = None):
        super().__init__(env)
        self.env = env
        self.task_queue = task_queue
        self.update_queue = update_queue
        self.task_space = task_space
        self.update_on_step = update_on_step
        self.global_task_completion = global_task_completion
        self.task_completion = 0.0
        self.warned_once = False
        self.step_results = []
        if task_space.contains(default_task):
            self.default_task = default_task

        # Request initial task
        update = {
            "update_type": "noop",
            "metrics": None,
            "request_sample": True
        }
        self.update_queue.put(update)

    @property
    def agents(self):
        return self.env.agents

    def reset(self, *args, **kwargs):
        self.step_results = []

        # Update curriculum
        update = {
            "update_type": "complete",
            "metrics": (self.task_space.encode(self.env.task), self.task_completion),
            "request_sample": True
        }
        self.update_queue.put(update)
        self.task_completion = 0.0

        # Sample new task
        if self.task_queue.empty():
            # Choose default task if it is set, or keep the current task
            next_task = self.default_task if self.default_task is not None else self.task_space.sample()
            if not self.warned_once:
                print("\nTask queue was empty, selecting default task. This warning will not print again for this environment.\n")
                self.warned_once = False
        else:
            message = self.task_queue.get()
            next_task = self.task_space.decode(message["next_task"])
            if "add_task" in message:
                self.env.add_task(message["add_task"])
        return self.env.reset(*args, new_task=next_task, **kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if "task_completion" in info:
            if self.global_task_completion is not None:
                self.task_completion = self.global_task_completion(self.curriculum, obs, rew, done, info)
            else:
                self.task_completion = info["task_completion"]

        if self.update_on_step:
            self.step_results.append((obs, rew, done, info))
            if len(self.step_results) >= 2000:
                update = {
                    "update_type": "step_batch",
                    "metrics": (self.step_results,),
                    "request_sample": False
                }
                self.update_queue.put(update)
                self.step_results = []

        return obs, rew, done, info

    def add_task(self, task):
        update = {
            "update_type": "add_task",
            "metrics": task
        }
        self.update_queue.put(update)
    
    def __getattr__(self, attr):
        env_attr = getattr(self.env, attr, None)
        if env_attr:
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
        self.update_on_step = update_on_step    # Disable to improve performance 10x
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
        obs, rew, done, info = self.env.step(action)

        if "task_completion" in info:
            if self.global_task_completion is not None:
                # TODO: Hide rllib interface?
                self.task_completion = self.global_task_completion(self.curriculum, obs, rew, done, info)
            else:
                self.task_completion = info["task_completion"]

        # TODO: Optimize
        if self.update_on_step:
            self.step_results.append((obs, rew, done, info))
            if len(self.step_results) >= 1000 or done:
                update = {
                    "update_type": "step_batch",
                    "metrics": (self.step_results,),
                    "request_sample": False
                }
                self.curriculum.update.remote(update)
                self.step_results = []

        return obs, rew, done, info

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
