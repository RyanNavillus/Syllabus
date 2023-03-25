import time
from typing import Any, Callable, Dict
from multiprocessing import SimpleQueue
import numpy as np

import gym
import ray
from syllabus.core import Curriculum, TaskWrapper


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
                 default_task=None,
                 task_space: gym.Space = None,
                 global_task_completion: Callable[[Curriculum, np.ndarray, float, bool, Dict[str, Any]], bool] = None):
        assert isinstance(env, TaskWrapper), "Env must implement the task API"
        super().__init__(env)
        self.env = env
        self.task_queue = task_queue
        self.update_queue = update_queue
        self.task_space = task_space
        self.update_on_step = update_on_step
        self.global_task_completion = global_task_completion
        self.task_completion = 0.0
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

    def reset(self, *args, **kwargs):
        self.step_results = []

        # Update curriculum
        update = {
            "update_type": "complete",
            "metrics": (self.env.task, self.task_completion),
            "request_sample": True
        }
        self.update_queue.put(update)
        self.task_completion = 0.0

        # Sample new task
        if self.task_queue.empty():
            # Choose default task if it is set, or keep the current task
            next_task = self.default_task if self.default_task else self.env.task
        else:
            next_task = self.task_queue.get()
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
            if len(self.step_results) >= 1000 or done:
                update = {
                    "update_type": "step_batch",
                    "metrics": (self.step_results,),
                    "request_sample": False
                }
                self.update_queue.put(update)
                self.step_results = []

        return obs, rew, done, info


class RaySyncWrapper(gym.Wrapper):
    """
    This wrapper is used to set the task on reset for a Gym environments running
    on parallel processes created using ray. Meant to be used with a
    RayLearningProgressCurriculum running on the main process.
    """
    def __init__(self,
                 env,
                 update_on_step: bool = True,
                 default_task=None,
                 task_space: gym.Space = None,
                 global_task_completion: Callable[[Curriculum, np.ndarray, float, bool, Dict[str, Any]], bool] = None):
        assert isinstance(env, TaskWrapper), "Env must implement the task API"
        super().__init__(env)
        self.env = env
        self.update_on_step = update_on_step    # Disable to improve performance 10x
        self.task_space = task_space
        if task_space.contains(default_task):
            self.default_task = default_task
        self.curriculum = ray.get_actor("curriculum")
        self.task_completion = 0.0
        self.global_task_completion = global_task_completion
        self.step_results = []

    def reset(self, *args, **kwargs):
        self.step_results = []

        # Update curriculum
        update = {
            "update_type": "complete",
            "metrics": (self.env.task, self.task_completion),
            "request_sample": True
        }
        self.curriculum.update_curriculum.remote(update)
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
                self.curriculum.update_curriculum.remote(update)
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
