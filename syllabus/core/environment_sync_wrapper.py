from typing import Any, Callable, Dict

import gymnasium as gym
import numpy as np
import ray
from gymnasium.utils.step_api_compatibility import step_api_compatibility

from syllabus.core import Curriculum, MultiProcessingCurriculumWrapper, MultiProcessingComponents, TaskEnv, TaskWrapper
from syllabus.task_space import TaskSpace


class MultiProcessingSyncWrapper(gym.Wrapper):
    """
    This wrapper is used to set the task on reset for a Gym environments running
    on parallel processes created using multiprocessing.Process. Meant to be used
    with a QueueLearningProgressCurriculum running on the main process.
    """

    def __init__(self,
                 env,
                 components: MultiProcessingComponents,
                 update_on_step: bool = False,   # TODO: Fine grained control over which step elements are used. Controlled by curriculum?
                 update_on_progress: bool = False,   # TODO: Fine grained control over which step elements are used. Controlled by curriculum?
                 batch_size: int = 100,
                 buffer_size: int = 2,  # Having an extra task in the buffer minimizes wait time at reset
                 task_space: TaskSpace = None,
                 global_task_completion: Callable[[Curriculum, np.ndarray, float, bool, Dict[str, Any]], bool] = None):
        # TODO: reimplement global task progress metrics
        assert isinstance(task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        super().__init__(env)
        self.env = env
        self.components = components
        self._latest_task = None
        self.task_queue = components.task_queue
        self.update_queue = components.update_queue
        self.task_space = task_space
        self.update_on_step = update_on_step
        self.update_on_progress = update_on_progress
        self.batch_size = batch_size
        self.global_task_completion = global_task_completion
        self.task_progress = 0.0
        self._batch_step = 0
        self.instance_id = components.get_id()

        self.episode_length = 0
        self.episode_return = 0

        # Create batch buffers for step updates
        if self.update_on_step:
            self._obs = [None] * self.batch_size
            self._rews = np.zeros(self.batch_size, dtype=np.float32)
            self._terms = np.zeros(self.batch_size, dtype=bool)
            self._truncs = np.zeros(self.batch_size, dtype=bool)
            self._infos = [None] * self.batch_size
            self._tasks = [None] * self.batch_size
            self._task_progresses = [None] * self.batch_size

        # Request initial task
        assert buffer_size > 0, "Buffer size must be greater than 0 to sample initial task for envs."
        for _ in range(buffer_size):
            update = {
                "update_type": "noop",
                "metrics": None,
                "request_sample": True,
            }
            self.components.put_update(update)

    def reset(self, *args, **kwargs):
        self.step_updates = []
        self.task_progress = 0.0
        self.episode_length = 0
        self.episode_return = 0

        message = self.components.get_task()    # Blocks until a task is available
        next_task = self.task_space.decode(message["next_task"])
        self._latest_task = next_task

        # Add any new tasks
        if "added_tasks" in message:
            added_tasks = message["added_tasks"]
            for add_task in added_tasks:
                self.env.add_task(add_task)
        obs, info = self.env.reset(*args, new_task=next_task, **kwargs)
        info["task"] = self.task_space.encode(self.get_task())
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = step_api_compatibility(self.env.step(action), output_truncation_bool=True)
        self.episode_length += 1
        self.episode_return += rew
        self.task_progress = info.get("task_completion", 0.0)

        # Update curriculum with step info
        if self.update_on_step:
            self._obs[self._batch_step] = obs
            self._rews[self._batch_step] = rew
            self._terms[self._batch_step] = term
            self._truncs[self._batch_step] = trunc
            self._infos[self._batch_step] = info
            self._tasks[self._batch_step] = self.task_space.encode(self.get_task())
            self._task_progresses[self._batch_step] = self.task_progress
            self._batch_step += 1

            # Send batched updates
            if self._batch_step >= self.batch_size or term or trunc:
                updates = self._package_step_updates()
                self.components.put_update(updates)
                self._batch_step = 0

        # Episode update
        if term or trunc:
            # Task progress
            task_update = {
                "update_type": "task_progress",
                "metrics": ((self.task_space.encode(self.env.task), self.task_progress)),
                "env_id": self.instance_id,
                "request_sample": False,
            }
            episode_update = {
                "update_type": "episode",
                "metrics": (self.episode_return, self.episode_length, self.task_space.encode(self.env.task)),
                "env_id": self.instance_id,
                "request_sample": True
            }
            self.components.put_update([task_update, episode_update])

        info["task"] = self.task_space.encode(self.get_task())

        return obs, rew, term, trunc, info

    def _package_step_updates(self):
        step_batch = {
            "update_type": "step_batch",
            "metrics": ([self._tasks[:self._batch_step], self._obs[:self._batch_step], self._rews[:self._batch_step], self._terms[:self._batch_step], self._truncs[:self._batch_step], self._infos[:self._batch_step]],),
            "env_id": self.instance_id,
            "request_sample": False
        }
        update = [step_batch]

        if self.update_on_progress:
            task_batch = {
                "update_type": "task_progress_batch",
                "metrics": (self._tasks[:self._batch_step], self._task_progresses[:self._batch_step],),
                "env_id": self.instance_id,
                "request_sample": False
            }
            update.append(task_batch)
        return update

    def add_task(self, task):
        update = {
            "update_type": "add_task",
            "metrics": task
        }
        self.update_queue.put(update)

    def get_task(self):
        # Allow user to reject task
        if hasattr(self.env, "task"):
            return self.env.task
        return self._latest_task

    def __getattr__(self, attr):
        env_attr = getattr(self.env, attr, None)
        if env_attr is not None:
            return env_attr


# TODO: Fix this and refactor
# class PettingZooMultiProcessingSyncWrapper(BaseParallelWraper):
#     """
#     This wrapper is used to set the task on reset for a Gym environments running
#     on parallel processes created using multiprocessing.Process. Meant to be used
#     with a QueueLearningProgressCurriculum running on the main process.
#     """
#     def __init__(self,
#                  env,
#                  task_queue: SimpleQueue,
#                  update_queue: SimpleQueue,
#                  update_on_step: bool = True,   # TODO: Fine grained control over which step elements are used. Controlled by curriculum?
#                  default_task=None,
#                  task_space: TaskSpace = None,
#                  global_task_completion: Callable[[Curriculum, np.ndarray, float, bool, Dict[str, Any]], bool] = None):
#         super().__init__(env)
#         self.env = env
#         self.task_queue = task_queue
#         self.update_queue = update_queue
#         self.task_space = task_space
#         self.update_on_step = update_on_step
#         self.global_task_completion = global_task_completion
#         self.task_completion = 0.0
#         self.warned_once = False
#         self.step_results = []
#         if task_space.contains(default_task):
#             self.default_task = default_task

#         # Request initial task
#         update = {
#             "update_type": "noop",
#             "metrics": None,
#             "request_sample": True
#         }
#         self.update_queue.put(update)

#     @property
#     def agents(self):
#         return self.env.agents

#     def reset(self, *args, **kwargs):
#         self.step_results = []

#         # Update curriculum
#         update = {
#             "update_type": "complete",
#             "metrics": (self.task_space.encode(self.env.task), self.task_completion),
#             "request_sample": True
#         }
#         self.update_queue.put(update)
#         self.task_completion = 0.0

#         # Sample new task
#         if self.task_queue.empty():
#             # Choose default task if it is set, or keep the current task
#             next_task = self.default_task if self.default_task is not None else self.task_space.sample()
#             if not self.warned_once:
#                 print("\nTask queue was empty, selecting default task. This warning will not print again for this environment.\n")
#                 self.warned_once = False
#         else:
#             message = self.task_queue.get()
#             next_task = self.task_space.decode(message["next_task"])
#             if "add_task" in message:
#                 self.env.add_task(message["add_task"])
#         return self.env.reset(*args, new_task=next_task, **kwargs)

#     def step(self, action):
#         obs, rew, term, trunc, info = self.env.step(action)

#         if "task_completion" in info:
#             if self.global_task_completion is not None:
#                 self.task_completion = self.global_task_completion(self.curriculum, obs, rew, term, trunc, info)
#             else:
#                 self.task_completion = info["task_completion"]

#         if self.update_on_step:
#             self.step_results.append((obs, rew, term, trunc, info))
#             if len(self.step_results) >= 2000:
#                 update = {
#                     "update_type": "step_batch",
#                     "metrics": (self.step_results,),
#                     "request_sample": False
#                 }
#                 self.update_queue.put(update)
#                 self.step_results = []

#         return obs, rew, term, trunc, info

#     def add_task(self, task):
#         update = {
#             "update_type": "add_task",
#             "metrics": task
#         }
#         self.update_queue.put(update)

#     def __getattr__(self, attr):
#         env_attr = getattr(self.env, attr, None)
#         if env_attr:
#             return env_attr


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
