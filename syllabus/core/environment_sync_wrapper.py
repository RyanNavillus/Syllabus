from multiprocessing import SimpleQueue, Lock
from typing import Any, Callable, Dict

import gymnasium as gym
import numpy as np
import ray
# from pettingzoo.utils.wrappers.base_parallel import BaseParallelWraper
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from syllabus.core import Curriculum, TaskEnv, TaskWrapper, MultiProcessingCurriculumWrapper  # , PettingZooTaskWrapper
from syllabus.task_space import TaskSpace
from multiprocessing.shared_memory import SharedMemory
from copy import copy


class MultiProcessingSyncWrapper(gym.Wrapper):
    """
    This wrapper is used to set the task on reset for a Gym environments running
    on parallel processes created using multiprocessing.Process. Meant to be used
    with a QueueLearningProgressCurriculum running on the main process.
    """
    instance_id = 0
    shared_mem = SharedMemory(size=1024, create=True)

    def __init__(self,
                 env,
                 components: MultiProcessingCurriculumWrapper.Components,
                 update_on_step: bool = True,   # TODO: Fine grained control over which step elements are used. Controlled by curriculum?
                 buffer_size: int = 1,
                 task_space: TaskSpace = None,
                 global_task_completion: Callable[[Curriculum, np.ndarray, float, bool, Dict[str, Any]], bool] = None):
        assert isinstance(task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        super().__init__(env)
        self.env = env
        self._latest_task = None
        self.task_queue = components.task_queue
        self.update_queue = components.update_queue
        self.task_space = task_space
        self.update_on_step = update_on_step
        self.global_task_completion = global_task_completion
        self.task_progress = 0.0
        self.step_updates = []
        self.warned_once = False
        self._first_episode = True
        components.instance_lock.acquire()
        self.instance_id = copy(MultiProcessingSyncWrapper.shared_mem.buf[0])
        MultiProcessingSyncWrapper.shared_mem.buf[0] += 1
        components.instance_lock.release()

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
        self._latest_task = next_task

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
                "env_id": self.instance_id,
                "request_sample": False
            })
            # Task progress
            self.step_updates.append({
                "update_type": "task_progress",
                "metrics": ((self.task_space.encode(self.env.task), self.task_progress)),
                "env_id": self.instance_id,
                "request_sample": term or trunc
            })
            # Send batched updates
            if len(self.step_updates) >= 100 or term or trunc:
                # Group updates into arrays
                updates = self._aggregate_step_updates()
                self.update_queue.put(updates)
                self.step_updates = []
        elif term or trunc:
            # Task progress
            update = {
                "update_type": "task_progress",
                "metrics": ((self.task_space.encode(self.env.task), self.task_progress)),
                "env_id": self.instance_id,
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

    def get_task(self):
        # Allow user to reject task
        if hasattr(self.env, "task"):
            return self.env.task
        return self._latest_task

    def __getattr__(self, attr):
        env_attr = getattr(self.env, attr, None)
        if env_attr is not None:
            return env_attr

    def _aggregate_step_updates(self):
        updates = []
        rews = []
        obs = []
        terms = []
        truncs = []
        infos = []
        for step_update in self.step_updates:
            if step_update["update_type"] == "step":
                ob, rew, term, trunc, info = step_update["metrics"]
                obs.append(ob)
                rews.append(rew)
                terms.append(term)
                truncs.append(trunc)
                infos.append(info)
            else:
                updates.append(step_update)
        step_batch = {
            "update_type": "step_batch",
            "metrics": ([np.array(obs), np.array(rews), np.array(terms), np.array(truncs), np.array(infos)],),
            "env_id": self.instance_id,
            "request_sample": False
        }
        updates.append(step_batch)
        return updates

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
