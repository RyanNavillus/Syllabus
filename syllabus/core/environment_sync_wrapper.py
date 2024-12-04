import copy
import time
import torch
from typing import Any, Callable, Dict

import gymnasium as gym
import numpy as np
import ray
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper

from syllabus.core import Curriculum, MultiProcessingComponents
from syllabus.core.task_interface import PettingZooTaskWrapper, TaskEnv, TaskWrapper
from syllabus.task_space import TaskSpace


class GymnasiumSyncWrapper(gym.Wrapper):
    """
    This wrapper is used to set the task on reset for a Gym environments running
    on parallel processes created using multiprocessing.Process. Meant to be used
    with a QueueLearningProgressCurriculum running on the main process.
    """

    def __init__(self,
                 env,
                 task_space: TaskSpace,
                 components: MultiProcessingComponents,
                 batch_size: int = 100,
                 buffer_size: int = 2,  # Having an extra task in the buffer minimizes wait time at reset
                 remove_keys: list = None,
                 change_task_on_completion: bool = False,
                 global_task_completion: Callable[[Curriculum, np.ndarray, float, bool, Dict[str, Any]], bool] = None):
        # TODO: reimplement global task progress metrics
        assert isinstance(
            task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        super().__init__(env)
        self.env = env
        self.task_space = task_space
        self.components = components
        self._latest_task = None
        self.batch_size = batch_size
        self.remove_keys = remove_keys if remove_keys is not None else []
        self.change_task_on_completion = change_task_on_completion
        self.global_task_completion = global_task_completion
        self.task_progress = 0.0
        self._batch_step = 0
        self.instance_id = components.get_id()
        self.update_on_step = components.requires_step_updates and components.should_sync(self.instance_id)

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
            self._task_progresses = np.zeros(self.batch_size, dtype=np.float32)

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
        self.task_progress = 0.0
        self.episode_length = 0
        self.episode_return = 0

        message = self.components.get_task()    # Blocks until a task is available
        next_task = self.task_space.decode(message["next_task"])
        self._latest_task = next_task

        obs, info = self.env.reset(*args, new_task=next_task, **kwargs)
        info["task"] = self.task_space.encode(self.get_task())
        if self.update_on_step:
            self._update_step(obs, 0.0, False, False, info, send=False)
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = step_api_compatibility(self.env.step(action), output_truncation_bool=True)
        info["task"] = self.task_space.encode(self.get_task())
        self.episode_length += 1
        self.episode_return += rew
        self.task_progress = info.get("task_completion", 0.0)

        # Update curriculum with step info
        if self.update_on_step:
            self._update_step(obs, rew, term, trunc, info)

        # Episode update
        if term or trunc:
            episode_update = {
                "update_type": "episode",
                "metrics": (self.episode_return, self.episode_length, self.task_space.encode(self.get_task()), self.task_progress),
                "env_id": self.instance_id,
                "request_sample": True
            }
            self.components.put_update([episode_update])

        if self.change_task_on_completion and self.task_progress >= 1.0:
            update = {
                "update_type": "task_progress",
                "metrics": (self.task_space.encode(self.get_task()), self.task_progress),
                "env_id": self.instance_id,
                "request_sample": True
            }

            self.components.put_update(update)
            message = self.components.get_task()    # Blocks until a task is available
            next_task = self.task_space.decode(message["next_task"])
            self.env.change_task(next_task)
            self._latest_task = next_task

        return obs, rew, term, trunc, info

    def _update_step(self, obs, rew, term, trunc, info, send=True):
        trimmed_obs = {key: obs[key]
                       for key in obs.keys() if key not in self.remove_keys} if isinstance(obs, dict) else obs
        self._obs[self._batch_step] = trimmed_obs
        self._rews[self._batch_step] = rew
        self._terms[self._batch_step] = term
        self._truncs[self._batch_step] = trunc
        self._infos[self._batch_step] = info
        self._tasks[self._batch_step] = self.task_space.encode(self.get_task())
        self._task_progresses[self._batch_step] = self.task_progress
        self._batch_step += 1

        # Send batched updates
        if send and (self._batch_step >= self.batch_size or term or trunc):
            updates = self._package_step_updates()
            self.components.put_update(updates)
            self._batch_step = 0

    def _package_step_updates(self):
        return [{
            "update_type": "step_batch",
            "metrics": ([
                self._tasks[:self._batch_step],
                self._obs[:self._batch_step],
                self._rews[:self._batch_step],
                self._terms[:self._batch_step],
                self._truncs[:self._batch_step],
                self._infos[:self._batch_step],
                self._task_progresses[:self._batch_step],
            ],),
            "env_id": self.instance_id,
            "request_sample": False
        }]

    def get_task(self):
        # Allow user to reject task
        if hasattr(self.env, "task"):
            return self.env.task
        return self._latest_task

    def __getattr__(self, attr):
        env_attr = getattr(self.env, attr, None)
        if env_attr is not None:
            return env_attr


class PettingZooSyncWrapper(BaseParallelWrapper):
    """
    This wrapper is used to set the task on reset for a Gym environments running
    on parallel processes created using multiprocessing.Process. Meant to be used
    with a QueueLearningProgressCurriculum running on the main process.
    """

    def __init__(self,
                 env,
                 task_space: TaskSpace,
                 components: MultiProcessingComponents,
                 batch_size: int = 100,
                 buffer_size: int = 2,  # Having an extra task in the buffer minimizes wait time at reset
                 remove_keys: list = None,
                 change_task_on_completion: bool = False,
                 global_task_completion: Callable[[Curriculum, np.ndarray, float, bool, Dict[str, Any]], bool] = None):
        # TODO: reimplement global task progress metrics
        assert isinstance(
            task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        super().__init__(env)
        self.env = env
        self.task_space = task_space
        self.components = components
        self._latest_task = None
        self.batch_size = batch_size
        self.remove_keys = remove_keys if remove_keys is not None else []
        self.change_task_on_completion = change_task_on_completion
        self.global_task_completion = global_task_completion
        self._batch_step = 0
        self.instance_id = components.get_id()
        self.update_on_step = components.requires_step_updates and components.should_sync(self.instance_id)

        self.task_progress = 0.0
        self.episode_length = 0
        self.episode_returns = {agent: 0 for agent in self.env.possible_agents}

        # Create template values for reset step update
        _template_rews = {agent: 0 for agent in self.env.possible_agents}
        _template_terms = {agent: False for agent in self.env.possible_agents}
        _template_truncs = {agent: False for agent in self.env.possible_agents}
        self._template_args = (_template_rews, _template_terms, _template_truncs)

        # Create batch buffers for step updates
        if self.update_on_step:
            num_agents = len(self.env.possible_agents)
            self.agent_map = {agent: i for i, agent in enumerate(self.env.possible_agents)}
            self._obs = [[None for _ in range(num_agents)]] * self.batch_size
            self._rews = np.zeros((self.batch_size, num_agents), dtype=np.float32)
            self._terms = np.zeros((self.batch_size, num_agents), dtype=bool)
            self._truncs = np.zeros((self.batch_size, num_agents), dtype=bool)
            self._infos = [[None for _ in range(num_agents)]] * self.batch_size
            self._tasks = [[None for _ in range(num_agents)]] * self.batch_size
            self._task_progresses = np.zeros((self.batch_size, num_agents), dtype=np.float32)

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
        self.task_progress = 0.0
        self.episode_length = 0
        self.episode_returns = {agent: 0 for agent in self.env.possible_agents}

        message = self.components.get_task()    # Blocks until a task is available
        next_task = self.task_space.decode(message["next_task"])
        self._latest_task = next_task

        obs, info = self.env.reset(*args, new_task=next_task, **kwargs)
        info["task"] = self.task_space.encode(self.get_task())
        if self.update_on_step:
            self._update_step(obs, *self._template_args, info, False, send=False)
        return self.env.reset(*args, new_task=next_task, **kwargs)

    def step(self, actions):
        obs, rews, terms, truncs, infos = self.env.step(actions)
        self.episode_length += 1
        for agent in rews.keys():
            self.episode_returns[agent] += rews[agent]

        if "task_completion" in list(infos.values())[0]:
            self.task_progress = max([info["task_completion"] for info in infos.values()])

        is_finished = (len(self.env.agents) == 0) or all(terms.values())
        # Update curriculum with step info
        if self.update_on_step:
            self._update_step(obs, rews, terms, truncs, infos, is_finished)

        if is_finished:
            episode_update = {
                "update_type": "episode",
                "metrics": (self.episode_returns, self.episode_length, self.task_space.encode(self.env.task), self.task_progress),
                "env_id": self.instance_id,
                "request_sample": True
            }
            self.components.put_update([episode_update])

        if self.change_task_on_completion and self.task_progress >= 1.0:
            update = {
                "update_type": "task_progress",
                "metrics": (self.task_space.encode(self.get_task()), self.task_progress),
                "env_id": self.instance_id,
                "request_sample": True
            }

            self.components.put_update(update)
            message = self.components.get_task()    # Blocks until a task is available
            next_task = self.task_space.decode(message["next_task"])
            self.env.change_task(next_task)

        return obs, rews, terms, truncs, infos

    def _update_step(self, obs, rews, terms, truncs, infos, is_finished, send=True):
        agent_indices = [self.agent_map[agent] for agent in rews.keys()]
        # Environment outputs
        trimmed_obs = self._trim_obs(obs)
        self._obs[self._batch_step] = trimmed_obs
        self._rews[self._batch_step][agent_indices] = list(rews.values())
        self._terms[self._batch_step][agent_indices] = list(terms.values())
        self._truncs[self._batch_step][agent_indices] = list(truncs.values())
        self._infos[self._batch_step] = infos
        self._tasks[self._batch_step] = self.task_space.encode(self.get_task())
        self._task_progresses[self._batch_step] = self.task_progress
        self._batch_step += 1

        # Send batched updates
        if self._batch_step >= self.batch_size or is_finished:
            updates = self._package_step_updates()
            self.components.put_update(updates)
            self._batch_step = 0

    def _package_step_updates(self):
        return [{
            "update_type": "step_batch",
            "metrics": ([
                self._tasks[:self._batch_step],
                self._obs[:self._batch_step],
                self._rews[:self._batch_step],
                self._terms[:self._batch_step],
                self._truncs[:self._batch_step],
                self._infos[:self._batch_step],
                self._task_progresses[:self._batch_step],
            ],),
            "env_id": self.instance_id,
            "request_sample": False
        }]

    def _trim_obs(self, obs):
        if len(self.agents) > 0 and isinstance(obs[self.agents[0]], dict):
            return {agent: {key: obs[agent][key] for key in obs[agent].keys() if key not in self.remove_keys} for agent in self.agents}
        else:
            return obs

    def get_task(self):
        # Allow user to reject task
        if hasattr(self.env, "task"):
            return self.env.task
        return self._latest_task

    def __getattr__(self, attr):
        env_attr = getattr(self.env, attr, None)
        if env_attr is not None:
            return env_attr


class RayGymnasiumSyncWrapper(gym.Wrapper):
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
        assert isinstance(env, TaskWrapper) or isinstance(env, TaskEnv) or isinstance(
            env, PettingZooTaskWrapper), "Env must implement the task API"
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

    def __getattr__(self, attr):
        env_attr = getattr(self.env, attr, None)
        if env_attr:
            return env_attr


class RayPettingZooSyncWrapper(BaseParallelWrapper):
    """
    This wrapper is used to set the task on reset for a Gym environments running
    on parallel processes created using ray. Meant to be used with a
    RayLearningProgressCurriculum running on the main process.
    """

    def __init__(self,
                 env,
                 task_space: TaskSpace,
                 update_on_step: bool = True,
                 global_task_completion: Callable[[Curriculum, np.ndarray, float, bool, Dict[str, Any]], bool] = None):
        assert isinstance(env, TaskWrapper) or isinstance(env, TaskEnv) or isinstance(
            env, PettingZooTaskWrapper), "Env must implement the task API"
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

    def __getattr__(self, attr):
        env_attr = getattr(self.env, attr, None)
        if env_attr:
            return env_attr
