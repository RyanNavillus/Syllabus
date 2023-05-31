""" Task wrapper for NLE that can change tasks at reset using the NLE's task definition format. """
import copy
import time
from typing import List
import numpy as np
import gym
from gym import spaces
from nle.env import base
from nle.env.tasks import (NetHackScore,
                           NetHackStaircase,
                           NetHackStaircasePet,
                           NetHackOracle,
                           NetHackGold,
                           NetHackEat,
                           NetHackScout)
from syllabus.core import TaskWrapper


class NethackTaskWrapper(TaskWrapper):
    """
    This wrapper allows you to change the task of an NLE environment.

    This wrapper was designed to meet two goals.
        1. Allow us to change the task of the NLE environment at the start of an episode
        2. Allow us to use the predefined NLE task definitions without copying/modifying their code.
           This allows us to integrate with other work on nethack tasks or curricula.

    However, each task is defined as a subclass of the NLE, so you need to cast and reinitialize the
    environment to change its task. This wrapper manipulates the __class__ property to achieve this,
    but does so in a safe way. Specifically, we ensure that the instance variables needed for each
    task are available and reset at the start of the episode regardless of which task is active.
    """
    def __init__(self, env: gym.Env, tasks: List[base.NLE] = None, use_provided_tasks: bool = True):
        super().__init__(env)
        self.env = env

        self.task: str = 6

        observation_keys = list(self.env._observation_keys)
        observation_keys.remove("program_state")
        observation_keys.remove("internal")
        self._init_kwargs = {
            "save_ttyrec_every": self.env._save_ttyrec_every,
            "savedir": self.env.savedir,
            "character": self.env.character,
            "max_episode_steps": self.env._max_episode_steps,
            "observation_keys": tuple(observation_keys),
            "options": None,
            "wizard": False,
            "allow_all_yn_questions": self.env._allow_all_yn_questions,
            "allow_all_modes": self.env._allow_all_modes,
            "spawn_monsters": True,
        }

        # This is set to False during reset
        self.done = True

        # Add nethack tasks provided by the base NLE
        task_list: List[base.NLE] = []
        if use_provided_tasks:
            task_list = [
                NetHackScore,
                NetHackStaircase,
                NetHackStaircasePet,
                NetHackOracle,
                NetHackGold,
                NetHackEat,
                NetHackScout,
            ]

        # Add in custom nethack tasks
        if tasks:
            for task in tasks:
                assert isinstance(task, base.NLE), "Env must subclass the base NLE"
                task_list.append(task)

        self.task_list = task_list
        self.task_space = spaces.Discrete(len(self.task_list))

        # Add goal space to observation
        self.observation_space = copy.deepcopy(self.env.observation_space)
        self.observation_space["goal"] = spaces.MultiBinary(len(self.task_list))

        # Task completion metrics
        self.episode_return = 0

        # Initialize all tasks
        original_class = self.env.__class__
        for task in task_list:
            self.env.__class__ = task
            self.env.__init__(**self._init_kwargs)

        self.env.__class__ = original_class
        self.env.__init__(**self._init_kwargs)

    @property
    def current_task(self):
        return self.env.__class__

    def _task_class(self, task):
        return self.task_list[task]

    def _task_name(self, task):
        return self._task_class(task).__name__

    def reset(self, new_task: int = None, **kwargs):
        """
        Resets the environment along with all available tasks, and change the current task.

        This ensures that all instance variables are reset, not just the ones for the current task.
        We do this efficiently by keeping track of which reset functions have already been called,
        since very few tasks override reset. If new_task is provided, we change the task before
        calling the final reset.
        """
        # Change task if new one is provided
        if new_task is not None:
            self.change_task(new_task)

        self.done = False
        self.episode_return = 0

        return self.observation(self.env.reset(**kwargs))

    def change_task(self, new_task: int):
        """
        Change task by directly editing environment class.

        Ignores requests for unknown tasks or task changes outside of a reset.
        """
        # Ignore new task if mid episode
        if self.current_task.__init__ != self._task_class(new_task).__init__ and not self.done:
            print(f"Given task {self._task_name(new_task)} needs to be reinitialized.\
                  Ignoring request to change task and keeping {self.current_task.__name__}")
            return

        # Ignore if task is unknown
        if new_task >= len(self.task_list):
            print(f"Given task {self._task_name(self.task)} not in task list.\
                  Ignoring request to change task and keeping {self.env.__class__.__name__}")
            return

        # Update current task
        self.task = new_task
        self.env.__class__ = self._task_class(new_task)

        # If task requires reinitialization
        if type(self.env).__init__ != NetHackScore.__init__:
            # TODO: Allow spawn_monsters to be disabled
            observation_keys = self.env._observation_keys
            observation_keys.remove("internal")
            self.env.__init__(**self._init_kwargs)

    def _encode_goal(self):
        goal_encoding = np.zeros(len(self.task_list))
        goal_encoding[self.task] = 1
        return goal_encoding

    def observation(self, observation):
        """
        Parses current inventory and new items gained this timestep from the observation.
        Returns a modified observation.
        """
        # Add goal to observation
        observation['goal'] = self._encode_goal()
        return observation

    def _task_completion(self, obs, rew, done, info):
        # TODO: Add real task completion metrics
        completion = 0.0
        if self.task == 0:
            completion = self.episode_return / 1000
        elif self.task == 1:
            completion = self.episode_return
        elif self.task == 2:
            completion = self.episode_return
        elif self.task == 3:
            completion = self.episode_return
        elif self.task == 4:
            completion = self.episode_return / 1000
        elif self.task == 5:
            completion = self.episode_return / 10
        elif self.task == 6:
            completion = self.episode_return / 100

        return min(max(completion, 0.0), 1.0)

    def step(self, action):
        """
        Step through environment and update task completion.
        """
        obs, rew, done, info = self.env.step(action)
        self.episode_return += rew
        self.done = done
        info["task_completion"] = self._task_completion(obs, rew, done, info)
        return self.observation(obs), rew, done, info


if __name__ == "__main__":
    def run_episode(env, task: str = None, verbose=1):
        env.reset(new_task=task)
        task_name = type(env.unwrapped).__name__
        done = False
        ep_rew = 0
        while not done:
            action = env.action_space.sample()
            _, rew, done, _ = env.step(action)
            ep_rew += rew
        if verbose:
            print(f"Episodic reward for {task_name}: {ep_rew}")

    print("Testing NethackTaskWrapper")
    N_EPISODES = 100

    # Initialize NLE
    nethack_env = NetHackScore()
    nethack_task_env = NethackTaskWrapper(nethack_env)

    start_time = time.time()

    for _ in range(N_EPISODES):
        run_episode(nethack_task_env, verbose=0)

    end_time = time.time()
    print(f"Run time same task: {end_time - start_time}")
    start_time = time.time()

    for _ in range(N_EPISODES):
        nethack_task = nethack_task_env.task_space.sample()
        run_episode(nethack_task_env, task=nethack_task, verbose=0)

    end_time = time.time()
    print(f"Run time swapping tasks: {end_time - start_time}")
