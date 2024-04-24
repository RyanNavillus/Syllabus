""" Task wrapper for NLE that can change tasks at reset using the NLE's task definition format. """
import time
from typing import Callable, Tuple, Union

import gymnasium as gym

from .task_wrapper import TaskWrapper


class ReinitTaskWrapper(TaskWrapper):
    """
    This is a general wrapper for tasks defined as subclasses of a base environment.

    This wrapper reinitializes the environment with the provided env function at the start of each episode.
    This is a simple, general solution to using Syllabus with tasks that need to be reinitialized, but it is inefficient.
    It's likely that you can achieve better performance by using a more specialized wrapper.
    """
    def __init__(self, env: gym.Env, env_fn: Callable, task_space: gym.Space = None):
        super().__init__(env)

        self.env_fn = env_fn
        self.task_envs = {}     # Save instance of each task environment to avoid reinitializing
        self.task_space = task_space
        self.task = None

    def encode_task(self, task):
        """
        Override to convert task description into an element of the MultiDiscrete task space.
        This is the identity function by default.
        """
        return task

    def decode_task(self, encoding):
        """ 
        Override to convert element of the MultiDiscrete task space into format usable by the reinit env_fn. 
        This is the identity function by default.  
        """
        return encoding

    def reset(self, new_task: Union[Tuple, int, float] = None, **kwargs):
        """
        Resets the environment along with all available tasks, and change the current task.
        """
        # Change task if new one is provided
        if new_task is not None:
            self.change_task(new_task)

        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def change_task(self, new_task: Union[Tuple, int, float]):
        """
        Change task by directly editing environment class.

        This ensures that all instance variables are reset, not just the ones for the current task.
        We do this efficiently by keeping track of which reset functions have already been called,
        since very few tasks override reset. If new_task is provided, we change the task before
        calling the final reset.
        """

        # Update current task
        if new_task not in self.task_envs:
            self.task_envs[new_task] = self.env_fn(self.decode_task(new_task))

        self.env = self.task_envs[new_task]
        self.task = new_task

    def step(self, action):
        """
        Step through environment and update task completion.
        """
        obs, rew, term, trunc, info = self.env.step(action)
        info["task_completion"] = self._task_completion(obs, rew, term, trunc, info)
        return self.observation(obs), rew, term, trunc, info


if __name__ == "__main__":
    from nle.env.tasks import (NetHackEat, NetHackGold, NetHackOracle,
                               NetHackScore, NetHackScout, NetHackStaircase,
                               NetHackStaircasePet)

    def run_episode(env, task: str = None, verbose=1):
        env.reset(new_task=task)
        task_name = type(env.unwrapped).__name__
        term = trunc = False
        ep_rew = 0
        while not (term or trunc):
            action = env.action_space.sample()
            _, rew, term, trunc, _ = env.step(action)
            ep_rew += rew
        if verbose:
            print(f"Episodic reward for {task_name}: {ep_rew}")

    print("Testing NethackTaskWrapper")
    N_EPISODES = 100

    # Initialize NLE
    def create_env(task):
        task_class = [NetHackScore, NetHackStaircase, NetHackStaircasePet, NetHackOracle, NetHackGold, NetHackEat, NetHackScout][task]
        return task_class()

    nethack_env = NetHackScore()    
    nethack_task_env = ReinitTaskWrapper(nethack_env, create_env)

    start_time = time.time()

    for _ in range(N_EPISODES):
        run_episode(nethack_task_env, verbose=0)

    end_time = time.time()
    print(f"Run time same task: {end_time - start_time}")
    start_time = time.time()

    for _ in range(N_EPISODES):
        nethack_task = gym.spaces.Discrete(7).sample()
        run_episode(nethack_task_env, task=nethack_task, verbose=0)

    end_time = time.time()
    print(f"Run time swapping tasks: {end_time - start_time}")
