import gymnasium as gym

from syllabus.core import TaskWrapper
from syllabus.task_space import DiscreteTaskSpace

PROCGEN_RETURN_BOUNDS = {
    "coinrun": (5, 10),
    "starpilot": (2.5, 64),
    "caveflyer": (3.5, 12),
    "dodgeball": (1.5, 19),
    "fruitbot": (-1.5, 32.4),
    "chaser": (0.5, 13),
    "miner": (1.5, 13),
    "jumper": (3, 10),
    "leaper": (3, 10),
    "maze": (5, 10),
    "bigfish": (1, 40),
    "heist": (3.5, 10),
    "climber": (2, 12.6),
    "plunder": (4.5, 30),
    "ninja": (3.5, 10),
    "bossfight": (0.5, 13),
}


class ProcgenTaskWrapper(TaskWrapper):
    """
    This wrapper allows you to change the task of an NLE environment.
    """

    def __init__(self, env: gym.Env, env_id, seed=0):
        super().__init__(env)
        self.task_space = DiscreteTaskSpace(200)
        self.env_id = env_id
        self.task = seed
        self.seed(seed)
        self.episode_return = 0

        self.observation_space = self.env.observation_space

    def seed(self, seed):
        self.env.gym_env.unwrapped._venv.seed(int(seed), 0)

    def reset(self, new_task=None, **kwargs):
        """
        Resets the environment along with all available tasks, and change the current task.

        This ensures that all instance variables are reset, not just the ones for the current task.
        We do this efficiently by keeping track of which reset functions have already been called,
        since very few tasks override reset. If new_task is provided, we change the task before
        calling the final reset.
        """
        self.episode_return = 0.0

        # Change task if new one is provided
        if new_task is not None:
            self.change_task(new_task)

        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def change_task(self, new_task: int):
        """
        Change task by directly editing environment class.

        Ignores requests for unknown tasks or task changes outside of a reset.
        """
        seed = int(new_task)
        self.task = seed
        self.seed(seed)

    def step(self, action):
        """
        Step through environment and update task completion.
        """
        obs, rew, term, trunc, info = self.env.step(action)
        self.episode_return += rew

        env_min, env_max = PROCGEN_RETURN_BOUNDS[self.env_id]
        normalized_return = (self.episode_return - env_min) / float(env_max - env_min)
        clipped_return = 1 if normalized_return > 0.1 else 0    # Binary progress
        info["task_completion"] = clipped_return

        return self.observation(obs), rew, term, trunc, info

    def observation(self, obs):
        return obs
