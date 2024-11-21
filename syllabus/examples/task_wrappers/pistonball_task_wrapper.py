""" Task wrapper for NLE that can change tasks at reset using the NLE's task definition format. """
from pettingzoo.butterfly import pistonball_v6
from pettingzoo.utils.env import ParallelEnv

from syllabus.core import PettingZooTaskWrapper
from syllabus.task_space import DiscreteTaskSpace


class PistonballTaskWrapper(PettingZooTaskWrapper):
    """
    This wrapper simply changes the seed of a Minigrid environment.
    """

    def __init__(self, env: ParallelEnv):
        super().__init__(env)
        self.env = env
        self.env.unwrapped.task = 1
        self.task = None

        # Task completion metrics
        self.episode_return = 0
        self.task_space = DiscreteTaskSpace(11)   # 0.1 - 1.0 friction

    def reset(self, new_task: int = None, **kwargs):
        # Change task if new one is provided
        # if new_task is not None:
        #     self.change_task(new_task)

        self.episode_return = 0
        if new_task is not None:
            task = new_task / 10
            # Inject current_task into the environment
            self.env = pistonball_v6.parallel_env(
                ball_friction=task, continuous=False, max_cycles=125
            )
            self.task = new_task
        return self.observation(self.env.reset(**kwargs))
