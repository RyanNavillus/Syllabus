""" Test curriculum synchronization across multiple processes. """
import time
import random
from multiprocessing import SimpleQueue, Process
from gym.spaces import Tuple, Dict
import ray

from nle.env.tasks import NetHackScore
from syllabus.examples import NethackTaskWrapper
from syllabus.curricula import NoopCurriculum, Uniform
from syllabus.core import (MultiProcessingSyncWrapper,
                           RaySyncWrapper,
                           RayCurriculumWrapper,
                           MultiProcessingCurriculumWrapper,
                           MultitaskWrapper,
                           make_multiprocessing_curriculum,
                           make_ray_curriculum)
from syllabus.tests import test_single_process, test_native_multiprocess, test_ray_multiprocess

N_ENVS = 1
N_EPISODES = 16

class MultivariateNethackTaskWrapper(NethackTaskWrapper):
    def reset(self, new_task = None, **kwargs):
        assert len(new_task) == 16, new_task
        if new_task is not None:
            self.change_task(new_task[0])

        self.done = False
        self.episode_return = 0

        return self.observation(self.env.reset(**kwargs))


def create_multivariate_nethack_env():
    env = NetHackScore()
    env = MultivariateNethackTaskWrapper(env)
    return env


def create_multivariate_nethack_env_queue(task_queue, update_queue, update_on_step=False):
    env = NetHackScore()
    env = MultivariateNethackTaskWrapper(env)
    env = MultiProcessingSyncWrapper(env,
                                     task_queue,
                                     update_queue,
                                     update_on_step=update_on_step,
                                     default_task=0,
                                     task_space=env.task_space)
    return env


def create_multivariate_nethack_env_ray(update_on_step=False):
    env = NetHackScore()
    env = MultivariateNethackTaskWrapper(env)
    env = RaySyncWrapper(env, update_on_step=update_on_step, default_task=0, task_space=env.task_space)
    return env

if __name__ == "__main__":
    sample_env = create_multivariate_nethack_env()

    # Test Queue multiprocess speed with Syllabus
    curriculum = Uniform(sample_env.task_space)
    curriculum = MultitaskWrapper(curriculum, num_components=16)
    curriculum, task_queue, update_queue = make_multiprocessing_curriculum(curriculum, N_ENVS)
    print("\nRUNNING: Python multiprocess test with Syllabus...")
    native_syllabus_speed = test_native_multiprocess(curriculum=curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Python multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")

    # Test Ray multiprocess speed with Syllabus
    curriculum = Uniform(sample_env.task_space)
    curriculum = MultitaskWrapper(curriculum, num_components=16)
    curriculum = make_ray_curriculum(curriculum)
    print("\nRUNNING: Ray multiprocess test with Syllabus...")
    ray_syllabus_speed = test_ray_multiprocess(curriculum=curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Ray multiprocess test with Syllabus: {ray_syllabus_speed:.2f}s")

