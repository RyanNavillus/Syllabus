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


N_ENVS = 128
N_EPISODES = 16

class MultivariateNethackTaskWrapper(NethackTaskWrapper):
    def reset(self, new_task: int = None, **kwargs):
        assert len(new_task) == 16, new_task
        if new_task is not None:
            self.change_task(new_task[0])

        self.done = False
        self.episode_return = 0

        return self.observation(self.env.reset(**kwargs))


def create_nethack_env():
    env = NetHackScore()
    env = MultivariateNethackTaskWrapper(env)
    return env


def create_nethack_env_queue(task_queue, update_queue, update_on_step=False):
    env = NetHackScore()
    env = MultivariateNethackTaskWrapper(env)
    env = MultiProcessingSyncWrapper(env,
                                     task_queue,
                                     update_queue,
                                     update_on_step=update_on_step,
                                     default_task=0,
                                     task_space=env.task_space)
    return env


def create_nethack_env_ray(update_on_step=False):
    env = NetHackScore()
    env = MultivariateNethackTaskWrapper(env)
    env = RaySyncWrapper(env, update_on_step=update_on_step, default_task=0, task_space=env.task_space)
    return env


def run_episode(env, new_task=None, curriculum=None):
    if new_task:
        obs = env.reset(new_task=new_task)
    else:
        obs = env.reset()
    done = False
    ep_rew = 0
    while not done:
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        # if curriculum:
        #     curriculum.on_step(obs, rew, done, info)
        ep_rew += rew
    return ep_rew


def run_episodes(curriculum=None):
    env = create_nethack_env()
    ep_rews = []
    for _ in range(N_EPISODES):
        if curriculum:
            task = curriculum.sample()[0]
            ep_rews.append(run_episode(env, new_task=task, curriculum=curriculum))
            curriculum._complete_task(task, success_prob=random.random())
        else:
            ep_rews.append(run_episode(env))


def run_episodes_queue(task_queue, update_queue, update_on_step=False):
    env = create_nethack_env_queue(task_queue, update_queue, update_on_step=update_on_step)
    ep_rews = []
    for _ in range(N_EPISODES):
        ep_rews.append(run_episode(env))


@ray.remote
def run_episodes_ray_syllabus(update_on_step=False):
    env = create_nethack_env_ray(update_on_step=update_on_step)
    ep_rews = []
    for _ in range(N_EPISODES):
        ep_rews.append(run_episode(env))


@ray.remote
def run_episodes_ray():
    env = create_nethack_env()
    ep_rews = []
    for _ in range(N_EPISODES):
        ep_rews.append(run_episode(env))


if __name__ == "__main__":
    sample_env = create_nethack_env()

    # Test Queue multiprocess speed with Syllabus
    curriculum = Uniform(sample_env.task_space)
    curriculum = MultitaskWrapper(curriculum, num_components=16)
    curriculum, task_queue, update_queue = make_multiprocessing_curriculum(curriculum, N_ENVS)
    print("\nRunning Python multiprocess test with Syllabus...")
    start = time.time()
    actors = []
    for _ in range(N_ENVS):
        actors.append(Process(target=run_episodes_queue, args=(task_queue, update_queue)))

    for actor in actors:
        actor.start()
    for actor in actors:
        actor.join()
    end = time.time()
    del curriculum
    native_syllabus_speed = end - start
    print(f"Python multiprocess test with Syllabus passed: {native_syllabus_speed:.2f}s")

    # Test Ray multiprocess speed with Syllabus
    curriculum = Uniform(sample_env.task_space)
    curriculum = MultitaskWrapper(curriculum, num_components=16)
    curriculum = make_ray_curriculum(curriculum)
    print("\nRunning Ray multiprocess test with Syllabus...")
    start = time.time()
    remotes = []
    for _ in range(N_ENVS):
        remotes.append(run_episodes_ray_syllabus.remote())
    ray.get(remotes)
    del curriculum
    end = time.time()
    ray_syllabus_speed = end - start
    print(f"Ray multiprocess test with Syllabus passed: {ray_syllabus_speed:.2f}s")

