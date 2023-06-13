""" Test curriculum synchronization across multiple processes. """
import gym
import time
import random
from multiprocessing import SimpleQueue, Process

import ray

from nle.env.tasks import NetHackScore
from syllabus.examples import NethackTaskWrapper
from syllabus.curricula import NoopCurriculum, Uniform
from syllabus.core import (MultiProcessingSyncWrapper,
                           RaySyncWrapper,
                           MultiProcessingCurriculumWrapper,
                           make_multiprocessing_curriculum,
                           make_ray_curriculum)


N_ENVS = 128
N_EPISODES = 16


class MutableNethackTaskWrapper(NethackTaskWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_space = gym.spaces.Discrete(5)


def create_nethack_env():
    env = NetHackScore()
    env = NethackTaskWrapper(env)
    return env


def create_nethack_env_queue(task_queue, update_queue, update_on_step=True):
    env = NetHackScore()
    env = NethackTaskWrapper(env)
    env = MultiProcessingSyncWrapper(env,
                                     task_queue,
                                     update_queue,
                                     update_on_step=update_on_step,
                                     default_task=0,
                                     task_space=env.task_space)
    return env


def create_nethack_env_ray(update_on_step=True):
    env = NetHackScore()
    env = NethackTaskWrapper(env)
    env = RaySyncWrapper(env, update_on_step=update_on_step, default_task=0, task_space=env.task_space)
    return env


def run_episode(env, new_task=None, curriculum=None, update_on_step=True):
    if new_task:
        obs = env.reset(new_task=new_task)
    else:
        obs = env.reset()
    done = False
    ep_rew = 0
    while not done:
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if curriculum is not None and update_on_step:
            curriculum._on_step(obs, rew, done, info)
        ep_rew += rew
    return ep_rew


def run_episodes(curriculum=None, update_on_step=True):
    env = create_nethack_env()
    ep_rews = []
    for _ in range(N_EPISODES):
        if curriculum:
            task = curriculum.sample()[0]
            ep_rews.append(run_episode(env, new_task=task, curriculum=curriculum, update_on_step=update_on_step))
            curriculum._complete_task(task, success_prob=random.random())
            if curriculum.task_space.n < 7:
                curriculum.add_task("")
        else:
            ep_rews.append(run_episode(env))


def run_episodes_queue(task_queue, update_queue, update_on_step=True):
    env = create_nethack_env_queue(task_queue, update_queue, update_on_step=update_on_step)
    ep_rews = []
    for _ in range(N_EPISODES):
        ep_rews.append(run_episode(env))
        if env.task_space.n < 7:
            env.add_task("")



@ray.remote
def run_episodes_ray_syllabus(update_on_step=True):
    env = create_nethack_env_ray(update_on_step=update_on_step)
    ep_rews = []
    for _ in range(N_EPISODES):
        ep_rews.append(run_episode(env))
        if env.task_space.n < 7:
            env.add_task("")


@ray.remote
def run_episodes_ray():
    env = create_nethack_env()
    ep_rews = []
    for _ in range(N_EPISODES):
        ep_rews.append(run_episode(env))


if __name__ == "__main__":

    # Test single process speed
    sample_env = create_nethack_env()
    curriculum = Uniform(sample_env.task_space, random_start_tasks=10)
    print("\nRunning Python single process test (4 envs)...")
    start = time.time()
    actors = []
    for _ in range(4):
        run_episodes(curriculum=curriculum, update_on_step=False)
    end = time.time()
    native_speed = end - start
    print(f"Python single process test passed: {native_speed:.2f}s")

    # Test Queue multiprocess speed with Syllabus
    sample_env = create_nethack_env()
    curriculum = Uniform(sample_env.task_space, random_start_tasks=10)
    curriculum, task_queue, update_queue = make_multiprocessing_curriculum(curriculum, N_ENVS)
    print("\nRunning Python multiprocess test with Syllabus...")
    start = time.time()
    actors = []
    for _ in range(N_ENVS):
        actors.append(Process(target=run_episodes_queue, args=(task_queue, update_queue, False)))

    for actor in actors:
        actor.start()
    for actor in actors:
        actor.join()
    end = time.time()
    del curriculum
    native_syllabus_speed = end - start
    print(f"Python multiprocess test with Syllabus passed: {native_syllabus_speed:.2f}s")

    # Test Ray multiprocess speed with Syllabus
    sample_env = create_nethack_env()
    curriculum = Uniform(sample_env.task_space, random_start_tasks=10)
    curriculum = make_ray_curriculum(curriculum)
    print("\nRunning Ray multiprocess test with Syllabus...")
    start = time.time()
    remotes = []
    for _ in range(N_ENVS):
        remotes.append(run_episodes_ray_syllabus.remote(update_on_step=False))
    ray.get(remotes)
    del curriculum
    end = time.time()
    ray_syllabus_speed = end - start
    print(f"Ray multiprocess test with Syllabus passed: {ray_syllabus_speed:.2f}s")


