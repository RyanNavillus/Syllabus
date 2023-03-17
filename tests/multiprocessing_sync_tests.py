""" Test curriculum synchronization across multiple processes. """
import time
import random
from multiprocessing import SimpleQueue, Process

import ray

from nle.env.tasks import NetHackScore
from syllabus.examples import NethackTaskWrapper
from syllabus.curricula import LearningProgressCurriculum
from syllabus.core import (MultiProcessingSyncWrapper,
                           RaySyncWrapper,
                           RayCurriculumWrapper,
                           MultiProcessingCurriculumWrapper,
                           make_multiprocessing_curriculum,
                           make_ray_curriculum)


N_ENVS = 8
N_EPISODES = 50


def create_nethack_env():
    env = NetHackScore()
    env = NethackTaskWrapper(env)
    return env


def create_nethack_env_queue(task_queue, update_queue):
    env = NetHackScore()
    env = NethackTaskWrapper(env)
    env = MultiProcessingSyncWrapper(env,
                                     task_queue,
                                     update_queue,
                                     update_on_step=True,
                                     default_task=0,
                                     task_space=env.task_space)
    return env


def create_nethack_env_ray():
    env = NetHackScore()
    env = NethackTaskWrapper(env)
    env = RaySyncWrapper(env, update_on_step=True, default_task=0, task_space=env.task_space)
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
        if curriculum:
            curriculum.on_step(obs, rew, done, info)
        ep_rew += rew
    return ep_rew


def run_episodes(curriculum):
    env = create_nethack_env()
    ep_rews = []
    for _ in range(N_EPISODES):
        task = curriculum.sample()[0]
        ep_rews.append(run_episode(env, new_task=task, curriculum=curriculum))
        curriculum._complete_task(task, success_prob=random.random())


def run_episodes_queue(task_queue, update_queue):
    env = create_nethack_env_queue(task_queue, update_queue)
    ep_rews = []
    for _ in range(N_EPISODES):
        ep_rews.append(run_episode(env))


@ray.remote
def run_episodes_ray():
    env = create_nethack_env_ray()
    ep_rews = []
    for _ in range(N_EPISODES):
        ep_rews.append(run_episode(env))


if __name__ == "__main__":
    # Test single process
    sample_env = create_nethack_env()
    curriculum = LearningProgressCurriculum(sample_env.task_space, random_start_tasks=10)
    print("\nRunning single process test...")
    start = time.time()
    for _ in range(N_ENVS):
        run_episodes(curriculum)
    end = time.time()
    print(f"Single process test passed: {end - start:.2f}s")

    # Test Queue multi process
    curriculum, task_queue, update_queue = make_multiprocessing_curriculum(LearningProgressCurriculum,
                                                                           sample_env.task_space,
                                                                           random_start_tasks=10)
    print("\nRunning Python multiprocess test...")
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
    print(f"Python multiprocess test passed: {end - start:.2f}s")

    # Test Ray multi process
    curriculum = make_ray_curriculum(LearningProgressCurriculum, sample_env.task_space, random_start_tasks=10)
    print("\nRunning Ray multiprocess test...")
    start = time.time()
    remotes = []
    for _ in range(N_ENVS):
        remotes.append(run_episodes_ray.remote())
    ray.get(remotes)
    del curriculum
    end = time.time()
    print(f"Ray multiprocess test passed: {end - start:.2f}s")
