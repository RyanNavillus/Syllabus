""" Test curriculum synchronization across multiple processes. """
import time
import random
from multiprocessing import SimpleQueue, Process

import ray

from nle.env.tasks import NetHackScore
from examples import NethackTaskWrapper
from curricula import LearningProgressCurriculum
from syllabus import (MultiProcessingSyncWrapper,
                      RaySyncWrapper,
                      RayCurriculumWrapper,
                      MultiProcessingCurriculumWrapper)


N_ENVS = 8
N_EPISODES = 20


def create_nethack_env():
    env = NetHackScore()
    env = NethackTaskWrapper(env)
    return env


def create_nethack_env_queue(task_queue, complete_queue, step_queue):
    env = NetHackScore()
    env = NethackTaskWrapper(env)
    env = MultiProcessingSyncWrapper(env,
                                     task_queue,
                                     complete_queue,
                                     step_queue=step_queue,
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
        curriculum.complete_task(task, success_prob=random.random())


def run_episodes_queue(task_queue, complete_queue, step_queue):
    env = create_nethack_env_queue(task_queue, complete_queue, step_queue)
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
    del sample_env

    print("\nRunning single process test...")
    start = time.time()
    for _ in range(N_ENVS):
        run_episodes(curriculum)
    end = time.time()
    print(f"Single process test passed: {end - start:.2f}s")

    # Test Queue multi process
    task_queue = SimpleQueue()
    complete_queue = SimpleQueue()
    step_queue = SimpleQueue()
    sample_env = create_nethack_env_queue(task_queue, complete_queue, step_queue)
    curriculum = LearningProgressCurriculum(sample_env.task_space, random_start_tasks=10)
    curriculum = MultiProcessingCurriculumWrapper(curriculum,
                                                  task_queue=task_queue,
                                                  complete_queue=complete_queue,
                                                  step_queue=step_queue,
                                                  task_space=sample_env.task_space)
    del sample_env
    curriculum.start()
    time.sleep(3)

    print("\nRunning Python multi process test...")
    start = time.time()
    actors = []
    for _ in range(N_ENVS):
        actors.append(Process(target=run_episodes_queue, args=(task_queue, complete_queue, step_queue)))

    for actor in actors:
        actor.start()
    for actor in actors:
        actor.join()
    end = time.time()
    curriculum.stop()
    del curriculum
    print(f"Multi process test passed: {end - start:.2f}s")

    # Test Ray multi process
    sample_env = NethackTaskWrapper(NetHackScore())
    curriculum = RayCurriculumWrapper(LearningProgressCurriculum,
                                      sample_env.task_space,
                                      random_start_tasks=10)
    del sample_env

    print("\nRunning Ray multi process test...")
    start = time.time()
    remotes = []
    for _ in range(N_ENVS):
        remotes.append(run_episodes_ray.remote())
    ray.get(remotes)
    end = time.time()
    print(f"Multi process test passed: {end - start:.2f}s")
