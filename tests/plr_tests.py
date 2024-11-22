""" Test curriculum synchronization across multiple processes. """
import time
import random
from multiprocessing import Process

import ray

from nle.env.tasks import NetHackScore
from syllabus.examples import NethackTaskWrapper
from syllabus.curricula import LearningProgressCurriculum, PrioritizedLevelReplay
from syllabus.core import (GymnasiumSyncWrapper,
                           RayGymnasiumSyncWrapper,
                           RayCurriculumSyncWrapper,
                           CurriculumSyncWrapper,
                           make_multiprocessing_curriculum,
                           make_ray_curriculum)


N_ENVS = 2
N_EPISODES = 10


def create_nethack_env():
    env = NetHackScore()
    env = NethackTaskWrapper(env)
    return env


def create_nethack_env_queue(task_queue, update_queue):
    env = NetHackScore()
    env = NethackTaskWrapper(env)
    env = GymnasiumSyncWrapper(env,
                               env.task_space,
                               task_queue,
                               update_queue,
                               update_on_step=False,
                               default_task=0)
    return env


def create_nethack_env_ray():
    env = NetHackScore()
    env = NethackTaskWrapper(env)
    env = RayGymnasiumSyncWrapper(env, update_on_step=False, default_task=0, task_space=env.task_space)
    return env


def run_episode(env, new_task=None, curriculum=None):
    if new_task:
        obs = env.reset(new_task=new_task)
    else:
        obs = env.reset()
    term = trunc = False
    ep_rew = 0
    while not (term or trunc):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        update = {
            "update_type": "on_demand",
            "metrics": {
                "level_seeds": 1,
                "action_log_dist": 1,
                "masks": 1,
                "value_preds": 1,
                "next_value": 1,
            }
        }
        curriculum.update_curriculum(update)
        ep_rew += rew
    return ep_rew


def run_episodes(curriculum):
    env = create_nethack_env()
    ep_rews = []
    for _ in range(N_EPISODES):
        task = curriculum.sample()[0]
        ep_rews.append(run_episode(env, new_task=task, curriculum=curriculum))


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
    sample_env.reset()
    print(sample_env.observation_space.shape)
    curriculum = PrioritizedLevelReplay(([1], sample_env.action_space),
                                        {},
                                        sample_env.action_space,
                                        sample_env.task_space,
                                        random_start_tasks=10)

    print("\nRunning single process test...")
    start = time.time()
    for _ in range(N_ENVS):
        run_episodes(curriculum)
    end = time.time()
    del curriculum
    print(f"Single process test passed: {end - start:.2f}s")

    # Test Queue multi process
    curriculum, task_queue, update_queue = make_multiprocessing_curriculum(PrioritizedLevelReplay,
                                                                           ([1], sample_env.action_space),
                                                                           {},
                                                                           sample_env.action_space,
                                                                           sample_env.task_space,
                                                                           random_start_tasks=10)
    print("\nRunning Python multi process test...")
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
    curriculum = make_ray_curriculum(PrioritizedLevelReplay,
                                     ([1], sample_env.action_space),
                                     {},
                                     action_space=sample_env.action_space,
                                     sample_env.task_space,
                                     random_start_tasks=10)
    print("\nRunning Ray multi process test...")
    start = time.time()
    remotes = []

    for _ in range(N_ENVS):
        remotes.append(run_episodes_ray.remote())
    ray.get(remotes)
    end = time.time()
    del curriculum
    print(f"Python multiprocess test passed: {end - start:.2f}s")

    strategies = ["random",
                  "sequential",
                  "policy_entropy",
                  "least_confidence",
                  "min_margin",
                  "gae",
                  "value_l1",
                  "one_step_td_error"]
    for requires_buffers in strategies:
        curriculum = make_ray_curriculum(PrioritizedLevelReplay,
                                         ([1], sample_env.action_space),
                                         {},
                                         action_space=sample_env.action_space,
                                         sample_env.task_space,
                                         random_start_tasks=10)
        print(f"\nRunning {requires_buffers} test...")
        start = time.time()
        remotes = []

        for _ in range(N_ENVS):
            remotes.append(run_episodes_ray.remote())
        ray.get(remotes)
        end = time.time()
        del curriculum
        print(f"{requires_buffers} test passed: {end - start:.2f}s")
