""" Test curriculum synchronization across multiple processes. """
import time
import random
from multiprocessing import SimpleQueue, Process
from copy import deepcopy

import ray

from nle.env.tasks import (NetHackScore,
                           NetHackStaircase,
                           NetHackStaircasePet,
                           NetHackOracle,
                           NetHackGold,
                           NetHackEat,
                           NetHackScout)

from syllabus.tests import SyncTestCurriculum, SyncTestEnv
from syllabus.core import make_multiprocessing_curriculum, make_ray_curriculum, RayCurriculumWrapper
from syllabus.tests import test_single_process, test_native_multiprocess, test_ray_multiprocess, create_synctest_env

N_ENVS = 4
N_EPISODES = 2

if __name__ == "__main__":
    ray.init()
    sample_env = create_synctest_env(env_args=(N_EPISODES,))

    print("")
    print("*" * 80)
    print("Testing curriculum synchronization")
    print("*" * 80)
    print("")

    def evaluate_curriculum(curriculum, num_envs=N_ENVS):
        stats = curriculum.get_stats()
        assert stats["total_reward"] == 10 * num_envs * N_EPISODES, f"Curriculum total reward is {stats['total_reward']}, expected {10 * N_ENVS * N_EPISODES}"
        for task, count in stats["task_counts"].items():
            assert count == num_envs, f"Curriculum task {task} count is {count}, expected {num_envs}"
        


    # Test single process speed
    print("RUNNING: Python single process test (4 envs)...")
    test_curriculum = SyncTestCurriculum(4, N_EPISODES, sample_env.task_space)
    native_speed = test_single_process(create_synctest_env, env_args=(N_EPISODES,), curriculum=test_curriculum, num_envs=4, num_episodes=N_EPISODES)
    evaluate_curriculum(test_curriculum, num_envs=4)
    print(f"PASSED: single process test passed: {native_speed:.2f}s")

    # Test Queue multiprocess speed with Syllabus
    test_curriculum = SyncTestCurriculum(N_ENVS, N_EPISODES, sample_env.task_space)
    test_curriculum, task_queue, update_queue = make_multiprocessing_curriculum(test_curriculum, N_ENVS)
    print("\nRUNNING: Python multiprocess test with Syllabus...")
    native_syllabus_speed = test_native_multiprocess(create_synctest_env, env_args=(N_EPISODES,), curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
    evaluate_curriculum(test_curriculum.curriculum)
    print(f"PASSED: Python multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")

    # Test Ray multiprocess speed with Syllabus
    test_curriculum = SyncTestCurriculum(N_ENVS, N_EPISODES, sample_env.task_space)
    test_curriculum = make_ray_curriculum(test_curriculum)
    print("\nRUNNING: Ray multiprocess test with Syllabus...")
    ray_syllabus_speed = test_ray_multiprocess(create_synctest_env, env_args=(N_EPISODES,), num_envs=N_ENVS, num_episodes=N_EPISODES)
    #evaluate_curriculum(test_curriculum)
    print(f"PASSED: Ray multiprocess test with Syllabus: {ray_syllabus_speed:.2f}s")


