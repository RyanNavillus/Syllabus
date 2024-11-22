""" Test curriculum synchronization across multiple processes. """
from copy import deepcopy

import ray

from nle.env.tasks import (NetHackScore,
                           NetHackStaircase,
                           NetHackStaircasePet,
                           NetHackOracle,
                           NetHackGold,
                           NetHackEat,
                           NetHackScout)

from syllabus.curricula import SequentialCurriculum
from syllabus.core import make_multiprocessing_curriculum, make_ray_curriculum
from syllabus.tests import run_single_process, run_native_multiprocess, run_ray_multiprocess, create_nethack_env

N_ENVS = 2
N_EPISODES = 2

if __name__ == "__main__":
    ray.init()
    sample_env = create_nethack_env()

    manual_tasks = [NetHackScore, NetHackStaircase, NetHackStaircasePet,
                    NetHackOracle, NetHackGold, NetHackEat, NetHackScout]
    num_repeats = [N_ENVS * N_EPISODES] * len(manual_tasks)
    curriculum = SequentialCurriculum(manual_tasks, sample_env.task_space, num_repeats=num_repeats, repeat_list=False)
    print("")
    print("*" * 80)
    print("Testing curriculum:", curriculum.__class__.__name__)
    print("*" * 80)
    print("")

    # Test single process speed
    print("RUNNING: Python single process test (4 envs)...")
    test_curriculum = deepcopy(curriculum)
    native_speed = run_single_process(create_nethack_env, curriculum=test_curriculum,
                                      num_envs=4, num_episodes=N_EPISODES)
    print(f"PASSED: single process test passed: {native_speed:.2f}s")

    # Test Queue multiprocess speed with Syllabus
    test_curriculum = deepcopy(curriculum)
    test_curriculum, task_queue, update_queue = make_multiprocessing_curriculum(test_curriculum, N_ENVS)
    print("\nRUNNING: Python multiprocess test with Syllabus...")
    native_syllabus_speed = run_native_multiprocess(
        create_nethack_env, curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Python multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")

    # Test Ray multiprocess speed with Syllabus
    test_curriculum = deepcopy(curriculum)
    test_curriculum = make_ray_curriculum(test_curriculum)
    print("\nRUNNING: Ray multiprocess test with Syllabus...")
    ray_syllabus_speed = run_ray_multiprocess(create_nethack_env, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Ray multiprocess test with Syllabus: {ray_syllabus_speed:.2f}s")
