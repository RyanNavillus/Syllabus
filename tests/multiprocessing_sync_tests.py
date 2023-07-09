""" Test curriculum synchronization across multiple processes. """
import time
import random
from multiprocessing import SimpleQueue, Process

import ray

from nle.env.tasks import NetHackScore
from syllabus.examples import NethackTaskWrapper
from syllabus.curricula import NoopCurriculum
from syllabus.core import (MultiProcessingSyncWrapper,
                           RaySyncWrapper,
                           MultiProcessingCurriculumWrapper,
                           make_multiprocessing_curriculum,
                           make_ray_curriculum)
from syllabus.tests import test_single_process, test_native_multiprocess, test_ray_multiprocess, create_nethack_env

N_ENVS = 128
N_EPISODES = 16

if __name__ == "__main__":
    sample_env = create_nethack_env()

    # Test single process speed
    print("\nRUNNING: Python single process test (4 envs)...")
    native_speed = test_single_process(create_nethack_env, num_envs=4, num_episodes=N_EPISODES)
    print(f"PASSED: single process test passed: {native_speed:.2f}s")

    # Test Queue multiprocess speed with Syllabus
    curriculum = NoopCurriculum(NetHackScore, sample_env.task_space, random_start_tasks=10)
    curriculum, task_queue, update_queue = make_multiprocessing_curriculum(curriculum, N_ENVS)
    print("\nRUNNING: Python multiprocess test with Syllabus...")
    native_syllabus_speed = test_native_multiprocess(create_nethack_env, curriculum=curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Python multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")

    # Test Ray multiprocess speed with Syllabus
    curriculum = NoopCurriculum(NetHackScore, sample_env.task_space, random_start_tasks=10)
    curriculum = make_ray_curriculum(curriculum)
    print("\nRUNNING: Ray multiprocess test with Syllabus...")
    ray_syllabus_speed = test_ray_multiprocess(create_nethack_env, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Ray multiprocess test with Syllabus: {ray_syllabus_speed:.2f}s")


