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

from syllabus.tests import test_native_multiprocess, test_ray_multiprocess, create_nethack_env, create_synctest_env

N_ENVS = 128
N_EPISODES = 50


if __name__ == "__main__":
    ray.init()
    env_fn = create_nethack_env
    default_task = NetHackScore
    env_args=()
    env_kwargs={}
    sample_env = env_fn(env_args=env_args, env_kwargs=env_kwargs)
    # TODO: Test single process speed with Syllabus (with and without step updates)

    # Test Queue multiprocess speed
    print("\nRUNNING: Python multiprocess test...")
    native_speed = test_native_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Python multiprocess test: {native_speed:.2f}s")

    # Test Queue multiprocess speed with Syllabus
    curriculum = NoopCurriculum(default_task, sample_env.task_space, random_start_tasks=10)
    curriculum, task_queue, update_queue = make_multiprocessing_curriculum(curriculum)
    print("\nRUNNING: Python multiprocess test with Syllabus...")
    native_syllabus_speed = test_native_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, curriculum=curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Python multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")

    # Test Ray multi process
    print("\nRUNNING: Ray multiprocess test...")
    ray_speed = test_ray_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Ray multiprocess test: {ray_speed:.2f}s")

    # Test Ray multiprocess speed with Syllabus
    curriculum = NoopCurriculum(default_task, sample_env.task_space, random_start_tasks=10)
    curriculum = make_ray_curriculum(curriculum)
    print("\nRUNNING: Ray multiprocess test with Syllabus...")
    ray_syllabus_speed = test_ray_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, curriculum=curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Ray multiprocess test with Syllabus: {ray_syllabus_speed:.2f}s")

    # Print native speed comparisons
    print("\n")
    print(f"Relative speed of native multiprocessing with Syllabus: {100 * native_speed / native_syllabus_speed:.2f}%")
    print(f"Relative speed Ray multiprocessing with Syllabus: {100 * ray_speed / ray_syllabus_speed:.2f}%")
    print("")

    # Test Queue multiprocess speed with Syllabus
    curriculum = NoopCurriculum(default_task, sample_env.task_space, random_start_tasks=10)
    curriculum, task_queue, update_queue = make_multiprocessing_curriculum(curriculum)
    # Test Queue multiprocess speed with Syllabus (no step updates)
    print("\nRUNNING: Python multiprocess test with Syllabus (no step updates) ...")
    native_syllabus_speed_nostep = test_native_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, curriculum=curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES, update_on_step=False)
    print(f"PASSED: Python multiprocess test with Syllabus (no step updates): {native_syllabus_speed_nostep:.2f}s")

    # Test Ray multiprocess speed with Syllabus (no step updates)
    curriculum = NoopCurriculum(default_task, sample_env.task_space, random_start_tasks=10)
    curriculum = make_ray_curriculum(curriculum)
    print("\nRUNNING: Ray multiprocess test with Syllabus (no step updates) ...")
    ray_syllabus_speed_nostep = test_ray_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, curriculum=curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES, update_on_step=False)
    print(f"PASSED: Ray multiprocess test with Syllabus (no step updates): {ray_syllabus_speed_nostep:.2f}s")

    print("\n")
    print(f"Relative speed of native multiprocessing with Syllabus without step updates: {100 * native_speed / native_syllabus_speed_nostep:.2f}%")
    print(f"Relative speed Ray multiprocessing with Syllabus without step updates: {100 * ray_speed / ray_syllabus_speed_nostep:.2f}%")
    print("\n")
