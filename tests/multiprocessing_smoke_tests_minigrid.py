""" Test curriculum synchronization across multiple processes. """

from copy import deepcopy

import ray

from syllabus.curricula import NoopCurriculum, DomainRandomization, LearningProgressCurriculum, CentralizedPrioritizedLevelReplay
from syllabus.core import make_multiprocessing_curriculum, make_ray_curriculum
from syllabus.tests import run_single_process, run_native_multiprocess, run_ray_multiprocess, create_minigrid_env

N_ENVS = 128
N_EPISODES = 16


if __name__ == "__main__":
    sample_env = create_minigrid_env()
    curricula = [
        NoopCurriculum("MiniGrid-DoorKey-5x5-v0", sample_env.task_space, random_start_tasks=10),
        DomainRandomization(sample_env.task_space, random_start_tasks=10),
        LearningProgressCurriculum(sample_env.task_space, random_start_tasks=10),
        CentralizedPrioritizedLevelReplay(sample_env.task_space, random_start_tasks=10, device="cpu", suppress_usage_warnings=True)
    ]
    for curriculum in curricula:
        print("")
        print("*" * 80)
        print("Testing curriculum:", curriculum.__class__.__name__)
        print("*" * 80)
        print("")

        # Test single process speed
        print("RUNNING: Python single process test (4 envs)...")
        test_curriculum = deepcopy(curriculum)
        native_speed = run_single_process(create_minigrid_env, curriculum=test_curriculum, num_envs=4, num_episodes=N_EPISODES)
        print(f"PASSED: single process test passed: {native_speed:.2f}s")

        # Test Queue multiprocess speed with Syllabus
        test_curriculum = deepcopy(curriculum)
        test_curriculum, task_queue, update_queue = make_multiprocessing_curriculum(test_curriculum)
        print("\nRUNNING: Python multiprocess test with Syllabus...")
        native_syllabus_speed = run_native_multiprocess(create_minigrid_env, curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
        print(f"PASSED: Python multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")

        # Test Ray multiprocess speed with Syllabus
        test_curriculum = deepcopy(curriculum)
        test_curriculum = make_ray_curriculum(test_curriculum)
        print("\nRUNNING: Ray multiprocess test with Syllabus...")
        ray_syllabus_speed = run_ray_multiprocess(create_minigrid_env, num_envs=N_ENVS, num_episodes=N_EPISODES)
        print(f"PASSED: Ray multiprocess test with Syllabus: {ray_syllabus_speed:.2f}s")
