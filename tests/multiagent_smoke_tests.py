""" Test curriculum synchronization across multiple processes. """
import ray
import pettingzoo
import gymnasium as gym

from nle.env.tasks import NetHackScore
from syllabus.curricula import NoopCurriculum, DomainRandomization, LearningProgressCurriculum, PrioritizedLevelReplay
from syllabus.core import make_multiprocessing_curriculum, make_ray_curriculum
from syllabus.tests import test_single_process, test_native_multiprocess, test_ray_multiprocess, create_pistonball_env, create_nethack_env, get_test_values

N_ENVS = 2
N_EPISODES = 2

if __name__ == "__main__":
    ray.init()
    pistonball_env = create_pistonball_env()
    curricula = [
        (NoopCurriculum, create_pistonball_env, (0, pistonball_env.task_space), {}),
        (DomainRandomization, create_pistonball_env, (pistonball_env.task_space,), {}),
        (LearningProgressCurriculum, create_pistonball_env, (pistonball_env.task_space,), {}),
        (PrioritizedLevelReplay, create_pistonball_env, (pistonball_env.task_space, pistonball_env.observation_space), {"get_value": get_test_values, "device": "cpu", "num_processes": N_ENVS, "num_steps": 2048}),
        # (SimpleBoxCurriculum, create_cartpole_env, (cartpole_env.task_space,), {}),
    ]

    for curriculum, env_fn, args, kwargs in curricula:
        print("")
        print("*" * 80)
        print("Testing curriculum:", curriculum.__name__)
        print("*" * 80)
        print("")

        # Test single process speed
        print("RUNNING: Single process test (2 envs)...")
        test_curriculum = curriculum(*args, **kwargs)
        native_speed = test_single_process(env_fn, curriculum=test_curriculum, num_envs=2, num_episodes=N_EPISODES)
        print(f"PASSED: Single process test (2 envs) passed: {native_speed:.2f}s")

        # Test multiprocess process speed without Syllabus
        print("RUNNING: Python native multiprocess test (2 envs)...")
        test_curriculum = curriculum(*args, **kwargs)
        native_speed = test_native_multiprocess(env_fn, num_envs=N_ENVS, num_episodes=N_EPISODES)
        print(f"PASSED: Python native multiprocess test (2 envs) passed: {native_speed:.2f}s")

        # Test Queue multiprocess speed with Syllabus
        test_curriculum = curriculum(*args, **kwargs)
        test_curriculum = make_multiprocessing_curriculum(test_curriculum, sequential_start=False)
        print("\nRUNNING: Python native multiprocess test with Syllabus...")
        native_syllabus_speed = test_native_multiprocess(env_fn, curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
        print(f"PASSED: Python native multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")

        # Test Ray multiprocess speed with Syllabus
        test_curriculum = curriculum(*args, **kwargs)
        test_curriculum = make_ray_curriculum(test_curriculum)
        print("\nRUNNING: Ray multiprocess test with Syllabus...")
        ray_syllabus_speed = test_ray_multiprocess(env_fn, num_envs=N_ENVS, num_episodes=N_EPISODES)
        print(f"PASSED: Ray multiprocess test with Syllabus: {ray_syllabus_speed:.2f}s")
