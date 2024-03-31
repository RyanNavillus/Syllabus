""" Test curriculum synchronization across multiple processes. """
import ray

from nle.env.tasks import NetHackScore
from syllabus.curricula import NoopCurriculum, DomainRandomization, LearningProgressCurriculum, CentralizedPrioritizedLevelReplay, SimpleBoxCurriculum
from syllabus.core import make_multiprocessing_curriculum, make_ray_curriculum
from syllabus.tests import run_single_process, run_native_multiprocess, run_ray_multiprocess, create_nethack_env, create_cartpole_env

N_ENVS = 2
N_EPISODES = 2

if __name__ == "__main__":
    ray.init()
    nethack_env = create_nethack_env()
    cartpole_env = create_cartpole_env()
    curricula = [
        (DomainRandomization, create_nethack_env, (nethack_env.task_space,), {"random_start_tasks": 10}),
        (CentralizedPrioritizedLevelReplay, create_nethack_env, (nethack_env.task_space,), {"device": "cpu", "suppress_usage_warnings": True, "num_processes": N_ENVS, "random_start_tasks": 10}),
    ]
    for curriculum, env_fn, args, kwargs in curricula:
        print("")
        print("*" * 80)
        print("Testing curriculum:", curriculum.__name__)
        print("*" * 80)
        print("")

        # Test single process speed
        print("RUNNING: Python single process test (2 envs)...")
        test_curriculum = curriculum(*args, **kwargs)
        native_speed = run_single_process(env_fn, curriculum=test_curriculum, num_envs=2, num_episodes=N_EPISODES)
        print(f"PASSED: single process test (2 envs) passed: {native_speed:.2f}s")

        # Test Queue multiprocess speed with Syllabus
        test_curriculum = curriculum(*args, **kwargs)
        test_curriculum, task_queue, update_queue = make_multiprocessing_curriculum(test_curriculum)
        print("\nRUNNING: Python multiprocess test with Syllabus...")
        native_syllabus_speed = run_native_multiprocess(env_fn, curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
        print(f"PASSED: Python multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")

        # Test Ray multiprocess speed with Syllabus
        test_curriculum = curriculum(*args, **kwargs)
        test_curriculum = make_ray_curriculum(test_curriculum)
        print("\nRUNNING: Ray multiprocess test with Syllabus...")
        ray_syllabus_speed = run_ray_multiprocess(env_fn, num_envs=N_ENVS, num_episodes=N_EPISODES)
        print(f"PASSED: Ray multiprocess test with Syllabus: {ray_syllabus_speed:.2f}s")
