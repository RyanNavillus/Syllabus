""" Test curriculum synchronization across multiple processes. """
import ray

from syllabus.tests import SyncTestCurriculum
from syllabus.core import make_multiprocessing_curriculum, make_ray_curriculum
from syllabus.tests import test_single_process, test_native_multiprocess, test_ray_multiprocess, create_synctest_env

N_ENVS = 128
N_EPISODES = 300

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
        expected_reward = 100 * num_envs * N_EPISODES
        assert stats["total_reward"] == expected_reward, f"Curriculum total reward is {stats['total_reward']}, expected {expected_reward}"
        for task, count in stats["task_counts"].items():
            if task == 0:
                assert count == 0, "Received completed error tasks, expected 0"
            else:
                assert count == num_envs, f"Curriculum task '{task}' count is {count}, expected {num_envs}"
        expected_dones = num_envs * N_EPISODES
        assert stats["total_dones"] == expected_dones, f"Curriculum total dones is {stats['total_dones']}, expected {expected_dones}"

    # Test single process speed
    print("RUNNING: Python single process test ...")
    test_curriculum = SyncTestCurriculum(N_ENVS, N_EPISODES, sample_env.task_space)
    native_speed = test_single_process(
        create_synctest_env, env_args=(N_EPISODES,), curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES
    )
    evaluate_curriculum(test_curriculum, num_envs=N_ENVS)
    print(f"PASSED: single process test passed: {native_speed:.2f}s")

    # Test Queue multiprocess speed with Syllabus
    test_curriculum = SyncTestCurriculum(N_ENVS, N_EPISODES, sample_env.task_space)
    test_curriculum = make_multiprocessing_curriculum(test_curriculum, sequential_start=False)
    print("\nRUNNING: Python multiprocess test with Syllabus...")
    native_syllabus_speed = test_native_multiprocess(
        create_synctest_env, env_args=(N_EPISODES,), curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES
    )
    evaluate_curriculum(test_curriculum.curriculum)
    print(f"PASSED: Python multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")

    # Test Ray multiprocess speed with Syllabus
    test_curriculum = SyncTestCurriculum(N_ENVS, N_EPISODES, sample_env.task_space)
    test_curriculum = make_ray_curriculum(test_curriculum)
    print("\nRUNNING: Ray multiprocess test with Syllabus...")
    ray_syllabus_speed = test_ray_multiprocess(create_synctest_env, env_args=(N_EPISODES,), num_envs=N_ENVS, num_episodes=N_EPISODES)
    # TODO: Implement Ray checks
    # evaluate_curriculum(test_curriculum)
    print(f"PASSED: Ray multiprocess test with Syllabus: {ray_syllabus_speed:.2f}s")
