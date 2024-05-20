""" Test curriculum synchronization across multiple processes. """
from syllabus.tests import SyncTestCurriculum
from syllabus.core import make_multiprocessing_curriculum, make_ray_curriculum

from syllabus.tests import run_single_process, run_native_multiprocess, run_ray_multiprocess, create_synctest_env

# Setup global variables
N_ENVS = 128
N_EPISODES = 300


def evaluate_curriculum(curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES):
    stats = curriculum.get_stats()
    expected_reward = 100 * num_envs * num_episodes
    assert stats["total_reward"] == expected_reward, f"Curriculum total reward is {stats['total_reward']}, expected {expected_reward}"
    for task, count in stats["task_counts"].items():
        if task == 0:
            assert count == 0, "Received completed error tasks, expected 0"
        else:
            assert count == num_envs, f"Curriculum task '{task}' count is {count}, expected {num_envs}"
    expected_dones = num_envs * num_episodes
    assert stats["total_dones"] == expected_dones, f"Curriculum total dones is {stats['total_dones']}, expected {expected_dones}"


def generate_environment(num_episodes=N_EPISODES):
    return create_synctest_env(env_args=(num_episodes,))


def test_single_process():
    sample_env = generate_environment()
    test_curriculum = SyncTestCurriculum(N_ENVS, N_EPISODES, sample_env.task_space)
    native_speed = run_single_process(
        create_synctest_env, env_args=(N_EPISODES,), curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES
    )
    evaluate_curriculum(test_curriculum, num_envs=N_ENVS)


def test_queue_multiprocess():
    sample_env = generate_environment()
    test_curriculum = SyncTestCurriculum(N_ENVS, N_EPISODES, sample_env.task_space)
    test_curriculum = make_multiprocessing_curriculum(test_curriculum, sequential_start=False)
    native_syllabus_speed = run_native_multiprocess(
        create_synctest_env, env_args=(N_EPISODES,), curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES
    )
    evaluate_curriculum(test_curriculum.curriculum)


def test_ray_multiprocess(ray_session):
    sample_env = generate_environment()
    test_curriculum = SyncTestCurriculum(N_ENVS, N_EPISODES, sample_env.task_space)
    test_curriculum = make_ray_curriculum(test_curriculum)
    ray_syllabus_speed = run_ray_multiprocess(create_synctest_env, env_args=(N_EPISODES,), num_envs=N_ENVS, num_episodes=N_EPISODES)
    # TODO: Implement Ray checks
    # evaluate_curriculum(test_curriculum)


if __name__ == "__main__":
    test_single_process()
    test_queue_multiprocess()
    test_ray_multiprocess()
