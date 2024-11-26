""" Test curriculum synchronization across multiple processes. """
import ray

from syllabus.tests import SyncTestCurriculum
from syllabus.core import make_multiprocessing_curriculum, make_ray_curriculum, MultiagentSharedCurriculumWrapper
from syllabus.tests import run_single_process, run_native_multiprocess, run_ray_multiprocess, create_pettingzoo_synctest_env

N_ENVS = 8
N_EPISODES = 10


def evaluate_curriculum(curriculum, num_envs=N_ENVS, num_agents=2):
    stats = curriculum.get_stats()
    expected_reward = 100 * num_envs * N_EPISODES
    # Multiply by 2 for the 2 agents in teh environment
    assert stats["total_reward"] == expected_reward * \
        num_agents, f"Curriculum total reward is {stats['total_reward']}, expected {expected_reward}"
    for task, count in stats["task_counts"].items():
        if task == 0:
            assert count == 0, "Received completed error tasks, expected 0"
        else:
            assert count == num_envs * \
                num_agents, f"Curriculum task '{task}' count is {count}, expected {num_envs * num_agents}"
    expected_dones = num_envs * N_EPISODES * num_agents
    assert stats["total_dones"] == expected_dones, f"Curriculum total dones is {stats['total_dones']}, expected {expected_dones}"


def generate_environment(num_episodes=N_EPISODES):
    return create_pettingzoo_synctest_env(env_args=(num_episodes,))


def test_single_process():
    sample_env = generate_environment()
    test_curriculum = SyncTestCurriculum(N_ENVS, N_EPISODES, sample_env.task_space)
    test_curriculum = MultiagentSharedCurriculumWrapper(test_curriculum, sample_env.possible_agents)
    native_speed = run_single_process(
        create_pettingzoo_synctest_env, env_args=(
            N_EPISODES,), curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES
    )
    evaluate_curriculum(test_curriculum, num_envs=N_ENVS)


def test_queue_multiprocess():
    sample_env = generate_environment()
    test_curriculum = SyncTestCurriculum(N_ENVS, N_EPISODES, sample_env.task_space)
    test_curriculum = MultiagentSharedCurriculumWrapper(test_curriculum, sample_env.possible_agents)
    test_curriculum = make_multiprocessing_curriculum(test_curriculum, start=False)
    native_syllabus_speed = run_native_multiprocess(
        create_pettingzoo_synctest_env, env_args=(
            N_EPISODES,), curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES
    )
    evaluate_curriculum(test_curriculum.curriculum)


if __name__ == "__main__":
    test_single_process()
    test_queue_multiprocess()


# if __name__ == "__main__":
#     ray.init()
#     sample_env = create_pettingzoo_synctest_env(env_args=(N_EPISODES,))

#     print("")
#     print("*" * 80)
#     print("Testing curriculum synchronization")
#     print("*" * 80)
#     print("")

#     # Test single process speed
#     print("RUNNING: Python single process test ...")
#     test_curriculum = SyncTestCurriculum(N_ENVS, N_EPISODES, sample_env.task_space)
#     test_curriculum = MultiagentSharedCurriculumWrapper(test_curriculum, sample_env.possible_agents)
#     native_speed = run_single_process(
#         create_pettingzoo_synctest_env, env_args=(
#             N_EPISODES,), curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES
#     )
#     evaluate_curriculum(test_curriculum.curriculum, num_envs=N_ENVS, num_agents=len(sample_env.possible_agents))
#     print(f"PASSED: single process test passed: {native_speed:.2f}s")

#     # Test Queue multiprocess speed with Syllabus
#     test_curriculum = SyncTestCurriculum(N_ENVS, N_EPISODES, sample_env.task_space)
#     test_curriculum = MultiagentSharedCurriculumWrapper(test_curriculum, sample_env.possible_agents)
#     test_curriculum = make_multiprocessing_curriculum(test_curriculum, sequential_start=False)
#     print("\nRUNNING: Python multiprocess test with Syllabus...")
#     native_syllabus_speed = run_native_multiprocess(
#         create_pettingzoo_synctest_env, env_args=(
#             N_EPISODES,), curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES
#     )
#     evaluate_curriculum(test_curriculum.curriculum.curriculum, num_agents=len(sample_env.possible_agents))
#     print(f"PASSED: Python multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")

#     # Test Ray multiprocess speed with Syllabus
#     test_curriculum = SyncTestCurriculum(N_ENVS, N_EPISODES, sample_env.task_space)
#     test_curriculum = make_ray_curriculum(test_curriculum)
#     print("\nRUNNING: Ray multiprocess test with Syllabus...")
#     ray_syllabus_speed = run_ray_multiprocess(create_pettingzoo_synctest_env, env_args=(
#         N_EPISODES,), num_envs=N_ENVS, num_episodes=N_EPISODES)
#     # TODO: Implement Ray checks
#     # evaluate_curriculum(test_curriculum)
#     print(f"PASSED: Ray multiprocess test with Syllabus: {ray_syllabus_speed:.2f}s")
