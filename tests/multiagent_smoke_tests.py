""" Test curriculum synchronization across multiple processes. """
import ray
from syllabus.core.evaluator import DummyEvaluator
from syllabus.curricula import NoopCurriculum, DomainRandomization, LearningProgressCurriculum, PrioritizedLevelReplay
from syllabus.core import make_multiprocessing_curriculum, make_ray_curriculum, MultiagentSharedCurriculumWrapper
from syllabus.tests import run_single_process, run_native_multiprocess, run_ray_multiprocess, create_simpletag_env, create_pistonball_env, get_test_values

N_ENVS = 2
N_EPISODES = 2

if __name__ == "__main__":
    ray.init()
    # simpletag_env = create_simpletag_env()
    pistonball_env = create_pistonball_env()
    # default_task = simpletag_env.task_space.encode((4, 4, 4))
    default_task = pistonball_env.task_space.encode(1)
    evaluator = DummyEvaluator(pistonball_env.action_space("piston_0"))

    curricula = [
        (NoopCurriculum, create_pistonball_env, (default_task, pistonball_env.task_space), {}),
        (DomainRandomization, create_pistonball_env, (pistonball_env.task_space,), {}),
        # (LearningProgressCurriculum, create_pistonball_env, (pistonball_env.task_space,), {}),
        (PrioritizedLevelReplay, create_pistonball_env, (pistonball_env.task_space, pistonball_env.observation_space), {
         "evaluator": evaluator, "device": "cpu", "num_processes": N_ENVS*len(pistonball_env.possible_agents), "num_steps": 2048}),
        # (SimpleBoxCurriculum, create_cartpole_env, (cartpole_env.task_space,), {}),
    ]

    for curriculum, env_fn, args, kwargs in curricula:
        print("")
        print("*" * 80)
        print("Testing curriculum:", curriculum.__name__)
        print("*" * 80)
        print("")

        sample_env = env_fn()

        # Test single process speed
        print("RUNNING: Single process test (2 envs)...")
        test_curriculum = curriculum(*args, **kwargs)
        test_curriculum = MultiagentSharedCurriculumWrapper(test_curriculum, sample_env.possible_agents)
        native_speed = run_single_process(env_fn, curriculum=test_curriculum, num_envs=2, num_episodes=N_EPISODES)
        print(f"PASSED: Single process test (2 envs) passed: {native_speed:.2f}s")

        # Test single process speed
        print("\nRUNNING: Single process test (2 envs)...")
        test_curriculum = curriculum(*args, **kwargs)
        test_curriculum = MultiagentSharedCurriculumWrapper(
            test_curriculum, sample_env.possible_agents, joint_policy=True)
        native_speed = run_single_process(env_fn, curriculum=test_curriculum, num_envs=2, num_episodes=N_EPISODES)
        print(f"PASSED: Single process test (2 envs) passed: {native_speed:.2f}s")

        # Test multiprocess process speed without Syllabus
        print("\nRUNNING: Python native multiprocess test (2 envs)...")
        test_curriculum = curriculum(*args, **kwargs)
        test_curriculum = MultiagentSharedCurriculumWrapper(test_curriculum, sample_env.possible_agents)
        native_speed = run_native_multiprocess(env_fn, num_envs=N_ENVS, num_episodes=N_EPISODES)
        print(f"PASSED: Python native multiprocess test (2 envs) passed: {native_speed:.2f}s")

        # Test multiprocess process speed without Syllabus
        print("\nRUNNING: Python native multiprocess test (2 envs)...")
        test_curriculum = curriculum(*args, **kwargs)
        test_curriculum = MultiagentSharedCurriculumWrapper(
            test_curriculum, sample_env.possible_agents, joint_policy=True)
        native_speed = run_native_multiprocess(env_fn, num_envs=N_ENVS, num_episodes=N_EPISODES)
        print(f"PASSED: Python native multiprocess test (2 envs) passed: {native_speed:.2f}s")

        # Test Queue multiprocess speed with Syllabus
        test_curriculum = curriculum(*args, **kwargs)
        test_curriculum = MultiagentSharedCurriculumWrapper(test_curriculum, sample_env.possible_agents)
        test_curriculum = make_multiprocessing_curriculum(test_curriculum, sequential_start=False)
        print("\nRUNNING: Python native multiprocess test with Syllabus...")
        native_syllabus_speed = run_native_multiprocess(
            env_fn, curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
        print(f"PASSED: Python native multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")

        # Test Queue multiprocess speed with Syllabus
        test_curriculum = curriculum(*args, **kwargs)
        test_curriculum = MultiagentSharedCurriculumWrapper(
            test_curriculum, sample_env.possible_agents, joint_policy=True)
        test_curriculum = make_multiprocessing_curriculum(test_curriculum, sequential_start=False)
        print("\nRUNNING: Python native multiprocess test with Syllabus...")
        native_syllabus_speed = run_native_multiprocess(
            env_fn, curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
        print(f"PASSED: Python native multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")
