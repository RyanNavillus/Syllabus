import gym
import procgen  # noqa: F401
import numpy as np

from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from syllabus.examples.task_wrappers import ProcgenTaskWrapper
from syllabus.tests import evaluate_random_policy

N_EPISODES = 10
import os

seeds = [int.from_bytes(os.urandom(4), byteorder="little") for _ in range(10)]


def print_if_verbose(verbose, *args, **kwargs):
    if verbose > 0:
        print(*args, **kwargs)


def make_env(seed=42):
    env_id = "bigfish"
    env = gym.make(f"procgen-{env_id}-v0", start_level=int(seed), num_levels=1, distribution_mode="easy")
    env = GymV21CompatibilityV0(env=env)
    env = ProcgenTaskWrapper(env, env_id, seed)
    return env


def compare_obs(obs1, obs2):
    if not isinstance(obs1, type(obs2)):
        print(f"Type mismatch: {type(obs1)} != {type(obs2)}")
        return False

    if isinstance(obs1, dict):
        for key in obs1.keys():
            if key not in obs2:
                print(f"Key {key} not in obs2")
                return False
            if not compare_obs(obs1[key], obs2[key]):
                return False
        return True
    if isinstance(obs1, (list, tuple, np.ndarray)):
        if len(obs1) != len(obs2):
            print(f"Length mismatch: {len(obs1)} != {len(obs2)}")
            return False
        for i in range(len(obs1)):
            if not compare_obs(obs1[i], obs2[i]):
                return False
        return True
    else:
        return obs1 == obs2


def compare_episodes(make_env, task1, task2, verbose=0):
    term1 = trunc1 = term2 = trunc2 = False
    step = 0
    num_obs_failed = 0
    num_rews_failed = 0

    # Set tasks
    env1 = make_env()
    env1.reset(new_task=task1)
    env2 = make_env()
    env2.reset(new_task=task2)

    # Seed spaces
    env1.action_space.seed(0)
    env1.observation_space.seed(0)
    env1.task_space.seed(0)
    env2.action_space.seed(0)
    env2.observation_space.seed(0)
    env2.task_space.seed(0)

    while not (term1 or trunc1 or term2 or trunc2):
        action1 = env1.action_space.sample()
        action2 = env2.action_space.sample()

        # Check actions
        if action1 != action2:
            print_if_verbose(verbose, f"Step {step}: Actions are not the same: {action1} != {action2}. Stopping test.")
            return False

        obs1, rew1, term1, trunc1, info1 = env1.step(action1)
        obs2, rew2, term2, trunc2, info2 = env2.step(action2)

        # Check observations
        if not compare_obs(obs1, obs2):
            if num_obs_failed == 0:
                print_if_verbose(verbose, f"Step {step}: Obs are not the same: {obs1} != {obs2}. This message will not print for future steps.")
            num_obs_failed += 1

        # Check rewards
        if rew1 != rew2:
            if num_rews_failed == 0:
                print_if_verbose(verbose, f"Step {step}: Rewards are not the same: {rew1} != {rew2}. This message will not print for future steps.")
            num_rews_failed += 1

        # Check terms
        if term1 != term2:
            print_if_verbose(verbose, f"Step {step}: Terms are not the same: {term1} != {term2}. Stopping test.")
            return False

        # Check truncs
        if trunc1 != trunc2:
            print_if_verbose(verbose, f"Step {step}: Truncs are not the same: {trunc1} != {trunc2}. Stopping test.")
            return False

        step += 1
    return num_obs_failed == 0 and num_rews_failed == 0


def test_determinism(make_env, verbose=0):
    # TODO: Use the task space to sampele seeds/tasks
    test_env = make_env()
    assert hasattr(test_env, "task_space"), "Environment does not have a task space. make_env must return a TaskEnv or use a TaskWrapper."
    task_space = test_env.task_space

    print_if_verbose(verbose, "Runnning determinism tests...")

    # Test full episode returns
    print_if_verbose(verbose, "\nTesting average episodic returns...")
    seeds = [task_space.sample() for _ in range(N_EPISODES)]
    return1, _ = evaluate_random_policy(make_env, num_episodes=N_EPISODES, seeds=seeds)
    return2, _ = evaluate_random_policy(make_env, num_episodes=N_EPISODES, seeds=seeds)
    full_return_test = return1 == return2
    if full_return_test:
        print_if_verbose(verbose, "PASSED: Random policy returns are deterministic!")
    else:
        print_if_verbose(verbose, f"FAILED: Random policy returns are not deterministic! {return1} != {return2}")

    # Test individual episode returns
    print_if_verbose(verbose, "\nTesting individual episode rewards...")
    avg_returns, returns = evaluate_random_policy(make_env, num_episodes=N_EPISODES, seeds=[task_space.sample()] * N_EPISODES)
    return_test = all(returns == avg_returns)
    if return_test:
        print_if_verbose(verbose, "PASSED: Episodes returns are deterministic!")
    else:
        print_if_verbose(verbose, f"FAILED: Episodes returns are not deterministic! {avg_returns} != {returns}")

    print_if_verbose(verbose, "\nTesting different seeds...")
    task1 = task2 = task_space.sample()
    while task1 == task2:
        task2 = task_space.sample()
    return1, _ = evaluate_random_policy(make_env, num_episodes=N_EPISODES, seeds=[task1] * N_EPISODES)
    return2, _ = evaluate_random_policy(make_env, num_episodes=N_EPISODES, seeds=[task2] * N_EPISODES)

    test3 = return1 != return2
    if test3:
        print_if_verbose(verbose, "PASSED: Random policy returns with different seeds are different.")
    else:
        print_if_verbose(verbose, f"FAILED: Random policy returns with different seeds are the same. {return1} == {return2}")

    print_if_verbose(verbose, "\nTesting actions, rewards, and observations seeds...")
    task1 = task2 = task_space.sample()
    step_tests_same = compare_episodes(make_env, task1, task2, verbose=verbose)
    while task1 == task2:
        task2 = task_space.sample()
    step_tests_different = compare_episodes(make_env, task1, 2, verbose=0)

    if step_tests_same and not step_tests_different:
        print_if_verbose(verbose, "PASSED: Environment returns on individual steps are deterministic with respect to seed.")
    elif step_tests_different:
        print_if_verbose(verbose, "FAILED: Environment returns on individual steps are deterministic even with different seeds.")
    else:
        print_if_verbose(verbose, "FAILED: Environment returns on individual steps are not deterministic with the same seed.")

    return {
        "avg_episodic_returns": full_return_test,
        "episodic_returns": return_test,
        "step_values": step_tests_same
    }


test_determinism(make_env, verbose=1)
