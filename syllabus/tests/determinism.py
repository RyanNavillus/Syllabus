from random import randint

import numpy as np
from pettingzoo.utils.env import ParallelEnv

from syllabus.tests import evaluate_random_policy


def print_if_verbose(verbose, *args, **kwargs):
    if verbose > 0:
        print(*args, **kwargs)


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
    env = make_env()
    if isinstance(env, ParallelEnv):
        return compare_episodes_pettingzoo(make_env, task1, task2, verbose=verbose)
    else:
        return compare_episodes_gymnasium(make_env, task1, task2, verbose=verbose)


def compare_episodes_gymnasium(make_env, task1, task2, verbose=0):
    step = 0
    num_obs_failed = 0
    num_rews_failed = 0

    # Set tasks
    env1 = make_env()
    env1.seed(seed=task1)
    env1.reset()
    env2 = make_env()
    env2.seed(seed=task2)
    env2.reset()

    # Seed spaces
    env1.action_space.seed(0)
    env1.observation_space.seed(0)
    env1.task_space.seed(0)
    env2.action_space.seed(0)
    env2.observation_space.seed(0)
    env2.task_space.seed(0)

    term1 = trunc1 = term2 = trunc2 = False
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
                print_if_verbose(
                    verbose, f"Step {step}: Obs are not the same: {obs1} != {obs2}. This message will not print for future steps.")
            num_obs_failed += 1

        # Check rewards
        if rew1 != rew2:
            if num_rews_failed == 0:
                print_if_verbose(
                    verbose, f"Step {step}: Rewards are not the same: {rew1} != {rew2}. This message will not print for future steps.")
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


def compare_episodes_pettingzoo(make_env, task1, task2, verbose=0):
    step = 0
    num_obs_failed = 0
    num_rews_failed = 0

    # Set tasks
    env1 = make_env()
    env1.seed(seed=task1)
    env1.reset()
    env2 = make_env()
    env2.seed(seed=task2)
    env2.reset()

    # Seed spaces
    for agent in env1.possible_agents:
        env1.action_space(agent).seed(0)
        env1.observation_space(agent).seed(0)
        env1.task_space.seed(0)
        env2.action_space(agent).seed(0)
        env2.observation_space(agent).seed(0)
        env2.task_space.seed(0)

    term1 = trunc1 = term2 = trunc2 = {agent: False for agent in env1.agents}
    while not (all(term1.values()) or all(trunc1.values()) or all(term2.values()) or all(trunc2.values())):
        action1 = {agent: env1.action_space(agent).sample() for agent in env1.agents}
        action2 = {agent: env2.action_space(agent).sample() for agent in env2.agents}

        # Check actions
        if action1 != action2:
            print_if_verbose(verbose, f"Step {step}: Actions are not the same: {action1} != {action2}. Stopping test.")
            return False

        obs1, rew1, term1, trunc1, info1 = env1.step(action1)
        obs2, rew2, term2, trunc2, info2 = env2.step(action2)

        all_agents = set(obs1.keys()).intersection(set(obs2.keys()))

        # Check observations
        for agent in all_agents:
            obs1_agent = obs1[agent]
            obs2_agent = obs2[agent]
            if not compare_obs(obs1_agent, obs2_agent):
                if num_obs_failed == 0:
                    print_if_verbose(
                        verbose, f"Step {step}: Obs are not the same for agent {agent}: {obs1_agent} != {obs2_agent}. This message will not print for future steps.")
                num_obs_failed += 1

        # Check rewards
        for agent in all_agents:
            rew1_agent = rew1[agent]
            rew2_agent = rew2[agent]

            if rew1_agent != rew2_agent:
                if num_rews_failed == 0:
                    print_if_verbose(
                        verbose, f"Step {step}: Rewards are not the same for agent {agent}: {rew1_agent} != {rew2_agent}. This message will not print for future steps.")
                num_rews_failed += 1

        # Check terms
        for agent in all_agents:
            term1_agent = term1[agent]
            term2_agent = term2[agent]
            if term1_agent != term2_agent:
                print_if_verbose(
                    verbose, f"Step {step}: Terms are not the same for agent {agent}: {term1_agent} != {term2_agent}. Stopping test.")
                return False

        # Check truncs
        for agent in all_agents:
            trunc1_agent = trunc1[agent]
            trunc2_agent = trunc2[agent]
            if trunc1_agent != trunc2_agent:
                print_if_verbose(
                    verbose, f"Step {step}: Truncs are not the same for agent {agent}: {trunc1_agent} != {trunc2_agent}. Stopping test.")
                return False

        step += 1
    return num_obs_failed == 0 and num_rews_failed == 0


def test_determinism(make_env, num_episodes=10, verbose=0):
    # TODO: Use the task space to sampele seeds/tasks
    test_env = make_env()
    assert hasattr(
        test_env, "task_space"), "Environment does not have a task space. make_env must return a TaskEnv or use a TaskWrapper."
    task_space = test_env.task_space

    print_if_verbose(verbose, "Runnning determinism tests...")

    # Test full episode returns
    print_if_verbose(verbose, "\nTesting average episodic returns...")

    seeds = [i for i in range(num_episodes)]
    return1, _ = evaluate_random_policy(make_env, num_episodes=num_episodes, seeds=seeds)
    return2, _ = evaluate_random_policy(make_env, num_episodes=num_episodes, seeds=seeds)
    full_return_test = return1 == return2
    if full_return_test:
        print_if_verbose(verbose, "PASSED: Random policy returns are deterministic!")
    else:
        print_if_verbose(verbose, f"FAILED: Random policy returns are not deterministic! {return1} != {return2}")

    # Test individual episode returns
    print_if_verbose(verbose, "\nTesting individual episode rewards...")
    avg_returns, returns = evaluate_random_policy(make_env, num_episodes=num_episodes, seeds=[
                                                  randint(0, 1000000)] * num_episodes)
    return_test = all([ret == avg_returns for ret in returns])
    if return_test:
        print_if_verbose(verbose, "PASSED: Episodes returns are deterministic!")
    else:
        print_if_verbose(verbose, f"FAILED: Episodes returns are not deterministic! {avg_returns} != {returns}")

    print_if_verbose(verbose, "\nTesting different seeds...")
    task1 = task2 = randint(0, 1000000)
    while task1 == task2:
        task2 = randint(0, 1000000)
    return1, _ = evaluate_random_policy(make_env, num_episodes=num_episodes, seeds=[task1] * num_episodes)
    return2, _ = evaluate_random_policy(make_env, num_episodes=num_episodes, seeds=[task2] * num_episodes)

    test3 = return1 != return2
    if test3:
        print_if_verbose(verbose, "PASSED: Random policy returns with different seeds are different.")
    else:
        print_if_verbose(
            verbose, f"FAILED: Random policy returns with different seeds are the same. {return1} == {return2}")

    print_if_verbose(verbose, "\nTesting actions, rewards, and observations seeds...")
    task1 = task2 = randint(0, 1000000)
    step_tests_same = compare_episodes(make_env, task1, task2, verbose=verbose)
    while task1 == task2:
        task2 = randint(0, 1000000)
    step_tests_different = compare_episodes(make_env, task1, task2, verbose=0)

    if step_tests_same and not step_tests_different:
        print_if_verbose(verbose, "PASSED: Environment returns on individual steps are deterministic with respect to seed.")
    elif step_tests_different:
        print_if_verbose(
            verbose, "FAILED: Environment returns on individual steps are deterministic even with different seeds.")
    else:
        print_if_verbose(
            verbose, "FAILED: Environment returns on individual steps are not deterministic with the same seed.")

    return {
        "avg_episodic_returns": full_return_test,
        "episodic_returns": return_test,
        "step_values": step_tests_same
    }
