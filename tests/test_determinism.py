import gym
import procgen  # noqa: F401
import random
from syllabus.tests import evaluate_random_policy
from syllabus.examples.task_wrappers import ProcgenTaskWrapper

N_EPISODES = 10
import os
seeds = [int.from_bytes(os.urandom(4), byteorder="little") for _ in range(10)]
print(seeds)


def make_env(seed=None):
    env_id = "bigfish"
    env = gym.make(f"procgen-{env_id}-v0", start_level=seed, num_levels=1, distribution_mode="easy")
    env = ProcgenTaskWrapper(env, env_id, seed)

    # Seed environment
    gym.utils.seeding.np_random(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


return1, _ = evaluate_random_policy(make_env, num_episodes=N_EPISODES, seeds=list(range(N_EPISODES)), reseed_after_reset=True)
return2, _ = evaluate_random_policy(make_env, num_episodes=N_EPISODES, seeds=list(range(N_EPISODES)), reseed_after_reset=True)
assert return1 == return2, f"Random policy returns are not deterministic! {return1} != {return2}"


avg_returns, returns = evaluate_random_policy(make_env, num_episodes=N_EPISODES, seeds=list(range(N_EPISODES)))
assert all(returns == avg_returns), f"Episodes are not deterministic! {returns}"


return1, _ = evaluate_random_policy(make_env, num_episodes=N_EPISODES, seed=2, reseed_after_reset=True)
return2, _ = evaluate_random_policy(make_env, num_episodes=N_EPISODES, seed=10, reseed_after_reset=True)
assert return1 != return2, f"Random policy returns with different seeds are the same. {return1} == {return2}"
