import gym
import procgen  # noqa: F401
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

from syllabus.examples.task_wrappers import ProcgenTaskWrapper
from syllabus.tests import test_determinism


def make_env(seed=42):
    env_id = "bigfish"
    env = gym.make(f"procgen-{env_id}-v0", start_level=int(seed), num_levels=1, distribution_mode="easy")
    env = GymV21CompatibilityV0(env=env)
    env = ProcgenTaskWrapper(env, env_id, seed)
    return env


test_determinism(make_env, verbose=1)
