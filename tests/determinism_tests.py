import gym
import procgen  # noqa: F401
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

from syllabus.examples.task_wrappers import ProcgenTaskWrapper
from syllabus.tests import test_determinism

from nle.env.tasks import NetHackScore, NetHackStaircase, NetHackStaircasePet, NetHackScout, NetHackEat, NetHackGold
from syllabus.examples.task_wrappers import NethackTaskWrapper
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0


def make_nethack_env(seed=42):
    env = NetHackScore()
    env = GymV21CompatibilityV0(env=env)
    env = NethackTaskWrapper(env, seed=seed)
    return env


def make_env(seed=42):
    env_id = "bigfish"
    env = gym.make(f"procgen-{env_id}-v0", start_level=int(seed), num_levels=1, distribution_mode="easy")
    env = GymV21CompatibilityV0(env=env)
    env = ProcgenTaskWrapper(env, env_id, seed)
    return env


test_determinism(make_nethack_env, verbose=1)
