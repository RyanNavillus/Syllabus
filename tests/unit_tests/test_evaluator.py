
import gymnasium as gym
from syllabus.core import Evaluator, CleanRLDiscreteEvaluator
from syllabus.examples.models import ProcgenAgent, CartPoleAgent


def make_env():
    def thunk():
        env = gym.make("CartPole-v1")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def test_evaluator():
    envs = gym.vector.AsyncVectorEnv(
        [make_env() for i in range(4)]
    )
    agent = CartPoleAgent(envs)
    evaluator = CleanRLDiscreteEvaluator(agent)

    obs, _ = envs.reset()

    # Test the Evaluator
    action, value = evaluator.get_action_value(obs)

    print(action, value)


if __name__ == "__main__":
    test_evaluator()
