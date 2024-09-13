
import random
import gym
import gymnasium
import numpy as np
import torch
import procgen      # type: ignore # noqa: F401
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from syllabus.core import Evaluator, CleanRLDiscreteEvaluator
from syllabus.examples.models import ProcgenAgent, CartPoleAgent


def make_env(env_id):
    def thunk():
        try:
            env = gymnasium.make(env_id)
        except gymnasium.error.NameNotFound:
            env = gym.make(env_id, start_level=0, num_levels=1)
            env = GymV21CompatibilityV0(env=env)

        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        env.observation_space.seed(1)
        env.action_space.seed(1)
        return env

    return thunk


def test_evaluator():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    envs = gymnasium.vector.AsyncVectorEnv(
        [make_env("CartPole-v1") for i in range(4)]
    )
    agent = CartPoleAgent(envs)
    evaluator = CleanRLDiscreteEvaluator(agent)

    obs, _ = envs.reset(seed=1)

    # Test the Evaluator
    action, value = evaluator.get_action_value(obs)

    assert all(torch.where((action == 0) | (action == 1), True, False)), "Action is out of bounds"
    assert torch.equal(action, torch.Tensor([1, 0, 1, 1])), "Action is incorrect"
    assert value.shape == (4, 1), "Value shape is incorrect"
    assert np.allclose(
        value.detach().numpy(),
        np.array([[0.02202436], [-0.01820074], [-0.00138423], [-0.01939073]])
    ), f"Value is incorrect: {value.detach().numpy()}"


# TODO: Test Procgen model
def test_evaluator_procgen():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    envs = gymnasium.vector.AsyncVectorEnv(
        [make_env("procgen-bigfish-v0") for i in range(4)]
    )
    agent = ProcgenAgent((64, 64, 3), 15)
    evaluator = Evaluator(agent, get_value=agent.get_value, get_action=agent.get_action)

    obs, _ = envs.reset()

    # Test the Evaluator
    action, value = evaluator.get_action_value(obs)

    assert all(torch.where((action >= 0) & (action <= 14), True, False)), "Action is out of bounds"
    assert torch.equal(action, torch.Tensor([7, 1, 12, 14])), "Action is incorrect"
    assert value.shape == (4, 1), "Value shape is incorrect"
    assert np.allclose(
        value.detach().numpy(),
        np.array([[0.00294205], [0.00294205], [0.00294205], [0.00294205]])
    ), f"Value is incorrect: {value.detach().numpy()}"
