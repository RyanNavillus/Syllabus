
import random
import gym
import gymnasium
import numpy as np
import torch
import procgen      # type: ignore # noqa: F401
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from syllabus.core import Evaluator, CleanRLDiscreteEvaluator
from syllabus.examples.models import ProcgenAgent, CartPoleAgent
from syllabus.examples.models.procgen_model import ProcgenLSTMAgent


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
    action, value, extras = evaluator.get_action_and_value(obs)

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
    evaluator = CleanRLDiscreteEvaluator(agent)

    obs, _ = envs.reset()

    # Test the Evaluator
    action, value, extras = evaluator.get_action_and_value(obs)
    print(value.detach().numpy())
    assert all(torch.where((action >= 0) & (action <= 14), True, False)), "Action is out of bounds"
    assert torch.equal(action, torch.Tensor([0, 10, 0, 1])), "Action is incorrect"
    assert value.shape == (4, 1), "Value shape is incorrect"
    assert np.allclose(
        value.detach().numpy(),
        np.array([[-0.01905577], [-0.01905577], [-0.01905577], [-0.01905577]])
    ), f"Value is incorrect: {value.detach().numpy()}"


def test_evaluator_lstm():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    envs = gymnasium.vector.AsyncVectorEnv(
        [make_env("procgen-bigfish-v0") for i in range(4)]
    )
    agent = ProcgenLSTMAgent((64, 64, 3), 15, {"hidden_size": 256})
    evaluator = CleanRLDiscreteEvaluator(agent)

    obs, _ = envs.reset()

    # Test the Evaluator
    lstm_state = (
        torch.zeros(1, 4, 256),
        torch.zeros(1, 4, 256),
    )
    action, value, extras = evaluator.get_action_and_value(obs, lstm_state=lstm_state, done=torch.zeros(1, 4, 1))
    next_lstm_state = extras["lstm_state"]
    print(torch.all(torch.not_equal(next_lstm_state[0], torch.zeros(1, 4, 256))))
    assert next_lstm_state[0].shape == lstm_state[0].shape, "Cell state is the wrong shape"
    assert next_lstm_state[1].shape == lstm_state[1].shape, "Hidden state is the wrong shape"
    assert torch.all(torch.not_equal(next_lstm_state[0], torch.zeros(1, 4, 256))), "Cell state is zero after step"
    assert torch.all(torch.not_equal(next_lstm_state[1], torch.zeros(1, 4, 256))), "Hidden state is zero after step"
    assert all(torch.where((action >= 0) & (action <= 14), True, False)), "Action is out of bounds"
    assert torch.equal(action, torch.Tensor([13, 6, 11, 5])), "Action is incorrect"
    assert value.shape == (4, 1), "Value shape is incorrect"
    assert np.allclose(
        value.detach().numpy(),
        np.array([[-0.03898524], [-0.03898524], [-0.03898524], [-0.03898524]])
    ), f"Value is incorrect: {value.detach().numpy()}"
