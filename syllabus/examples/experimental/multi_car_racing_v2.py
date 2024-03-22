# flake8: noqa: F401
import argparse
from typing import TypeVar

import gym
import gym_multi_car_racing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm.auto import tqdm

from syllabus.core import (
    PettingZooMultiProcessingSyncWrapper,
    TaskWrapper,
    make_multiprocessing_curriculum,
)
from syllabus.curricula import DomainRandomization
from syllabus.task_space.task_space import TaskSpace

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")

parser = argparse.ArgumentParser(description="Train a PPO agent for the CarRacing-v0")
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    metavar="G",
    help="discount factor (default: 0.99)",
)
parser.add_argument(
    "--action-repeat",
    type=int,
    default=8,
    metavar="N",
    help="repeat action in N frames (default: 8)",
)
parser.add_argument(
    "--img-stack",
    type=int,
    default=4,
    metavar="N",
    help="stack N image in a state (default: 4)",
)
parser.add_argument(
    "--seed", type=int, default=0, metavar="N", help="random seed (default: 0)"
)
parser.add_argument("--render", action="store_true", help="render the environment")
parser.add_argument("--vis", action="store_true", help="use visdom")
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="interval between training status logs (default: 10)",
)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

transition = np.dtype(
    [
        ("s", np.float64, (args.img_stack, 96, 96)),
        ("a", np.float64, (3,)),
        ("a_logp", np.float64),
        ("r", np.float64),
        ("s_", np.float64, (args.img_stack, 96, 96)),
    ]
)


# class Env:
#     """
#     Environment wrapper for CarRacing
#     """

#     def __init__(self):
#         self.env = gym.make("CarRacing-v0")
#         self.env.seed(args.seed)
#         self.reward_threshold = self.env.spec.reward_threshold

#     def reset(self):
#         self.counter = 0
#         self.av_r = self.reward_memory()

#         self.die = False
#         img_rgb = self.env.reset()
#         img_gray = self.rgb2gray(img_rgb)
#         self.stack = [img_gray] * args.img_stack  # four frames for decision
#         return np.array(self.stack)

#     def step(self, action):
#         total_reward = 0
#         for i in range(args.action_repeat):
#             img_rgb, reward, die, _ = self.env.step(action)
#             # don't penalize "die state"
#             if die:
#                 reward += 100
#             # green penalty
#             if np.mean(img_rgb[:, :, 1]) > 185.0:
#                 reward -= 0.05
#             total_reward += reward
#             # if no reward recently, end the episode
#             done = True if self.av_r(reward) <= -0.1 else False
#             if done or die:
#                 break
#         img_gray = self.rgb2gray(img_rgb)
#         self.stack.pop(0)
#         self.stack.append(img_gray)
#         assert len(self.stack) == args.img_stack
#         return np.array(self.stack), total_reward, done, die

#     def render(self, *arg):
#         self.env.render(*arg)

#     @staticmethod
#     def rgb2gray(rgb, norm=True):
#         # rgb image -> gray [0, 1]
#         gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
#         if norm:
#             # normalize
#             gray = gray / 128.0 - 1.0
#         return gray

#     @staticmethod
#     def reward_memory():
#         # record reward for last 100 steps
#         count = 0
#         length = 100
#         history = np.zeros(length)

#         def memory(reward):
#             nonlocal count
#             history[count] = reward
#             count = (count + 1) % length
#             return np.mean(history)

#         return memory


class MultiCarRacingParallelWrapper(TaskWrapper):
    """
    Wrapper ensuring compatibility with the PettingZoo Parallel API.

    Car Racing Environment:
        * Action shape:  ``n_agents`` * `Box([-1. 0. 0.], 1.0, (3,), float32)`
        * Observation shape: ``n_agents`` * `Box(0, 255, (96, 96, 3), uint8)`
        * Done: ``done`` is a single boolean value
        * Info: ``info`` is unused and represented as an empty dictionary
    """

    def __init__(self, n_agents, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_agents = n_agents
        self.task = None
        self.episode_return = 0
        self.task_space = TaskSpace(
            spaces.Box(
                low=np.array([-1.0, 0.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0]),
                shape=(3,),
                dtype=np.float32,
            )
        )
        self.possible_agents = np.arange(
            self.n_agents
        )  # TODO: is this the intended use?

    def _actions_pz_to_np(self, action: dict[AgentID, ActionType]) -> np.ndarray:
        """
        Converts actions defined in PZ format to a numpy array.
        """
        assert action.__len__() == self.n_agents

        action = np.array(list(action.values()))
        assert action.shape == (self.n_agents, 3)
        return action

    def _np_array_to_pz_dict(self, array: np.ndarray) -> dict[int : np.ndarray]:
        """
        Returns a dictionary containing individual observations for each agent.
        """
        out = {}
        for idx, i in enumerate(array):
            out[idx] = i
        return out

    def _singleton_to_pz_dict(self, value: bool) -> dict[int:bool]:
        """
        Broadcasts the `done` and `trunc` flags to dictionaries keyed by agent id.
        """
        return {idx: value for idx in range(self.n_agents)}

    def reset(self) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """
        Resets the environment and returns a dictionary of observations
        keyed by agent ID.
        """
        # TODO: what is the second output (dict[AgentID, dict]])?
        obs = self.env.reset()
        pz_obs = self._np_array_to_pz_dict(obs)

        return pz_obs

    def step(self, action: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        """
        Takes inputs in the PettingZoo (PZ) Parallel API format, performs a step and
        returns outputs in PZ format.
        """
        # convert action to numpy format to perform the env step
        np_action = self._actions_pz_to_np(action)
        obs, rew, term, info = self.env.step(np_action)
        trunc = 0  # there is no `truncated` flag in this environment
        self.task_completion = self._task_completion(obs, rew, term, trunc, info)
        # convert outputs back to PZ format
        obs, rew = tuple(map(self._np_array_to_pz_dict, [obs, rew]))
        term, trunc, info = tuple(
            map(self._singleton_to_pz_dict, [term, trunc, self.task_completion])
        )

        return self.observation(obs), rew, term, trunc, info


class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class Agent:
    """
    Agent for training
    """

    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self):
        self.training_step = 0
        self.net = Net().double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self):
        torch.save(self.net.state_dict(), "param/ppo_net_params.pkl")

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer["s"], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer["a"], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer["r"], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer["s_"], dtype=torch.double).to(device)

        old_a_logp = (
            torch.tensor(self.buffer["a_logp"], dtype=torch.double)
            .to(device)
            .view(-1, 1)
        )

        with torch.no_grad():
            target_v = r + args.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False
            ):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * adv[index]
                )
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2.0 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()


if __name__ == "__main__":
    agent = Agent()
    """ CURRICULUM SETUP """
    n_agents = 2
    env = gym.make(
        "MultiCarRacing-v0",
        num_agents=n_agents,
        direction="CCW",
        use_random_direction=True,
        backwards_flag=True,
        h_ratio=0.25,
        use_ego_color=False,
    )
    env = MultiCarRacingParallelWrapper(env=env, n_agents=n_agents)
    curriculum = DomainRandomization(env.task_space)
    curriculum, task_queue, update_queue = make_multiprocessing_curriculum(curriculum)
    env = PettingZooMultiProcessingSyncWrapper(
        env,
        task_queue,
        update_queue,
        update_on_step=False,
        task_space=env.task_space,
    )

    n_episodes = 10
    n_steps = 100

    training_records = []
    running_score = 0
    state = env.reset()
    for i_ep in tqdm(range(n_episodes)):
        score = 0
        state = env.reset()

        for t in tqdm(range(n_steps)):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(
                action * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])
            )
            if args.render:
                env.render()
            if agent.store((state, action, a_logp, reward, state_)):
                print("updating")
                agent.update()
            score += reward
            state = state_
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01

        if i_ep % args.log_interval == 0:
            print(
                "Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}".format(
                    i_ep, score, running_score
                )
            )
            # agent.save_param()
        if running_score > env.reward_threshold:
            print(
                f"""Solved! Running reward is now {running_score} and the last \\
                      episode runs to"""
            )
            break
