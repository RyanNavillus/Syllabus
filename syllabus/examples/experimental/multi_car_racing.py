# flake8: noqa: F401
from typing import TypeVar

import gym
import gym_multi_car_racing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.normal import Normal
from tqdm.auto import tqdm

from syllabus.core import TaskWrapper, make_multiprocessing_curriculum
from syllabus.curricula import DomainRandomization
from syllabus.task_space import TaskSpace

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.observation_shape = (3, 96, 96)
        self.action_shape = (3,)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(self.observation_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(self.observation_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(self.action_shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.action_shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.float().reshape(x.size(0), -1)

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    obs = obs.transpose(0, -1, 1, 2)
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    x = np.stack([x[a] for a in x], axis=0)
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x


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


if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    stack_size = 3
    frame_size = (96, 96)
    action_size = 3
    max_cycles = 125
    total_episodes = 100

    # PLR Params
    num_steps = 128

    """ ENV SETUP """
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

    """ CURRICULUM SETUP """
    env = MultiCarRacingParallelWrapper(env=env, n_agents=n_agents)
    curriculum = DomainRandomization(env.task_space)
    curriculum, task_queue, update_queue = make_multiprocessing_curriculum(curriculum)

    """ LEARNER SETUP """
    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, n_agents, stack_size, *frame_size)).to(device)
    rb_actions = torch.zeros((max_cycles, n_agents, action_size)).to(device)
    rb_logprobs = torch.zeros((max_cycles, n_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, n_agents)).to(device)
    rb_dones = torch.zeros((max_cycles, n_agents)).to(device)
    rb_values = torch.zeros((max_cycles, n_agents)).to(device)

    done = {i: False for i in range(n_agents)}
    total_reward = {i: 0 for i in range(n_agents)}
    np.random.seed(0)

    """ TRAINING LOGIC """
    # train for n number of episodes
    global_cycles = 0
    for episode in tqdm(range(total_episodes)):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs = env.reset()
            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            for step in range(0, max_cycles):
                global_cycles += 1
                # rollover the observation
                obs = batchify_obs(next_obs, device)
                # get action from the agent
                actions, logprobs, entropy, values = agent.get_action_and_value(obs)

                # execute the environment and log data
                next_obs, rewards, trunc, dones, infos = env.step(
                    unbatchify(actions, env)
                )

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_dones[step] = batchify(dones, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # Update curriculum
                # TODO: adapt to DR
                if global_cycles % num_steps == 0:
                    update = {
                        "update_type": "on_demand",
                        "metrics": {
                            "action_log_dist": logprobs,
                            "value": values,
                            "next_value": (
                                agent.get_value(next_obs)
                                if step == num_steps - 1
                                else None
                            ),
                            "rew": rb_rewards[step],
                            "masks": torch.Tensor(1 - np.array(list(dones.values()))),
                            "tasks": [env.unwrapped.task],
                        },
                    }
                    curriculum.update_curriculum(update)

                # if we reach the end of the episode
                if any([dones[a] for a in dones]):
                    print(f"Breaking early at step {step} due to done signal.")
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_dones[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # TODO: at this step, `end_step` is still 0, causing b_obs to be empty
        # and the `repeat` loop not running as range(0, len(b_obs), batch_size) = 0
        # maybe set end_step to the current number of steps in the episode even if there
        # is no positive done flag ?

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                print(start)
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index], b_actions.long()[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"Training episode {episode}")
        print(f"Episodic Return: {np.mean(total_episodic_return)}")
        print(f"Episode Length: {end_step}")
        print("")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(clip_fracs)}")
        print(f"Explained Variance: {explained_var.item()}")
        print("\n-------------------------------------------\n")

    # ----- Dummy test loop -----
    # while not all(done.values()):
    # for episodes in tqdm(range(1)):  # testing with 5 truncated episodes
    #     obs = env.reset()
    #     for steps in range(100):
    #         action = np.random.normal(0, 1, (2, 3))
    #         pz_action = {i: action[i] for i in range(n_agents)}
    #         obs, reward, done, trunc, info = env.step(pz_action)
    #         for agent in range(n_agents):
    #             total_reward[agent] += reward[agent]
    #         env.render()

    # print("individual scores:", total_reward)
    # print(reward, done, trunc, info)
