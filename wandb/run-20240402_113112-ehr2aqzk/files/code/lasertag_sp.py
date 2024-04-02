import argparse
import os
import time
from copy import deepcopy
from typing import TypeVar

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import wandb
from lasertag import LasertagCorridor2
from syllabus.core import Curriculum, TaskWrapper
from syllabus.task_space import TaskSpace

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")
Agent = TypeVar("Agent")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=bool, default=False)
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="syllabus",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="rpegoud",
        help="the entity (team) of wandb's project",
    )
    parser.add_argument(
        "--logging-dir",
        type=str,
        default=".",
        help="the base directory for logging and wandb storage.",
    )

    args = parser.parse_args()
    return args


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, possible_agents: np.ndarray):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {agent: x[idx] for idx, agent in enumerate(possible_agents)}

    return x


class LasertagParallelWrapper(TaskWrapper):
    """
    Wrapper ensuring compatibility with the PettingZoo Parallel API.

    Lasertag Environment:
        * Action shape:  `n_agents` * `Discrete(5)`
        * Observation shape: Dict('image': Box(0, 255, (`n_agents`, 3, 5, 5), uint8))
    """

    def __init__(self, n_agents, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_agents = n_agents
        self.task = None
        self.episode_return = 0
        self.task_space = TaskSpace(spaces.MultiDiscrete(np.array([[2], [5]])))
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]

    def _np_array_to_pz_dict(self, array: np.ndarray) -> dict[str : np.ndarray]:
        """
        Returns a dictionary containing individual observations for each agent.
        Assumes that the batch dimension represents individual agents.
        """
        out = {}
        for idx, value in enumerate(array):
            out[self.possible_agents[idx]] = value
        return out

    def _singleton_to_pz_dict(self, value: bool) -> dict[str:bool]:
        """
        Broadcasts the `done` and `trunc` flags to dictionaries keyed by agent id.
        """
        return {str(agent_index): value for agent_index in range(self.n_agents)}

    def reset(self) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """
        Resets the environment and returns a dictionary of observations
        keyed by agent ID.
        """
        obs = self.env.reset()
        pz_obs = self._np_array_to_pz_dict(obs["image"])

        return pz_obs

    def step(self, action: dict[AgentID, ActionType], device: str) -> tuple[
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
        action = batchify(action, device)
        obs, rew, done, info = self.env.step(action)
        obs = obs["image"]
        trunc = False  # there is no `truncated` flag in this environment
        self.task_completion = self._task_completion(obs, rew, done, trunc, info)
        # convert outputs back to PZ format
        obs, rew = map(self._np_array_to_pz_dict, [obs, rew])
        done, trunc, info = map(
            self._singleton_to_pz_dict, [done, trunc, self.task_completion]
        )

        return self.observation(obs), rew, done, trunc, info


class SelfPlay(Curriculum):
    def __init__(self, agent, device: str, store_agents_on_cpu: bool = False):
        self.store_agents_on_cpu = store_agents_on_cpu
        self.storage_device = "cpu" if self.store_agents_on_cpu else device
        self.agent = deepcopy(agent).to(self.storage_device)

    def update_agent(self, agent) -> Agent:
        self.agent = deepcopy(agent).to(self.storage_device)

    def get_opponent(self, agent_id) -> tuple[Agent, int]:
        assert (
            agent_id == 0
        ), f"Self play only tracks the current agent. Expected agent id 0, got {agent_id}"
        return self.agent, 0

    def sample(self, k=1):
        return 0


class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Linear(3 * 5 * 5, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x, flatten_start_dim=1):
        x = torch.flatten(x, start_dim=flatten_start_dim)
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None, flatten_start_dim=1):
        x = torch.flatten(x, start_dim=flatten_start_dim)
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"lasertag__{args.exp_name}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            dir=args.logging_dir,
        )
        wandb.run.log_code(os.path.join(args.logging_dir, "/syllabus/examples"))

    writer = SummaryWriter(os.path.join(args.logging_dir, "./runs/{run_name}"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    stack_size = 3
    frame_size = (5, 5)
    max_cycles = 201  # lasertag has 200 maximum steps by default
    total_episodes = 500
    n_agents = 2
    num_actions = 5

    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)

    """ ENV SETUP """
    env = LasertagCorridor2(record_video=False)  # 2 agents by default
    env = LasertagParallelWrapper(env=env, n_agents=n_agents)
    curriculum = SelfPlay(agent=agent, device=device, store_agents_on_cpu=True)
    observation_size = env.observation_space["image"].shape[1:]

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, n_agents, stack_size, *frame_size)).to(device)
    rb_actions = torch.zeros((max_cycles, n_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, n_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, n_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, n_agents)).to(device)
    rb_values = torch.zeros((max_cycles, n_agents)).to(device)

    losses, episode_rewards = [], []
    agent_c_rew, opp_c_rew = 0, 0

    """ TRAINING LOGIC """

    # train for n number of episodes
    for episode in tqdm(range(total_episodes)):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs = env.reset()  # removed seed=None and info
            # reset the episodic return
            total_episodic_return = 0
            n_steps = 0

            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                joint_obs = batchify(next_obs, device)
                agent_obs, opponent_obs = joint_obs

                # get action from the agent and the opponent
                actions, logprobs, _, values = agent.get_action_and_value(
                    agent_obs, flatten_start_dim=0
                )

                opponent, opponent_id = curriculum.get_opponent(0)
                opponent_action, *_ = opponent.get_action_and_value(
                    opponent_obs, flatten_start_dim=0
                )
                # execute the environment and log data
                joint_actions = torch.tensor((actions, opponent_action))
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(joint_actions, env.possible_agents), device
                )
                episode_rewards.append(rewards)

                # add to episode storage
                rb_obs[step] = batchify(next_obs, device)
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = joint_actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values
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

                # normalize advantages
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
                losses.append(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # update opponent
        curriculum.update_agent(agent)

        agent_c_rew += rewards["agent_0"]
        opp_c_rew += rewards["agent_1"]
        writer.add_scalar("charts/agent_reward", agent_c_rew, episode)
        writer.add_scalar("charts/opponent_reward", opp_c_rew, episode)
        writer.add_scalar("charts/steps_per_ep", end_step, episode)
        writer.add_scalar("losses/value_loss", v_loss.item(), episode)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), episode)
        writer.add_scalar("losses/entropy", entropy_loss.item(), episode)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), episode)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), episode)

    rewards = pd.DataFrame(
        list(map(lambda x: batchify(x, device).numpy(), episode_rewards))
    )
    fig = px.line(rewards.cumsum(), title="Cumulative rewards per step")
    fig.show()
