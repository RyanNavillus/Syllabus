import argparse
import os
import sys
import time
from typing import Tuple, TypeVar

import joblib
import numpy as np
import plotly
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from gymnasium import spaces
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

sys.path.append("../")

from lasertag import LasertagAdversarial  # noqa: E402
from syllabus.core import (  # noqa: E402
    Curriculum,
    TaskWrapper,
    make_multiprocessing_curriculum,
)
from syllabus.curricula import DomainRandomization  # noqa: E402
from syllabus.task_space import TaskSpace  # noqa: E402

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")
Agent = TypeVar("Agent")
EnvTask = TypeVar("EnvTask")
AgentTask = TypeVar("AgentTask")


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
        # self.task_space = TaskSpace(spaces.MultiDiscrete(np.array([[2], [5]])))
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]

    def __getattr__(self, name):
        """
        Delegate attribute lookup to the wrapped environment if the attribute
        is not found in the LasertagParallelWrapper instance.
        """
        return getattr(self.env, name)

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

    def reset(
        self, env_task: int
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """
        Resets the environment and returns a dictionary of observations
        keyed by agent ID.
        """
        self.env.seed(env_task)
        obs = self.env.reset_random()  # random level generation
        pz_obs = self._np_array_to_pz_dict(obs["image"])

        return pz_obs

    def step(
        self, action: dict[AgentID, ActionType], device: str, agent_task: int
    ) -> tuple[
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
        info["agent_id"] = agent_task

        return self.observation(obs), rew, done, trunc, info


class FictitiousSelfPlay(Curriculum):
    def __init__(
        self,
        agent,
        storage_path: str,
        max_agents: int,
    ):
        self.name = "FSP"
        self.storage_path = storage_path
        if not os.path.exists(self.storage_path):
            # raise FileNotFoundError(f"Storage path {self.storage_path} not found.")
            os.makedirs(self.storage_path, exist_ok=True)

        self.n_stored_agents = 0
        self.max_agents = max_agents
        self.task_space = TaskSpace(spaces.Discrete(self.max_agents))
        self.update_agent(agent)  # creates the initial opponent

    def update_agent(self, agent):
        """Saves the current agent instance to a pickle file."""
        if self.n_stored_agents < self.max_agents:
            # TODO: define the expected behaviour when the limit is exceeded
            joblib.dump(
                agent,
                filename=f"{self.storage_path}/{self.name}_agent_checkpoint_{self.n_stored_agents}.pkl",
            )
            self.n_stored_agents += 1

    def get_opponent(self, agent_id) -> Agent:
        """Loads an agent from the buffer of saved agents."""
        return joblib.load(
            f"{self.storage_path}/{self.name}_agent_checkpoint_{agent_id}.pkl"
        )

    def sample(self, k=1):
        return np.random.randint(self.n_stored_agents)


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


class DualCurriculumWrapper:
    """Curriculum wrapper containing both an agent and environment-based curriculum."""

    def __init__(
        self,
        env: TaskWrapper,
        env_curriculum: Curriculum,
        agent_curriculum: Curriculum,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.env = env
        self.agent_curriculum = agent_curriculum
        self.env_curriculum = env_curriculum

        self.env_mp_curriculum, self.env_task_queue, self.env_update_queue = (
            make_multiprocessing_curriculum(env_curriculum)
        )
        self.agent_mp_curriculum, self.agent_task_queue, self.agent_update_queue = (
            make_multiprocessing_curriculum(agent_curriculum)
        )
        self.sample()  # initializes env_task and agent_task

    def sample(self) -> Tuple[EnvTask, AgentTask]:
        """Sets new tasks for the environment and agent curricula."""
        self.env_task = self.env_mp_curriculum.sample()
        self.agent_task = self.agent_mp_curriculum.sample()
        return self.env_task, self.agent_task

    def get_opponent(self, agent_task: AgentTask) -> Tuple[Agent, AgentTask]:
        return self.agent_mp_curriculum.curriculum.get_opponent(agent_task)

    def update_agent(self, agent: Agent) -> Agent:
        return self.agent_mp_curriculum.curriculum.update_agent(agent)

    def __getattr__(self, name):
        """Delegate attribute lookup to the curricula if not found."""
        if hasattr(self.env_curriculum, name):
            return getattr(self.env_curriculum, name)
        elif hasattr(self.agent_curriculum, name):
            return getattr(self.agent_curriculum, name)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )


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
    ent_coef = 0.0
    vf_coef = 0.5
    clip_coef = 0.2
    learning_rate = 1e-4
    epsilon = 1e-5
    gamma = 0.995
    gae_lambda = 0.95
    epochs = 5
    batch_size = 32
    stack_size = 3
    frame_size = (5, 5)
    max_cycles = 201  # lasertag has 200 maximum steps by default
    total_episodes = 5000
    n_agents = 2
    num_actions = 5
    max_agents = 10
    fsp_update_frequency = total_episodes / max_agents

    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=epsilon)

    """ ENV SETUP """
    env = LasertagAdversarial(record_video=False)  # 2 agents by default
    env = LasertagParallelWrapper(env=env, n_agents=n_agents)
    agent_curriculum = FictitiousSelfPlay(
        agent=agent, storage_path="fsp_agents", max_agents=max_agents
    )
    observation_size = env.observation_space["image"].shape[1:]
    env_curriculum = DomainRandomization(TaskSpace(spaces.Discrete(200)))
    curriculum = DualCurriculumWrapper(
        env=env,
        agent_curriculum=agent_curriculum,
        env_curriculum=env_curriculum,
    )

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, n_agents, stack_size, *frame_size)).to(device)
    rb_actions = torch.zeros((max_cycles, n_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, n_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, n_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, n_agents)).to(device)
    rb_values = torch.zeros((max_cycles, n_agents)).to(device)

    agent_tasks, env_tasks = [], []
    agent_c_rew, opp_c_rew = 0, 0
    n_ends, n_learner_wins = 0, 0
    info = {}

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in tqdm(range(total_episodes)):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            env_task, agent_task = curriculum.sample()

            env_tasks.append(env_task[0])
            agent_tasks.append(agent_task)

            next_obs = env.reset(env_task)
            # reset the episodic return
            total_episodic_return = 0
            n_steps = 0

            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                joint_obs = batchify(next_obs, device).squeeze()
                agent_obs, opponent_obs = joint_obs

                # get action from the agent and the opponent
                actions, logprobs, _, values = agent.get_action_and_value(
                    agent_obs, flatten_start_dim=0
                )

                opponent = curriculum.get_opponent(info.get("agent_id", 0)).to(device)
                opponent_action, *_ = opponent.get_action_and_value(
                    opponent_obs, flatten_start_dim=0
                )
                # execute the environment and log data
                joint_actions = torch.tensor((actions, opponent_action))
                next_obs, rewards, terms, truncs, info = env.step(
                    unbatchify(joint_actions, env.possible_agents), device, agent_task
                )

                opp_reward = rewards["agent_1"]
                if opp_reward != 0:
                    n_ends += 1
                    if opp_reward == -1:
                        n_learner_wins += 1

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

        with torch.no_grad():
            next_value = agent.get_value(
                torch.tensor(next_obs["agent_0"]), flatten_start_dim=0
            )
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(end_step)):
                if t == end_step - 1:
                    next_non_terminal = 1.0 - rb_terms[t + 1]
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - rb_terms[t + 1]
                    next_values = rb_values[t + 1]
                delta = (
                    rb_rewards[t]
                    + gamma * next_values * next_non_terminal
                    - rb_values[t]
                )
                rb_advantages[t] = last_gae_lam = (
                    delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
                )
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
        for repeat in range(epochs):
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
                rb_advantages = b_advantages[batch_index]
                rb_advantages = (rb_advantages - rb_advantages.mean()) / (
                    rb_advantages.std() + 1e-8
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

        # update opponent
        if episode % fsp_update_frequency == 0 and episode != 0:
            curriculum.update_agent(agent)

        agent_c_rew += rewards["agent_0"]
        opp_c_rew += rewards["agent_1"]
        grid_size = env.level[3]["grid_size_selected"]
        walls_percentage = env.level[3]["clutter_rate_selected"]

        writer.add_scalar("charts/steps_per_ep", end_step, episode)
        writer.add_scalar("charts/agent_reward", agent_c_rew, episode)
        writer.add_scalar("charts/opponent_reward", opp_c_rew, episode)
        writer.add_scalar("charts/grid_size", grid_size, episode)
        writer.add_scalar("charts/walls_percentage", walls_percentage, episode)
        writer.add_scalar("losses/value_loss", v_loss.item(), episode)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), episode)
        writer.add_scalar("losses/entropy", entropy_loss.item(), episode)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), episode)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), episode)

    learner_winrate = n_learner_wins / n_ends
    wandb.run.summary["n_episodes"] = total_episodes
    wandb.run.summary["learner_winrate"] = learner_winrate
    writer.add_scalar("charts/learner_winrate", learner_winrate)

    fig = px.histogram(agent_tasks, height=400)
    fig.update_layout(bargap=0.2, showlegend=False)
    wandb.log({"charts/agent_tasks": wandb.Html(plotly.io.to_html(fig))})

    fig = px.histogram(env_tasks, height=400)
    fig.update_layout(bargap=0.2, showlegend=False)

    wandb.log({"charts/env_tasks": wandb.Html(plotly.io.to_html(fig))})

    writer.close()
