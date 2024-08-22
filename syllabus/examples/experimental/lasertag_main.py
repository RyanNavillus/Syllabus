import argparse
import os
import sys
import time
from typing import Dict, Tuple, TypeVar

import joblib
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from gymnasium import spaces
from plotly.subplots import make_subplots
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

sys.path.append("../../..")
from lasertag import LasertagAdversarial  # noqa: E402
from syllabus.core import (  # noqa: E402
    DualCurriculumWrapper,
    TaskWrapper,
    make_multiprocessing_curriculum,
)

# noqa: E402
from syllabus.curricula import (  # noqa: E402
    CentralizedPrioritizedLevelReplay,
    DomainRandomization,
    FictitiousSelfPlay,
    PrioritizedFictitiousSelfPlay,
    SelfPlay,
)
from syllabus.task_space import TaskSpace  # noqa: E402

ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")
AgentType = TypeVar("AgentType")
EnvTask = TypeVar("EnvTask")
AgentTask = TypeVar("AgentTask")
ObsType = TypeVar("ObsType")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=bool, default=False)
    parser.add_argument("--total-updates", type=int, default=40000)
    parser.add_argument("--rollout_length", type=int, default=256)
    parser.add_argument(
        "--batch-size", type=int, default=32
    )  # TODO: determine correct value

    # PPO args
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=5)

    # curriculum args
    parser.add_argument(
        "--agent-curriculum", type=str, default="SP", choices=["SP", "FSP", "PFSP"]
    )
    parser.add_argument(
        "--env-curriculum", type=str, default="DR", choices=["DR", "PLR"]
    )
    parser.add_argument("--agent-update-frequency", type=int, default=8000)
    # number of opponents in FSP and PFSP
    parser.add_argument("--max-agents", type=int, default=10)
    parser.add_argument("--save-agent-checkpoints", type=bool, default=False)
    parser.add_argument(
        "--checkpoint-frequency", type=int, default=4000
    )  # agent checkpoints every N steps
    parser.add_argument("--n-env-tasks", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="lasertag_",
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
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.n_steps = 0

    def __getattr__(self, name):
        """
        Delegate attribute lookup to the wrapped environment if the attribute
        is not found in the LasertagParallelWrapper instance.
        """
        return getattr(self.env, name)

    def _np_array_to_pz_dict(self, array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary containing individual observations for each agent.
        Assumes that the batch dimension represents individual agents.
        """
        out = {}
        for idx, value in enumerate(array):
            out[self.possible_agents[idx]] = value
        return out

    def _singleton_to_pz_dict(self, value: bool) -> Dict[str, bool]:
        """
        Broadcasts the `done` and `trunc` flags to dictionaries keyed by agent id.
        """
        return {str(agent_index): value for agent_index in range(self.n_agents)}

    def reset(
        self, env_task: int
    ) -> Tuple[Dict[AgentID, ObsType], Dict[AgentID, dict]]:
        """
        Resets the environment and returns a dictionary of observations
        keyed by agent ID.
        """
        self.env.seed(env_task)
        obs = self.env.reset_random()  # random level generation
        pz_obs = self._np_array_to_pz_dict(obs["image"])

        return pz_obs

    def step(
        self, action: Dict[AgentID, ActionType], device: str, agent_task: int
    ) -> Tuple[
        Dict[AgentID, ObsType],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, dict],
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
        self.n_steps += 1

        return self.observation(obs), rew, done, trunc, info


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


agent_curriculums = {
    "SP": SelfPlay,
    "FSP": FictitiousSelfPlay,
    "PFSP": PrioritizedFictitiousSelfPlay,
}
env_curriculums = {
    "DR": DomainRandomization,
    "PLR": CentralizedPrioritizedLevelReplay,
}

if __name__ == "__main__":
    args = parse_args()
    exp_name = f"{args.exp_name}_{args.env_curriculum}_{args.agent_curriculum}"
    run_name = f"lasertag__{exp_name}__seed_{args.seed}__{int(time.time())}"

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
        wandb.run.log_code(os.path.join(args.logging_dir))

        hyperparameters = vars(args)
        html_table = "<table><tr><th>Parameter</th><th>Value</th></tr>"
        for key, value in hyperparameters.items():
            html_table += f"<tr><td>{key}</td><td>{value}</td></tr>"
        wandb.log({"hyperparameters": wandb.Html(html_table)})

    writer = SummaryWriter(
        os.path.join(args.logging_dir, f"{args.logging_dir}/runs/{run_name}")
    )
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    if (
        not os.path.exists(f"{args.logging_dir}/{exp_name}_checkpoints")
        and args.save_agent_checkpoints
    ):
        os.makedirs(f"{args.logging_dir}/{exp_name}_checkpoints", exist_ok=True)

    np.random.seed(args.seed)

    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stack_size = 3
    frame_size = (5, 5)
    n_agents = 2
    num_actions = 5

    agent_curriculum_settings = {
        "device": device,
        "storage_path": f"{args.agent_curriculum}_agents",
        "max_agents": args.max_agents,
        "seed": args.seed,
    }

    env_task_space = TaskSpace(spaces.Discrete(args.n_env_tasks))
    env_curriculum_settings = {
        "DR": {"task_space": env_task_space},
        "PLR": {
            "task_space": env_task_space,
            "num_steps": args.rollout_length,
            "num_processes": 1,  # TODO: modify if using vecenvs
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "task_sampler_kwargs_dict": {"strategy": "value_l1"},
        },
    }

    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=args.epsilon)

    """ ENV SETUP """
    env = LasertagAdversarial(record_video=False)  # 2 agents by default
    env = LasertagParallelWrapper(env=env, n_agents=n_agents)
    agent_curriculum = agent_curriculums[args.agent_curriculum](
        agent=agent, **agent_curriculum_settings
    )

    env_curriculum = env_curriculums[args.env_curriculum](
        **env_curriculum_settings[args.env_curriculum]
    )

    curriculum = DualCurriculumWrapper(
        env=env,
        agent_curriculum=agent_curriculum,
        env_curriculum=env_curriculum,
    )
    mp_curriculum = make_multiprocessing_curriculum(curriculum)

    """ ALGO LOGIC: EPISODE STORAGE"""
    total_episodic_return = 0
    rb_obs = torch.zeros((args.rollout_length, n_agents, stack_size, *frame_size)).to(
        device
    )
    rb_actions = torch.zeros((args.rollout_length, n_agents)).to(device)
    rb_logprobs = torch.zeros((args.rollout_length, n_agents)).to(device)
    rb_rewards = torch.zeros((args.rollout_length, n_agents)).to(device)
    rb_terms = torch.zeros((args.rollout_length, n_agents)).to(device)
    rb_values = torch.zeros((args.rollout_length, n_agents)).to(device)

    agent_tasks, env_tasks, rewards_history = [], [], []
    agent_c_rew, opp_c_rew = 0, 0
    episode, n_learner_wins = 0, 0
    n_updates = 0
    info = {}

    """ TRAINING LOGIC """
    with tqdm(total=args.total_updates) as pbar:
        while n_updates < args.total_updates:
            with torch.no_grad():
                env_task, agent_task = mp_curriculum.sample()

                env_tasks.append(env_task)
                agent_tasks.append(agent_task)

                next_obs = env.reset(env_task)
                total_episodic_return = 0

                for step in range(0, args.rollout_length):
                    joint_obs = batchify(next_obs, device).squeeze()
                    agent_obs, opponent_obs = joint_obs

                    actions, logprobs, _, values = agent.get_action_and_value(
                        agent_obs, flatten_start_dim=0
                    )

                    opponent = mp_curriculum.get_opponent(info.get("agent_id", 0)).to(
                        device
                    )
                    opponent_action, *_ = opponent.get_action_and_value(
                        opponent_obs, flatten_start_dim=0
                    )

                    joint_actions = torch.tensor((actions, opponent_action))
                    next_obs, rewards, dones, truncs, info = env.step(
                        unbatchify(joint_actions, env.possible_agents),
                        device,
                        agent_task,
                    )

                    opp_reward = rewards["agent_1"]
                    if opp_reward != 0:
                        mp_curriculum.update_winrate(info["agent_id"], opp_reward)
                        if opp_reward == -1:
                            n_learner_wins += 1

                    rb_obs[step] = batchify(next_obs, device)
                    rb_rewards[step] = batchify(rewards, device)
                    rb_terms[step] = batchify(dones, device)
                    rb_actions[step] = joint_actions
                    rb_logprobs[step] = logprobs
                    rb_values[step] = values.flatten()

                    total_episodic_return += rb_rewards[step].cpu().numpy()

                    agent_c_rew += rewards["agent_0"]
                    opp_c_rew += rewards["agent_1"]
                    grid_size = env.level[3]["grid_size_selected"]
                    walls_percentage = env.level[3]["clutter_rate_selected"]

                    if any([dones[a] for a in dones]) or any(
                        [truncs[a] for a in truncs]
                    ):
                        episode += 1
                        rewards_history.append(rewards)
                        env_task, agent_task = mp_curriculum.sample()
                        env_tasks.append(env_task)
                        agent_tasks.append(agent_task)

                        writer.add_scalar("charts/grid_size", grid_size, episode)
                        writer.add_scalar(
                            "charts/walls_percentage", walls_percentage, episode
                        )

                        next_obs = env.reset(env_task)

                        # store learner checkpoints
                        if (
                            args.save_agent_checkpoints
                            and n_updates % args.checkpoint_frequency == 0
                        ):
                            print(f"saving checkpoint --{n_updates}")
                            checkpoint_path = (
                                f"{args.logging_dir}/{exp_name}_checkpoints/"
                                f"{mp_curriculum.curriculum.env_curriculum.name}_"
                                f"{mp_curriculum.curriculum.agent_curriculum.name}_{n_updates}"
                                f"_seed_{args.seed}.pkl"
                            )
                            # --- local checkpoint ---
                            joblib.dump(
                                agent,
                                filename=checkpoint_path,
                            )

                            # --- wandb checkpoint ---
                            if args.track:
                                agent_artifact = wandb.Artifact("model", type="model")
                                agent_artifact.add_file(checkpoint_path)
                                wandb.log_artifact(agent_artifact)

                if args.env_curriculum == "PLR":
                    with torch.no_grad():
                        next_value = agent.get_value(
                            torch.tensor(next_obs["agent_0"]).to(device),
                            flatten_start_dim=0,
                        )
                    update = {
                        "update_type": "on_demand",
                        "metrics": {
                            "value": values,
                            "next_value": next_value,
                            "rew": rewards[
                                "agent_0"
                            ],  # TODO: is this the expected use?
                            "dones": dones,
                            "tasks": env_task,
                        },
                    }

            # gae
            with torch.no_grad():
                next_value = agent.get_value(
                    torch.tensor(next_obs["agent_0"]).to(device), flatten_start_dim=0
                )
                rb_advantages = torch.zeros_like(rb_rewards).to(device)
                last_gae_lam = 0
                for t in reversed(range(args.rollout_length - 1)):
                    if t == args.rollout_length - 1:
                        next_non_terminal = 1.0 - rb_terms[t + 1]
                        next_values = next_value
                    else:
                        next_non_terminal = 1.0 - rb_terms[t + 1]
                        next_values = rb_values[t + 1]
                    delta = (
                        rb_rewards[t]
                        + args.gamma * next_values * next_non_terminal
                        - rb_values[t]
                    )
                    rb_advantages[t] = last_gae_lam = (
                        delta
                        + args.gamma
                        * args.gae_lambda
                        * next_non_terminal
                        * last_gae_lam
                    )
                rb_returns = rb_advantages + rb_values

            b_obs = torch.flatten(rb_obs[: args.rollout_length], start_dim=0, end_dim=1)
            b_logprobs = torch.flatten(
                rb_logprobs[: args.rollout_length], start_dim=0, end_dim=1
            )
            b_actions = torch.flatten(
                rb_actions[: args.rollout_length], start_dim=0, end_dim=1
            )
            b_returns = torch.flatten(
                rb_returns[: args.rollout_length], start_dim=0, end_dim=1
            )
            b_values = torch.flatten(
                rb_values[: args.rollout_length], start_dim=0, end_dim=1
            )
            b_advantages = torch.flatten(
                rb_advantages[: args.rollout_length], start_dim=0, end_dim=1
            )

            b_index = np.arange(len(b_obs))
            clip_fracs = []
            for repeat in range(args.epochs):
                np.random.shuffle(b_index)
                for start in range(0, len(b_obs), args.batch_size):
                    # select the indices we want to train on
                    end = start + args.batch_size
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
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    # normalize advantages
                    rb_advantages = b_advantages[batch_index]
                    rb_advantages = (rb_advantages - rb_advantages.mean()) / (
                        rb_advantages.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -b_advantages[batch_index] * ratio
                    pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    value = value.flatten()
                    v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                    v_clipped = b_values[batch_index] + torch.clamp(
                        value - b_values[batch_index],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    writer.add_scalar("losses/value_loss", v_loss.item(), n_updates)
                    writer.add_scalar("losses/policy_loss", pg_loss.item(), n_updates)
                    writer.add_scalar("losses/entropy", entropy_loss.item(), n_updates)
                    writer.add_scalar(
                        "losses/old_approx_kl", old_approx_kl.item(), n_updates
                    )
                    writer.add_scalar("losses/approx_kl", approx_kl.item(), n_updates)
                    n_updates += 1
                    pbar.update(1)

                    # update opponent
                    if args.agent_curriculum in ["FSP", "PFSP"]:
                        if (
                            n_updates % args.agent_update_frequency == 0
                            and episode != 0
                        ):
                            mp_curriculum.update_agent(agent)

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

    if args.track:
        # agent tasks
        fig = px.histogram(agent_tasks, height=400)
        fig.update_layout(bargap=0.2)
        fig.update_layout(showlegend=False)
        wandb.log({"charts/agent_tasks": wandb.Html(plotly.io.to_html(fig))})

        # env tasks
        fig = px.histogram(env_tasks, height=400)
        fig.update_layout(bargap=0.2)
        fig.update_layout(showlegend=False)
        wandb.log({"charts/env_tasks": wandb.Html(plotly.io.to_html(fig))})

        learner_winrate = n_learner_wins / episode
        wandb.run.summary["learner_winrate"] = learner_winrate
        writer.add_scalar("charts/learner_winrate", learner_winrate)

        # agent rewards
        fig = px.line(
            pd.DataFrame(rewards_history).cumsum(),
            title="Agent rewards",
            labels={"index": "Episodes", "value": "Cumulative rewards"},
        )
        wandb.log({"charts/agent_rewards": wandb.Html(plotly.io.to_html(fig))})

        # win rates and replays
        if args.agent_curriculum in ["FSP", "PFSP"]:
            agent_ids = np.arange(agent_curriculum_settings["max_agents"])
            values = list(mp_curriculum.curriculum.agent_curriculum.history.values())
            winrates = [i["winrate"] for i in values]
            n_games = [i["n_games"] for i in values]

            fig = make_subplots(
                rows=2, cols=1, subplot_titles=("Win Rate", "Number of Games")
            )
            fig.add_trace(
                go.Bar(x=agent_ids, y=winrates, name="Win Rate", marker_color="blue"),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(
                    x=agent_ids,
                    y=n_games,
                    name="Number of Games",
                    marker_color="orange",
                ),
                row=2,
                col=1,
            )

            fig.update_yaxes(range=[0, 1], row=1, col=1)
            fig.update_layout(showlegend=False)
            wandb.log({"charts/opponent_winrates": wandb.Html(plotly.io.to_html(fig))})

        writer.close()
