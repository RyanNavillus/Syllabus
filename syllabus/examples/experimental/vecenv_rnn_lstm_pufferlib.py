# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy
import json
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple, TypeVar

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from colorama import Fore, Style, init
from pufferlib.emulation import PettingZooPufferEnv
from pufferlib.vector import Serial
from torch.distributions.categorical import Categorical
from tqdm import tqdm

sys.path.append("../../..")
from lasertag import LasertagAdversarial  # noqa: E402
from syllabus.core import PettingZooTaskWrapper  # noqa: E402
from syllabus.curricula import (  # noqa: E402
    CentralizedPrioritizedLevelReplay,
    DomainRandomization,
    FictitiousSelfPlay,
    PrioritizedFictitiousSelfPlay,
    SelfPlay,
)

ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")
AgentType = TypeVar("AgentType")
EnvTask = TypeVar("EnvTask")
AgentTask = TypeVar("AgentTask")
ObsType = TypeVar("ObsType")


@dataclass
class Config:
    # Experiment configuration
    exp_name: str = "lasertag_"
    wandb_project_name: str = "syllabus"
    wandb_entity: str = "rpegoud"
    logging_dir: str = "."
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    capture_video: bool = False

    # Experiment setup
    env_id: str = "lasertag"
    total_timesteps: int = int(1e5)
    seed: int = 0

    # Algorithm specific arguments
    gamma: float = 0.995
    gae_lambda: float = 0.95
    anneal_lr: bool = True
    norm_adv: bool = True
    target_kl = None  # type: float
    # Adam args
    adam_lr: float = 1e-4
    adam_eps: float = 1e-5
    # PPO args
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    clip_vloss: bool = True
    max_grad_norm: float = 0.5
    ent_coef: float = 0.0
    num_workers: int = 8
    num_minibatches: int = 4
    rollout_length: int = 256
    update_epochs: int = 5
    # Curricula
    agent_curriculum = "SP"
    env_curriculum = "DR"
    n_env_tasks = 4000
    max_agents = 10
    agent_update_frequency = 8000
    checkpoint_frequency = 4000


def parse_args() -> Config:
    init(autoreset=True)  # Initialize colorama
    config = tyro.cli(Config)
    config.batch_size = int(config.num_workers * config.rollout_length)
    print(
        f"{Fore.RED}{Style.BRIGHT} Setting BATCH_SIZE to: NUM_WORKERS * ROLLOUT_LENGTH"
        f"= {config.batch_size}"
    )
    config.minibatch_size = int(config.batch_size // config.num_minibatches)
    print(
        f"{Fore.RED}{Style.BRIGHT} Setting MINIBATCH_SIZE to :"
        f"BATCH_SIZE * NUM_MINIBATCHES= {config.minibatch_size}"
    )

    # log config
    config_dict = config.__dict__
    print(f"{Fore.GREEN}{Style.BRIGHT} Config:")
    print(
        f"{Fore.GREEN}{Style.BRIGHT}Hyperparameters:"
        f"{Style.NORMAL}{json.dumps(config_dict, sort_keys=True, indent=4)}{Style.RESET_ALL}"
    )

    print(
        f"{Fore.BLUE}{Style.BRIGHT} RUNNING PPO ON {config.env_id} "
        f"FOR {config.total_timesteps} TIMESTEPS ..."
    )
    return config


def log_rewards(r_batch: torch.tensor, run, step: int) -> None:
    r_logs = {f"agent_{agent_idx}": reward for agent_idx, reward in enumerate(r_batch)}

    run["charts/cumulative_rewards_per_update"].append(r_logs, step=step)


class Agent(nn.Module):
    def __init__(self, envs) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            self.layer_init(
                nn.Conv2d(
                    envs.single_observation_space.shape[0],
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                )
            ),
            nn.Flatten(),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=16 * 3 * 3, hidden_size=256, batch_first=True)
        self.lstm_init()

        self.mlp = nn.Sequential(
            nn.ReLU(),
            self.layer_init(nn.Linear(256, 32)),
            nn.ReLU(),
        )
        self.actor = self.layer_init(
            nn.Linear(32, envs.single_action_space.n), scale=0.01
        )
        self.critic = self.layer_init(nn.Linear(32, 1), scale=1)

    def get_states(self, x, lstm_states, done):
        batch_size = x.size(0)  # x shape: (num_envs, *obs_shape)
        hidden = self.conv(x / 255.0)
        hidden = hidden.view(batch_size, 1, -1)  # shape: (num_envs, 1, n_features)

        # shape: (num_layers, batch_size, hidden_size)
        h_states, c_states = lstm_states

        hidden, (new_h_states, new_c_states) = self.lstm(hidden, (h_states, c_states))

        # Reset LSTM state if done
        done = done.view(1, -1, 1)  # (shape: 1, num_envs, 1)
        new_h_states = new_h_states * (1 - done)
        new_c_states = new_c_states * (1 - done)

        hidden = hidden.squeeze(1)
        new_lstm_states = (new_h_states, new_c_states)

        return hidden, new_lstm_states

    def get_value(self, x, lstm_states, done):
        hidden, _ = self.get_states(x, lstm_states, done)
        hidden = self.mlp(hidden)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_states, done, action=None):
        hidden, new_lstm_states = self.get_states(x, lstm_states, done)
        hidden = self.mlp(hidden)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(hidden),
            new_lstm_states,
        )

    def layer_init(self, layer, scale=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, scale)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def lstm_init(self):
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)


def batchify(x, device=None):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x)

    if device is not None:
        x = x.to(device)

    return x


def unbatchify(x, possible_agents: np.ndarray):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {agent: x[idx] for idx, agent in enumerate(possible_agents)}

    return x


class LasertagParallelWrapper(PettingZooTaskWrapper):
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
        self.env.agents = self.possible_agents
        self.n_steps = 0
        self.env.render_mode = "human"

    def observation_space(self, agent):
        env_space = self.env.observation_space["image"]
        # Remove agent dimension
        return gymnasium.spaces.Box(
            low=env_space.low[0],
            high=env_space.high[0],
            shape=env_space.shape[1:],
            dtype=env_space.dtype,
        )

    def action_space(self, agent):
        return gymnasium.spaces.Discrete(5)

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
        return {agent: value for agent in self.agents}

    def reset(
        self, seed: int = None
    ) -> Tuple[Dict[AgentID, ObsType], Dict[AgentID, dict]]:
        """
        Resets the environment and returns a dictionary of observations
        keyed by agent ID.
        """
        self.env.seed(seed)
        obs = self.env.reset_random()  # random level generation
        pz_obs = self._np_array_to_pz_dict(obs["image"])
        return pz_obs, {}

    def step(
        self,
        action: Dict[AgentID, ActionType],
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
        action = batchify(action)
        obs, rew, done, info = self.env.step(action)
        obs = obs["image"]
        trunc = False  # there is no `truncated` flag in this environment
        self.task_completion = self._task_completion(obs, rew, done, trunc, info)
        # convert outputs back to PZ format
        obs, rew = map(self._np_array_to_pz_dict, [obs, rew])
        done, trunc, info = map(
            self._singleton_to_pz_dict, [done, trunc, self.task_completion]
        )
        # info["agent_id"] = agent_task
        self.n_steps += 1
        return self.observation(obs), rew, done, trunc, info


def split_batch(joint_obs: torch.Tensor) -> Tuple[torch.Tensor]:
    """Splits a batch of joint data in agent and opponent data."""
    assert (
        joint_obs.shape[0] == args.num_workers * 2
    ), f"Expected shape {args.num_workers * 2}, got: {joint_obs.shape[0]}"

    agent_indices = [i for i in np.arange(0, args.num_workers * 2, 2)]
    opp_indices = [i for i in np.arange(1, args.num_workers * 2, 2)]

    agent_data = joint_obs[agent_indices]
    opp_data = joint_obs[opp_indices]

    return agent_data, opp_data


def reconstruct_batch(
    agent_data: torch.Tensor, opponent_data: torch.Tensor, size: int
) -> torch.Tensor:
    """Reconstructs a batch of joint data from agent and opponent data"""
    batch = torch.zeros(size, dtype=agent_data.dtype)
    batch[np.arange(0, size, 2)] = agent_data  # even indices = agent
    batch[np.arange(1, size, 2)] = opponent_data  # odd indices = opponent
    return batch


def make_env():
    env = LasertagAdversarial()
    env = LasertagParallelWrapper(env=env, n_agents=2)
    env = PettingZooPufferEnv(env)
    return env


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
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import neptune
        from dotenv import dotenv_values

        env_variables = dotenv_values("credentials.env")
        run = neptune.init_run(
            project="rpegoud/syllabus", api_token=env_variables["NEPTUNE_API_KEY"]
        )
        run["parameters"] = args.__dict__

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # agent and opponent selection indices
    # joint obs, rewards, dones alternate between agent and opp
    agent_indices = torch.arange(0, args.num_workers * 2, 2)
    opponent_indices = torch.arange(1, args.num_workers * 2, 2)

    # env setup
    envs = [make_env for _ in range(args.num_workers)]
    envs = Serial(
        envs,
        [() for _ in range(args.num_workers)],
        [{} for _ in range(args.num_workers)],
        args.num_workers,
    )
    envs.is_vector_env = True

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.adam_lr, eps=args.adam_eps)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.rollout_length, args.num_workers) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.rollout_length, args.num_workers) + envs.single_action_space.shape
    ).to(device)
    dones = torch.zeros((args.rollout_length, args.num_workers)).to(device)
    logprobs = torch.zeros((args.rollout_length, args.num_workers)).to(device)
    rewards = torch.zeros((args.rollout_length, args.num_workers)).to(device)
    values = torch.zeros((args.rollout_length, args.num_workers)).to(device)

    lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_workers, agent.lstm.hidden_size).to(
            device
        ),
        torch.zeros(agent.lstm.num_layers, args.num_workers, agent.lstm.hidden_size).to(
            device
        ),
    )
    lstm_state_opp = (
        torch.zeros(agent.lstm.num_layers, args.num_workers, agent.lstm.hidden_size).to(
            device
        ),
        torch.zeros(agent.lstm.num_layers, args.num_workers, agent.lstm.hidden_size).to(
            device
        ),
    )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_workers).to(device)
    num_updates = int(args.total_timesteps // args.batch_size)

    cumulative_rewards = np.zeros(args.num_workers * 2)

    with tqdm(total=num_updates) as pbar:
        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.adam_lr
                optimizer.param_groups[0]["lr"] = lrnow

            initial_lstm_state = (lstm_state[0].clone(), lstm_state[1].clone())

            for step in range(0, args.rollout_length):
                global_step += 1 * args.num_workers * 2
                obs[step] = next_obs[agent_indices]
                dones[step] = next_done

                # action selection
                with torch.no_grad():
                    agent_obs, opp_obs = split_batch(next_obs)

                    agent_actions, logprob, _, agent_value, lstm_state = (
                        agent.get_action_and_value(agent_obs, lstm_state, next_done)
                    )

                    opp_actions, opp_logprob, _, opp_value, lstm_state_opp = (
                        agent.get_action_and_value(  # TODO: add opponent action
                            opp_obs, lstm_state_opp, next_done
                        )
                    )

                    values[step] = agent_value.flatten().cpu()

                joint_actions = reconstruct_batch(
                    agent_actions.cpu(),
                    opp_actions.cpu(),
                    size=args.num_workers * 2,
                )
                next_obs, reward, next_done, trunc, info = envs.step(
                    joint_actions.numpy()
                )

                rewards[step] = torch.tensor(reward[agent_indices]).to(device).view(-1)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.Tensor(next_done[agent_indices]).to(device)

                actions[step] = agent_actions.cpu()
                logprobs[step] = logprob.cpu()

                cumulative_rewards = cumulative_rewards + reward
                if args.track:
                    log_rewards(cumulative_rewards, run, global_step)

            # generalized advantage estimation (for the learning agent only)
            with torch.no_grad():
                agent_obs, opp_obs = split_batch(next_obs)
                next_value = agent.get_value(agent_obs, lstm_state, next_done).reshape(
                    1, -1
                )
                advantages = torch.zeros(args.rollout_length, args.num_workers).to(
                    device
                )

                lastgaelam = 0
                for t in reversed(range(args.rollout_length)):
                    if t == args.rollout_length - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_dones = dones.reshape((-1,))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    batch_index = b_inds[start:end]

                    b_lstm_state = (
                        initial_lstm_state[0].repeat(
                            1, args.minibatch_size // args.num_workers, 1
                        ),
                        initial_lstm_state[1].repeat(
                            1, args.minibatch_size // args.num_workers, 1
                        ),
                    )  # TODO: instead of a common initialization for each batch item,
                    # retrieve the training lstm state related to the item?

                    _, newlogprob, entropy, newvalue, b_lstm_state = (
                        agent.get_action_and_value(
                            b_obs[batch_index],
                            b_lstm_state,
                            b_dones[batch_index],
                            b_actions.long()[batch_index],
                        )
                    )
                    logratio = newlogprob - b_logprobs[batch_index]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[batch_index]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[batch_index]) ** 2
                        v_clipped = b_values[batch_index] + torch.clamp(
                            newvalue - b_values[batch_index],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[batch_index]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            if args.track:
                run["charts/learning_rate"].append(
                    optimizer.param_groups[0]["lr"], step=global_step
                )
                run["losses/value_loss"].append(v_loss.item(), step=global_step)
                run["losses/policy_loss"].append(pg_loss.item(), step=global_step)
                run["losses/entropy"].append(entropy_loss.item(), step=global_step)
                run["losses/old_approx_kl"].append(
                    old_approx_kl.item(), step=global_step
                )
                run["losses/approx_kl"].append(approx_kl.item(), step=global_step)
                run["losses/clipfrac"].append(np.mean(clipfracs), step=global_step)
                run["losses/explained_variance"].append(explained_var, step=global_step)

            print(
                f"{Fore.WHITE}{Style.BRIGHT} Steps per second:",
                int(global_step / (time.time() - start_time)),
            )
            pbar.update(1)

    envs.close()
    if args.track():
        run.stop()
