# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy
import json
import os
import queue
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import wandb
from colorama import Fore, Style, init
from gymnasium import spaces
from pufferlib.emulation import PettingZooPufferEnv
from pufferlib.vector import Serial, Multiprocessing
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append("../../..")
from lasertag import LasertagAdversarial  # noqa: E402  # noqa: E402
from syllabus.core import (  # noqa: E402
    DualCurriculumWrapper,
    MultiagentSharedCurriculumWrapper,
    PettingZooSyncWrapper,
    PettingZooTaskWrapper,
    make_multiprocessing_curriculum,
)
from syllabus.curricula import (  # noqa: E402
    CentralPrioritizedLevelReplay,
    DomainRandomization,
    FictitiousSelfPlay,
    PrioritizedFictitiousSelfPlay,
    SelfPlay,
)
from syllabus.task_space.task_space import (  # noqa: E402
    DiscreteTaskSpace,
    MultiDiscreteTaskSpace,
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
    wandb_project_name: str = "syllabus-testing"
    wandb_entity: str = "rpegoud"
    logging_dir: str = "."
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    capture_video: bool = False

    # Experiment setup
    env_id: str = "lasertag"
    total_timesteps: int = int(4e8)
    seed: int = 0

    # Algorithm specific arguments
    gamma: float = 0.995
    gae_lambda: float = 0.95
    anneal_lr: bool = False
    norm_adv: bool = False
    target_kl = None  # type: float
    # Adam args
    adam_lr: float = 1e-4
    adam_eps: float = 1e-5
    # PPO args
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    clip_vloss: bool = True
    max_grad_norm: float = 0.5
    ent_coef: float = 0.0  # TODO: is this supposed to be zero?
    num_workers: int = 32
    num_minibatches: int = 4
    rollout_length: int = 256
    update_epochs: int = 5

    # Curricula
    agent_curriculum: str = "PFSP"
    env_curriculum: str = "DR"
    n_env_tasks: int = 4000
    max_agents: int = 10
    checkpoint_frequency: int = 8000
    smoothing_constant: float = 0.01
    entropy_parameter: float = 2.0


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


def evaluate_agent(agent, eval_env, step):
    print(f"{Fore.YELLOW}{Style.BRIGHT} Evaluating agent ...")

    agent.eval()

    agent_task_indices = np.arange(0, args.num_workers * 2, 2)
    next_obs, _ = eval_env.reset()
    next_obs = next_obs[agent_task_indices]  # filter protagonist agent observations
    eval_episode_rewards = []
    lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_workers, agent.lstm.hidden_size).to(
            device
        ),
        torch.zeros(agent.lstm.num_layers, args.num_workers, agent.lstm.hidden_size).to(
            device
        ),
    )
    next_done = torch.zeros(args.num_workers).to(device)

    while len(eval_episode_rewards) < args.num_workers:
        with torch.no_grad():
            obs = torch.Tensor(next_obs).to(device)
            (
                agent_actions,
                _,
                _,
                _,
                lstm_state,
            ) = selected_opp.get_action_and_value(obs, lstm_state, next_done)

            opponent_actions = torch.randint(0, 5, (args.num_workers,))
            joint_actions = reconstruct_batch(
                agent_actions.cpu(), opponent_actions.cpu(), args.num_workers * 2
            )
            next_obs, rewards, next_done, _, _ = envs.step(joint_actions.numpy())
            next_obs = torch.Tensor(next_obs[agent_task_indices]).to(device)
            next_done = torch.Tensor(next_done[agent_task_indices]).to(device)

            for r in rewards[agent_task_indices]:
                if r != 0:
                    eval_episode_rewards.append(r)

    agent.train()

    winrate = (np.mean(eval_episode_rewards) / 2.0) + 0.5
    print(f"{Fore.YELLOW}{Style.BRIGHT} Average winrate: {winrate:.2f}%")
    if args.track:
        writer.add_scalar("eval/average_winrate_against_random", winrate, step)


class Agent(nn.Module):
    def __init__(self, obs_shape, n_actions) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            self.layer_init(
                nn.Conv2d(
                    obs_shape[0],
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
        self.actor = self.layer_init(nn.Linear(32, n_actions), scale=0.01)
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
        return spaces.Box(
            low=env_space.low[0],
            high=env_space.high[0],
            shape=env_space.shape[1:],
            dtype=env_space.dtype,
        )

    def action_space(self, agent):
        return spaces.Discrete(5)

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
        self, seed: int = None, **kwargs
    ) -> Tuple[Dict[AgentID, ObsType], Dict[AgentID, dict]]:
        """
        Resets the environment and returns a dictionary of observations
        keyed by agent ID.
        """
        if "new_task" in kwargs and kwargs["new_task"] is not None:
            self.task = kwargs.pop("new_task")
            seed = self.task[0]
        self.env.seed(seed)
        obs = self.env.reset_random()  # random level generation
        pz_obs = self._np_array_to_pz_dict(obs["image"])
        infos = {agent: {"solved": 0.0} for agent in self.possible_agents}
        infos["agent_id"] = self.task[1] if self.task is not None else 0
        return pz_obs, infos

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
        infos = {agent: {"solved": i} for agent, i in info.items()}
        infos["agent_id"] = self.task[1]
        self.n_steps += 1
        return self.observation(obs), rew, done, trunc, infos


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


def make_env_fn(components=None):
    def thunk(buf=None):
        env = LasertagAdversarial()
        env = LasertagParallelWrapper(env=env, n_agents=2)
        if components is not None:
            env = PettingZooSyncWrapper(
                env,
                MultiDiscreteTaskSpace([args.n_env_tasks, 1000]),
                components,
                buffer_size=2,
            )
        env = PettingZooPufferEnv(env, buf=buf)
        return env

    return thunk


agent_curriculums = {
    "SP": SelfPlay,
    "FSP": FictitiousSelfPlay,
    "PFSP": PrioritizedFictitiousSelfPlay,
}
env_curriculums = {
    "DR": DomainRandomization,
    "PLR": CentralPrioritizedLevelReplay,
}


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
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

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{Fore.RED}{Style.BRIGHT} Using device: {device}")

    # agent and opponent selection indices
    # joint obs, rewards, dones alternate between agent and opp
    agent_indices = torch.arange(0, args.num_workers * 2, 2)
    opponent_indices = torch.arange(1, args.num_workers * 2, 2)

    # agent setup
    exemplar_env = make_env_fn(None)()
    agent = Agent(
        exemplar_env.observation_space("agent_0").shape,
        exemplar_env.action_space("agent_0").n,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.adam_lr, eps=args.adam_eps)

    # curriculum setup
    env_task_space = DiscreteTaskSpace(spaces.Discrete(args.n_env_tasks))
    env_curriculum_settings = {
        "DR": {"task_space": env_task_space},
        "PLR": {
            "task_space": env_task_space,
            "num_steps": args.rollout_length,
            "num_processes": args.num_workers,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "task_sampler_kwargs_dict": {"strategy": "value_l1"},
        },
    }
    agent_curriculum_settings = {
        "device": device,
    }

    if args.agent_curriculum in ["PFSP", "FSP"]:
        agent_curriculum_settings["storage_path"] = f"{args.agent_curriculum}_agents"
        agent_curriculum_settings["max_agents"] = args.max_agents
        agent_curriculum_settings["seed"] = args.seed

    if args.agent_curriculum == "PFSP":
        agent_curriculum_settings["smoothing_constant"] = args.smoothing_constant
        agent_curriculum_settings["entropy_parameter"] = args.entropy_parameter

    env_curriculum = env_curriculums[args.env_curriculum](
        **env_curriculum_settings[args.env_curriculum]
    )
    env_curriculum = MultiagentSharedCurriculumWrapper(
        env_curriculum, exemplar_env.possible_agents
    )
    agent_curriculum = agent_curriculums[args.agent_curriculum](
        env_task_space, agent, **agent_curriculum_settings
    )
    curriculum = DualCurriculumWrapper(
        env_curriculum,
        agent_curriculum,
    )
    curriculum = make_multiprocessing_curriculum(curriculum, start=False)
    curriculum.add_agent(agent)

    # env setup
    envs = Multiprocessing(
        [make_env_fn(curriculum.components) for _ in range(args.num_workers)],
        [() for _ in range(args.num_workers)],
        [{} for _ in range(args.num_workers)],
        args.num_workers,
        overwork=True,
    )
    envs.is_vector_env = True

    eval_env = Multiprocessing(
        [make_env_fn(None) for _ in range(args.num_workers)],
        [() for _ in range(args.num_workers)],
        [{} for _ in range(args.num_workers)],
        args.num_workers,
        overwork=True,
    )
    eval_env.is_vector_env = True
    curriculum.start()

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

    num_updates = int(args.total_timesteps // args.batch_size)
    env_tasks, agent_tasks = [], np.zeros(args.num_workers)
    latest_agent_task = 0

    cumulative_rewards = np.zeros(args.num_workers * 2)
    next_obs, info = envs.reset()

    with tqdm(total=num_updates) as pbar:
        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.adam_lr
                optimizer.param_groups[0]["lr"] = lrnow

            initial_lstm_state = (lstm_state[0].clone(), lstm_state[1].clone())
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(args.num_workers).to(device)
            # task = 0

            for step in range(0, args.rollout_length):
                global_step += 1 * args.num_workers * 2
                obs[step] = next_obs[agent_indices]
                dones[step] = next_done

                # action selection
                with torch.no_grad():
                    agent_obs, opp_obs = split_batch(next_obs)

                    for task in set(agent_tasks):
                        # iterate over agent_tasks
                        selected_opp = curriculum.get_agent(int(task), agent)

                        # create batches for each task
                        agent_task_indices = np.where(agent_tasks == task)[0]
                        next_done_batch = next_done[agent_task_indices]
                        agent_obs_batch = agent_obs[agent_task_indices]
                        opp_obs_batch = opp_obs[agent_task_indices]

                        lstm_state_batch = (
                            lstm_state[0][:, agent_task_indices],
                            lstm_state[1][:, agent_task_indices],
                        )

                        lstm_state_opp_batch = (
                            lstm_state_opp[0][:, agent_task_indices],
                            lstm_state_opp[1][:, agent_task_indices],
                        )

                        # initialize
                        agent_actions = torch.zeros(
                            args.num_workers, dtype=torch.int32
                        ).to(device)
                        opp_actions = torch.zeros(
                            args.num_workers, dtype=torch.int32
                        ).to(device)
                        agent_value = torch.zeros(args.num_workers).to(device)
                        logprob = torch.zeros(args.num_workers).to(device)

                        # learner action selection
                        (
                            agent_actions_batch,
                            logprob_batch,
                            _,
                            agent_value_batch,
                            new_lstm_state_batch,
                        ) = agent.get_action_and_value(
                            agent_obs_batch, lstm_state_batch, next_done_batch
                        )

                        # opponent action selection
                        (
                            opp_actions_batch,
                            _,
                            _,
                            _,
                            new_lstm_state_opp_batch,
                        ) = selected_opp.get_action_and_value(
                            opp_obs_batch, lstm_state_opp_batch, next_done_batch
                        )

                        # reconstruct data from batches
                        agent_obs[agent_task_indices] = agent_obs_batch
                        agent_value[agent_task_indices] = (
                            agent_value_batch.flatten().float()
                        )
                        agent_actions[agent_task_indices] = agent_actions_batch.int()
                        opp_actions[agent_task_indices] = opp_actions_batch.int()
                        logprob[agent_task_indices] = logprob_batch.float()
                        next_done[agent_task_indices] = next_done_batch.float()

                        # reconstruct the LSTM state
                        for i, agent_task_idx in enumerate(agent_task_indices):
                            lstm_state[0][:, agent_task_idx] = new_lstm_state_batch[0][
                                :, i
                            ]
                            lstm_state[1][:, agent_task_idx] = new_lstm_state_batch[1][
                                :, i
                            ]
                            lstm_state_opp[0][:, agent_task_idx] = (
                                new_lstm_state_opp_batch[0][:, i]
                            )
                            lstm_state_opp[1][:, agent_task_idx] = (
                                new_lstm_state_opp_batch[1][:, i]
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
                agent_tasks = np.array([i["agent_id"] for i in info])

                rewards[step] = torch.tensor(reward[agent_indices]).to(device).view(-1)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.Tensor(next_done[agent_indices]).to(device)

                actions[step] = agent_actions.cpu()
                logprobs[step] = logprob.cpu()

                cumulative_rewards = cumulative_rewards + reward

                if args.track:
                    writer.add_scalar(
                        "train/average_return",
                        np.mean(cumulative_rewards),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/std_return",
                        np.std(cumulative_rewards),
                        global_step,
                    )
                # Syllabus curriculum update
                if args.curriculum and args.curriculum_method == "centralplr":
                    next_value = None
                    if step == args.num_steps - 1:
                        # TODO: Get value prediction from current agents
                        with torch.no_grad():
                            next_value = agent.get_value(next_obs)

                    current_tasks = np.array([i["task"] for i in info])

                    plr_update = {
                        "value": values[step],
                        "next_value": next_value,
                        "rew": rewards[step],
                        "dones": next_done,
                        "tasks": current_tasks,
                    }
                    curriculum.update(plr_update)

                if args.agent_curriculum == "PFSP" and any(next_done.cpu().numpy()):
                    agent_reward = reward[agent_indices]
                    agents_to_update = np.where(agent_reward != 0)[0]
                    for agent_to_update in agents_to_update:
                        opponent_id = agent_tasks[agent_to_update]
                        agent_curriculum.update_winrate(
                            opponent_id, agent_reward[agent_to_update]
                        )
                    # print(agent_curriculum.winrate_buffer)

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

            curriculum.add_agent(agent)

            evaluate_agent(agent, eval_env, step=global_step)

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            if args.track:
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar("losses/explained_variance", explained_var, global_step)

            print(
                f"{Fore.WHITE}{Style.BRIGHT} Steps per second:",
                int(global_step / (time.time() - start_time)),
            )
            pbar.update(1)

    envs.close()