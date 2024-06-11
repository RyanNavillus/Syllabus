# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy
import importlib
import random
import time
from dataclasses import dataclass

import gymnasium
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from colorama import Fore, Style, init
from torch.distributions.categorical import Categorical
from tqdm import tqdm


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
    env_id: str = "pong_v3"
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
    num_workers: int = 32
    num_minibatches: int = 4
    rollout_length: int = 256
    update_epochs: int = 5


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
        f"{Fore.RED}{Style.BRIGHT} Setting MINIBATCH_SIZE to:"
        f"BATCH_SIZE * NUM_MINIBATCHES= {config.minibatch_size}"
    )
    log_config(config)
    print(
        f"{Fore.BLUE}{Style.BRIGHT} RUNNING PPO ON {config.env_id} "
        f"FOR {config.total_timesteps} TIMESTEPS ..."
    )
    return config


def log_config(config: Config) -> None:
    config_dict = config.__dict__
    print(f"{Fore.GREEN}{Style.BRIGHT} Config:")
    log_str = " \n".join(
        [
            f"{key.replace('_', ' ').capitalize()}: {Fore.GREEN}{value}{Style.RESET_ALL}"
            for key, value in config_dict.items()
        ]
    )
    print(log_str)


def log_rewards(r_batch: torch.tensor, run, step: int) -> None:
    r_logs = {f"agent_{agent_idx}": reward for agent_idx, reward in enumerate(r_batch)}

    run["charts/cumulative_rewards_per_update"].append(r_logs, step=step)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(6, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(
            nn.Linear(512, envs.unwrapped.single_action_space.n), std=0.01
        )
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        x = x.clone()
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        return self.critic(self.network(x.permute((0, 3, 1, 2))))

    def get_action_and_value(self, x, action=None):
        x = x.clone()
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        hidden = self.network(x.permute((0, 3, 1, 2)))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


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

    # env setup
    env = importlib.import_module(f"pettingzoo.atari.{args.env_id}").parallel_env()
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(
        env, args.num_workers // 2, num_cpus=0, base_class="gymnasium"
    )
    envs.single_observation_space = envs.observation_space
    envs.unwrapped.single_action_space = envs.action_space
    envs.is_vector_env = True
    # TODO: record statistics by hand
    # envs = gymnasium.wrappers.RecordEpisodeStatistics(envs)

    assert isinstance(
        envs.unwrapped.single_action_space, gymnasium.spaces.discrete.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.adam_lr, eps=args.adam_eps)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.rollout_length, args.num_workers) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.rollout_length, args.num_workers) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.rollout_length, args.num_workers)).to(device)
    rewards = torch.zeros((args.rollout_length, args.num_workers)).to(device)
    dones = torch.zeros((args.rollout_length, args.num_workers)).to(device)
    values = torch.zeros((args.rollout_length, args.num_workers)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_workers).to(device)
    num_updates = int(args.total_timesteps // args.batch_size)

    cumulative_rewards = np.zeros(args.num_workers)

    with tqdm(total=num_updates) as pbar:
        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.adam_lr
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.rollout_length):
                global_step += 1 * args.num_workers
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, trunc, info = envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                    done
                ).to(device)

                cumulative_rewards = cumulative_rewards + reward
                log_rewards(cumulative_rewards, run, global_step)

            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
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
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
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
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

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
    run.stop()
