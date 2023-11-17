# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_procgenpy
import argparse
import os
import random
import time
from distutils.util import strtobool
from collections import deque

import gym
import numpy as np
import procgen  # noqa: F401
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from syllabus.core import (MultiProcessingSyncWrapper,
                           make_multiprocessing_curriculum)
from syllabus.curricula import PrioritizedLevelReplay, DomainRandomization, SequentialCurriculum
from syllabus.examples.task_wrappers import ProcgenTaskWrapper
from syllabus.examples.models import ProcgenAgent
from .vecenv import VecNormalize, VecMonitor, VecExtractDictObs
from procgen import ProcgenEnv


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="starpilot",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(25e6),
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=64,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.999,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=3,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    # Procgen arguments
    parser.add_argument("--full-dist", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Train on full distribution of levels.")

    # Curriculum arguments
    parser.add_argument("--curriculum", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will use curriculum learning")
    parser.add_argument("--curriculum-method", type=str, default="plr",
        help="curriculum method to use")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


PROCGEN_RETURN_BOUNDS = {
    "coinrun": (5, 10),
    "starpilot": (2.5, 64),
    "caveflyer": (3.5, 12),
    "dodgeball": (1.5, 19),
    "fruitbot": (-1.5, 32.4),
    "chaser": (0.5, 13),
    "miner": (1.5, 13),
    "jumper": (3, 10),
    "leaper": (3, 10),
    "maze": (5, 10),
    "bigfish": (1, 40),
    "heist": (3.5, 10),
    "climber": (2, 12.6),
    "plunder": (4.5, 30),
    "ninja": (3.5, 10),
    "bossfight": (0.5, 13),
}


def make_env(env_id, seed, task_queue, update_queue, start_level=0, num_levels=1):
    def thunk():
        env = gym.make(f"procgen-{env_id}-v0", distribution_mode="easy", start_level=start_level, num_levels=num_levels)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ProcgenTaskWrapper(env, env_id, seed)
        if args.curriculum:
            if task_queue is not None and update_queue is not None:
                env = MultiProcessingSyncWrapper(
                    env,
                    task_queue,
                    update_queue,
                    update_on_step=False,
                    default_task=start_level,
                    task_space=env.task_space,
                )
        # env.seed(seed)
        # gym.utils.seeding.np_random(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env
    return thunk


def wrap_vecenv(vecenv):
    vecenv.is_vector_env = True
    vecenv = VecMonitor(venv=vecenv, filename=None, keep_buf=100)
    vecenv = VecNormalize(venv=vecenv, ob=False, ret=True)
    return vecenv


def level_replay_evaluate(
    env_name,
    policy,
    num_episodes,
    device,
    num_levels=0
):
    policy.eval()
    eval_envs = ProcgenEnv(num_envs=1, env_name=env_name,
                           num_levels=num_levels, start_level=0,
                           distribution_mode="easy", paint_vel_info=False)
    eval_envs = VecExtractDictObs(eval_envs, "rgb")
    eval_envs = VecMonitor(venv=eval_envs, filename=None, keep_buf=100)
    eval_envs = VecNormalize(venv=eval_envs, ob=False, ret=True)
    # eval_envs = gym.wrappers.NormalizeReward(eval_envs, gamma=args.gamma)
    # eval_envs = gym.wrappers.TransformReward(eval_envs, lambda reward: np.clip(reward, -10, 10))

    eval_episode_rewards = []
    eval_obs = eval_envs.reset()

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            eval_action, _, _, _ = policy.get_action_and_value(torch.Tensor(eval_obs).to(device), deterministic=False)

        eval_obs, _, done, infos = eval_envs.step(eval_action.cpu().numpy())

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()
    mean_returns = np.mean(eval_episode_rewards)
    stddev_returns = np.std(eval_episode_rewards)
    env_min, env_max = PROCGEN_RETURN_BOUNDS[args.env_id]
    normalized_mean_returns = (mean_returns - env_min) / (env_max - env_min)
    policy.train()
    return mean_returns, stddev_returns, normalized_mean_returns


def evaluate(ev_envs, use_train_seeds=False):
    num_episodes = 10
    eval_returns = []
    eval_lengths = []
    if use_train_seeds:
        seeds = list(range(args.num_envs))
    else:
        seeds = [random.randint(0, 100000) for _ in range(args.num_envs)]
    ev_envs.seed(seeds)
    eval_obs = ev_envs.reset()
    while len(eval_returns) < num_episodes:
        with torch.no_grad():
            eval_action, _, _, _ = agent.get_action_and_value(torch.Tensor(eval_obs).to(device), deterministic=True)
        eval_obs, _, _, eval_info = ev_envs.step(eval_action.cpu().numpy())
        for item in eval_info:
            if "episode" in item.keys():
                eval_returns.append(item['episode']['r'])
                eval_lengths.append(item['episode']['l'])

    mean_returns = np.mean(eval_returns)
    stddev_returns = np.std(eval_returns)
    mean_lengths = np.mean(eval_lengths)
    env_min, env_max = PROCGEN_RETURN_BOUNDS[args.env_id]
    normalized_mean_returns = (mean_returns - env_min) / (env_max - env_min)
    return mean_returns, stddev_returns, mean_lengths, normalized_mean_returns


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        h, w, c = envs.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x.permute((0, 3, 1, 2)) / 255.0))  # "bhwc" -> "bchw"

    def get_action_and_value(self, x, action=None, full_log_probs=False):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        value = self.critic(hidden)
        logits = self.actor(hidden)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        action_log_probs = torch.squeeze(dist.log_prob(action))
        dist_entropy = dist.entropy()

        if full_log_probs:
            log_probs = torch.log(dist.probs)
            return action, action_log_probs, dist_entropy, value, log_probs

        return action, action_log_probs, dist_entropy, value


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.run.log_code("./syllabus/examples")
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Curriculum setup
    task_queue = update_queue = None
    if args.curriculum:
        sample_env = gym.make(f"procgen-{args.env_id}-v0")
        sample_env = ProcgenTaskWrapper(sample_env, args.env_id, args.seed)
        if args.curriculum_method == "plr":
            print("Using prioritized level replay.")
            curriculum = PrioritizedLevelReplay(
                sample_env.task_space,
                num_steps=args.num_steps,
                num_processes=args.num_envs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                task_sampler_kwargs_dict={"strategy": "value_l1"}
            )
        elif args.curriculum_method == "dr":
            print("Using domain randomization.")
            curriculum = DomainRandomization(sample_env.task_space)
        else:
            raise ValueError(f"Unknown curriculum method {args.curriculum_method}")
        curriculum, task_queue, update_queue = make_multiprocessing_curriculum(curriculum)

        del sample_env

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, task_queue, update_queue, num_levels=1 if args.curriculum else 0)
            for i in range(args.num_envs)
        ]
    )
    envs = wrap_vecenv(envs)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Full distribution eval environment
    # eval_envs = gym.vector.AsyncVectorEnv(
    #     [
    #         make_env(args.env_id, args.seed + i, None, None, num_levels=0)
    #         for i in range(args.num_envs)
    #     ]
    # )
    # eval_envs = wrap_vecenv(eval_envs)
    # eval_obs = eval_envs.reset()

    # print(envs.single_observation_space, envs.single_action_space)
    agent = ProcgenAgent(envs.single_observation_space.shape, envs.single_action_space.n, arch="large", base_kwargs={'recurrent': False, 'hidden_size': 256}).to(device)
    # agent = Agent(envs).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # optimizer = optim.RMSprop(agent.parameters(), lr=args.learning_rate, eps=1e-5, alpha=0.99)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    episode_rewards = deque(maxlen=10)
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    episode_rewards.append(info['episode']['r'])
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

            # Syllabus curriculum update
            if args.curriculum and args.curriculum_method == "plr":
                with torch.no_grad():
                    next_value = agent.get_value(next_obs)
                tasks = envs.get_attr("task")
                update = {
                    "update_type": "on_demand",
                    "metrics": {
                        "value": value,
                        "next_value": next_value,
                        "rew": reward,
                        "masks": torch.Tensor(1 - done),
                        "tasks": tasks,
                    },
                }
                curriculum.update_curriculum(update)
            if args.curriculum:
                curriculum.log_metrics(writer, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Evaluate agent
        mean_train_eval_returns, stddev_train_eval_returns, mean_train_eval_lengths, normalized_mean_train_eval_returns = evaluate(eval_envs, use_train_seeds=True)
        mean_eval_returns, stddev_eval_returns, mean_eval_lengths, normalized_mean_eval_returns = evaluate(eval_envs)
        mean_level_replay_eval_returns, stddev_level_replay_eval_returns, normalized_mean_level_replay_eval_returns = level_replay_evaluate(args.env_id, agent, 10, device)
        mean_level_replay_train_returns, stddev_level_replay_train_returns, normalized_mean_level_replay_train_returns = level_replay_evaluate(args.env_id, agent, 10, device, num_levels=200)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/episode_returns", np.mean(episode_rewards), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # writer.add_scalar("test_eval/mean_eval_return", mean_eval_returns, global_step)
        # writer.add_scalar("test_eval/stddev_eval_return", stddev_eval_returns, global_step)
        # writer.add_scalar("test_eval/mean_eval_length", mean_eval_lengths, global_step)
        # writer.add_scalar("test_eval/normalized_mean_eval_return", normalized_mean_eval_returns, global_step)
        writer.add_scalar("test_eval/mean_episode_return", mean_level_replay_eval_returns, global_step)
        writer.add_scalar("test_eval/normalized_mean_eval_return", normalized_mean_level_replay_eval_returns, global_step)
        writer.add_scalar("test_eval/stddev_eval_return", mean_level_replay_eval_returns, global_step)
        writer.add_scalar("train_eval/mean_episode_return", mean_level_replay_train_returns, global_step)
        writer.add_scalar("train_eval/normalized_mean_train_return", normalized_mean_level_replay_train_returns, global_step)
        writer.add_scalar("train_eval/stddev_train_return", mean_level_replay_train_returns, global_step)
        # writer.add_scalar("train_eval/mean_eval_return", mean_train_eval_returns, global_step)
        # writer.add_scalar("train_eval/stddev_eval_return", stddev_train_eval_returns, global_step)
        # writer.add_scalar("train_eval/mean_eval_length", mean_train_eval_lengths, global_step)
        # writer.add_scalar("train_eval/normalized_mean_eval_return", normalized_mean_train_eval_returns, global_step)

    # eval_envs.close()
    envs.close()
    writer.close()
