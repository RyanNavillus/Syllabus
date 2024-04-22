""" An example applying Syllabus Prioritized Level Replay to Procgen. This code is based on https://github.com/facebookresearch/level-replay/blob/main/train.py

NOTE: In order to efficiently change the seed of a procgen environment directly without reinitializing it,
we rely on Minqi Jiang's custom branch of procgen found here: https://github.com/minqi/procgen
"""
import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool

import gym as openai_gym
import gymnasium as gym
import numpy as np
import procgen  # noqa: F401
from procgen import ProcgenEnv
import torch
import torch.nn as nn
import torch.optim as optim
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from torch.utils.tensorboard import SummaryWriter

from syllabus.core import MultiProcessingSyncWrapper, make_multiprocessing_curriculum
from syllabus.curricula import PrioritizedLevelReplay, DomainRandomization, LearningProgressCurriculum, SequentialCurriculum
from syllabus.examples.models import ProcgenAgent
from syllabus.examples.task_wrappers import ProcgenTaskWrapper
from syllabus.examples.utils.vecenv import VecMonitor, VecNormalize, VecExtractDictObs


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
    parser.add_argument("--wandb-project-name", type=str, default="syllabus",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--logging-dir", type=str, default=".",
                        help="the base directory for logging and wandb storage.")

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
    parser.add_argument("--num-eval-episodes", type=int, default=10,
                        help="the number of episodes to evaluate the agent on after each policy update.")

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


def make_env(env_id, seed, curriculum=None, start_level=0, num_levels=1):
    def thunk():
        env = openai_gym.make(f"procgen-{env_id}-v0", distribution_mode="easy", start_level=start_level, num_levels=num_levels)
        env = GymV21CompatibilityV0(env=env)
        if curriculum is not None:
            env = ProcgenTaskWrapper(env, env_id, seed=seed)
            env = MultiProcessingSyncWrapper(
                env,
                curriculum.get_components(),
                update_on_step=True,
                task_space=env.task_space,
            )
        return env
    return thunk


def wrap_vecenv(vecenv):
    vecenv.is_vector_env = True
    vecenv = VecMonitor(venv=vecenv, filename=None, keep_buf=100)
    vecenv = VecNormalize(venv=vecenv, ob=False, ret=True)
    return vecenv


def full_level_replay_evaluate(
    env_name,
    policy,
    num_episodes,
    device,
    num_levels=1    # Not used
):
    policy.eval()

    eval_envs = ProcgenEnv(
        num_envs=args.num_eval_episodes, env_name=env_name, num_levels=1, start_level=0, distribution_mode="easy", paint_vel_info=False
    )
    eval_envs = VecExtractDictObs(eval_envs, "rgb")
    eval_envs = wrap_vecenv(eval_envs)

    # Seed environments
    seeds = [int.from_bytes(os.urandom(3), byteorder="little") for _ in range(num_episodes)]
    for i, seed in enumerate(seeds):
        eval_envs.seed(seed, i)

    eval_obs, _ = eval_envs.reset()
    eval_episode_rewards = [-1] * num_episodes

    while -1 in eval_episode_rewards:
        with torch.no_grad():
            eval_action, _, _, _ = policy.get_action_and_value(torch.Tensor(eval_obs).to(device), deterministic=False)

        eval_obs, _, truncs, terms, infos = eval_envs.step(eval_action.cpu().numpy())
        for i, info in enumerate(infos):
            if 'episode' in info.keys() and eval_episode_rewards[i] == -1:
                eval_episode_rewards[i] = info['episode']['r']

    mean_returns = np.mean(eval_episode_rewards)
    stddev_returns = np.std(eval_episode_rewards)
    env_min, env_max = PROCGEN_RETURN_BOUNDS[args.env_id]
    normalized_mean_returns = (mean_returns - env_min) / (env_max - env_min)
    policy.train()
    return mean_returns, stddev_returns, normalized_mean_returns


def level_replay_evaluate(
    env_name,
    policy,
    num_episodes,
    device,
    num_levels=0
):
    policy.eval()

    eval_envs = ProcgenEnv(
        num_envs=args.num_eval_episodes, env_name=env_name, num_levels=num_levels, start_level=0, distribution_mode="easy", paint_vel_info=False
    )
    eval_envs = VecExtractDictObs(eval_envs, "rgb")
    eval_envs = wrap_vecenv(eval_envs)
    eval_obs, _ = eval_envs.reset()
    eval_episode_rewards = [-1] * num_episodes

    while -1 in eval_episode_rewards:
        with torch.no_grad():
            eval_action, _, _, _ = policy.get_action_and_value(torch.Tensor(eval_obs).to(device), deterministic=False)

        eval_obs, _, truncs, terms, infos = eval_envs.step(eval_action.cpu().numpy())
        for i, info in enumerate(infos):
            if 'episode' in info.keys() and eval_episode_rewards[i] == -1:
                eval_episode_rewards[i] = info['episode']['r']

    # print(eval_episode_rewards)
    mean_returns = np.mean(eval_episode_rewards)
    stddev_returns = np.std(eval_episode_rewards)
    env_min, env_max = PROCGEN_RETURN_BOUNDS[args.env_id]
    normalized_mean_returns = (mean_returns - env_min) / (env_max - env_min)
    policy.train()
    return mean_returns, stddev_returns, normalized_mean_returns


def make_value_fn():
    def get_value(obs):
        obs = np.array(obs)
        with torch.no_grad():
            return agent.get_value(torch.Tensor(obs).to(device))
    return get_value


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
            dir=args.logging_dir
        )
        # wandb.run.log_code("./syllabus/examples")

    writer = SummaryWriter(os.path.join(args.logging_dir, "./runs/{run_name}"))
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
    print("Device:", device)

    # Curriculum setup
    curriculum = None
    if args.curriculum:
        sample_env = openai_gym.make(f"procgen-{args.env_id}-v0")
        sample_env = GymV21CompatibilityV0(env=sample_env)
        sample_env = ProcgenTaskWrapper(sample_env, args.env_id, seed=args.seed)

        # Intialize Curriculum Method
        if args.curriculum_method == "plr":
            print("Using prioritized level replay.")
            curriculum = PrioritizedLevelReplay(
                sample_env.task_space,
                sample_env.observation_space,
                num_steps=args.num_steps,
                num_processes=args.num_envs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                task_sampler_kwargs_dict={"strategy": "value_l1"},
                get_value=make_value_fn(),
            )
        elif args.curriculum_method == "dr":
            print("Using domain randomization.")
            curriculum = DomainRandomization(sample_env.task_space)
        elif args.curriculum_method == "lp":
            print("Using learning progress.")
            curriculum = LearningProgressCurriculum(sample_env.task_space)
        elif args.curriculum_method == "sq":
            print("Using sequential curriculum.")
            curricula = []
            stopping = []
            for i in range(199):
                curricula.append(i + 1)
                stopping.append("steps>=50000")
                curricula.append(list(range(i + 1)))
                stopping.append("steps>=50000")
            curriculum = SequentialCurriculum(curricula, stopping[:-1], sample_env.task_space)
        else:
            raise ValueError(f"Unknown curriculum method {args.curriculum_method}")
        curriculum = make_multiprocessing_curriculum(curriculum)
        del sample_env

    # env setup
    print("Creating env")
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + i,
                curriculum=curriculum if args.curriculum else None,
                num_levels=1 if args.curriculum else 0
            )
            for i in range(args.num_envs)
        ]
    )
    envs = wrap_vecenv(envs)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    print("Creating agent")
    agent = ProcgenAgent(
        envs.single_observation_space.shape,
        envs.single_action_space.n,
        arch="large",
        base_kwargs={'recurrent': False, 'hidden_size': 256}
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

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
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    episode_rewards = deque(maxlen=10)
    completed_episodes = 0

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
            next_obs, reward, term, trunc, info = envs.step(action.cpu().numpy())
            done = np.logical_or(term, trunc)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            completed_episodes += sum(done)

            for item in info:
                if "episode" in item.keys():
                    episode_rewards.append(item['episode']['r'])
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    if curriculum is not None:
                        curriculum.log_metrics(writer, global_step)
                    break

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
        mean_eval_returns, stddev_eval_returns, normalized_mean_eval_returns = level_replay_evaluate(
            args.env_id, agent, args.num_eval_episodes, device, num_levels=0
        )
        full_mean_eval_returns, full_stddev_eval_returns, full_normalized_mean_eval_returns = full_level_replay_evaluate(
            args.env_id, agent, args.num_eval_episodes, device, num_levels=0
        )
        mean_train_returns, stddev_train_returns, normalized_mean_train_returns = level_replay_evaluate(
            args.env_id, agent, args.num_eval_episodes, device, num_levels=200
        )
        full_mean_train_returns, full_stddev_train_returns, full_normalized_mean_train_returns = full_level_replay_evaluate(
            args.env_id, agent, args.num_eval_episodes, device, num_levels=200
        )

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

        writer.add_scalar("test_eval/mean_episode_return", mean_eval_returns, global_step)
        writer.add_scalar("test_eval/normalized_mean_eval_return", normalized_mean_eval_returns, global_step)
        writer.add_scalar("test_eval/stddev_eval_return", stddev_eval_returns, global_step)
        writer.add_scalar("test_eval/full_mean_episode_return", full_mean_eval_returns, global_step)
        writer.add_scalar("test_eval/full_normalized_mean_eval_return", full_normalized_mean_eval_returns, global_step)
        writer.add_scalar("test_eval/full_stddev_eval_return", full_stddev_eval_returns, global_step)

        writer.add_scalar("train_eval/mean_episode_return", mean_train_returns, global_step)
        writer.add_scalar("train_eval/normalized_mean_train_return", normalized_mean_train_returns, global_step)
        writer.add_scalar("train_eval/stddev_train_return", stddev_train_returns, global_step)
        writer.add_scalar("train_eval/full_mean_episode_return", full_mean_train_returns, global_step)
        writer.add_scalar("train_eval/full_normalized_mean_train_return", full_normalized_mean_train_returns, global_step)
        writer.add_scalar("train_eval/full_stddev_train_return", full_stddev_train_returns, global_step)

        writer.add_scalar("curriculum/completed_episodes", completed_episodes, step)

    envs.close()
    writer.close()
