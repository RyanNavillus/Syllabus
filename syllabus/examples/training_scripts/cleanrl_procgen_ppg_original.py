# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppg/#ppg_procgenpy
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from distutils.util import strtobool

import gym as openai_gym
import gymnasium as gym
import numpy as np
import procgen  # type: ignore # noqa: F401
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from procgen import ProcgenEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from torch import distributions as td
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from syllabus.core import GymnasiumSyncWrapper, make_multiprocessing_curriculum
from syllabus.core.evaluator import CleanRLEvaluator
from syllabus.curricula import (BatchedDomainRandomization,
                                CentralPrioritizedLevelReplay, Constant,
                                DirectPrioritizedLevelReplay,
                                DomainRandomization, LearningProgress,
                                PrioritizedLevelReplay, SequentialCurriculum)
from syllabus.examples.models import ProcgenAgent
from syllabus.examples.task_wrappers import ProcgenTaskWrapper
from syllabus.examples.utils.vecenv import (VecExtractDictObs, VecMonitor,
                                            VecNormalize)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "syllabus-testing"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    logging_dir: str = "."
    """the base directory for logging and wandb storage."""

    # Algorithm specific arguments
    env_id: str = "starpilot"
    """the id of the environment"""
    total_timesteps: int = int(25e6)
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    adv_norm_fullbatch: bool = True
    """Toggle full batch advantage normalization as used in PPG code"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # PPG specific arguments
    n_iteration: int = 32
    """N_pi: the number of policy update in the policy phase """
    e_policy: int = 1
    """E_pi: the number of policy update in the policy phase """
    v_value: int = 1
    """E_V: the number of policy update in the policy phase """
    e_auxiliary: int = 6
    """E_aux:the K epochs to update the policy"""
    beta_clone: float = 1.0
    """the behavior cloning coefficient"""
    num_aux_rollouts: int = 4
    """the number of mini batch in the auxiliary phase"""
    n_aux_grad_accum: int = 1
    """the number of gradient accumulation in mini batch"""

    # Syllabus specific arguments
    curriculum: bool = False
    """if toggled, this experiment will use curriculum learning"""
    curriculum_method: str = "plr"
    """curriculum method to use"""
    num_eval_episodes: int = 10
    """the number of episodes to evaluate the agent on after each policy update."""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    num_phases: int = 0
    """the number of phases (computed in runtime)"""
    aux_batch_rollouts: int = 0
    """the number of rollouts in the auxiliary phase (computed in runtime)"""


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


def make_env(env_id, seed, task_wrapper=False, curriculum_components=None, start_level=0, num_levels=1):
    def thunk():
        env = openai_gym.make(f"procgen-{env_id}-v0", distribution_mode="easy",
                              start_level=start_level, num_levels=num_levels)
        env = GymV21CompatibilityV0(env=env)

        if task_wrapper or curriculum_components is not None:
            env = ProcgenTaskWrapper(env, env_id, seed=seed)

        if curriculum_components is not None:
            env = GymnasiumSyncWrapper(
                env,
                env.task_space,
                curriculum_components,
                batch_size=256,
            )
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

    eval_envs = ProcgenEnv(
        num_envs=args.num_eval_episodes, env_name=env_name, num_levels=num_levels, start_level=0, distribution_mode="easy"
    )
    eval_envs = VecExtractDictObs(eval_envs, "rgb")
    eval_envs = wrap_vecenv(eval_envs)
    eval_obs, _ = eval_envs.reset()
    eval_episode_rewards = []

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            eval_action, _, _, _ = policy.get_action_and_value(torch.Tensor(eval_obs).to(device))

        eval_obs, _, truncs, terms, infos = eval_envs.step(eval_action.cpu().numpy())
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    mean_returns = np.mean(eval_episode_rewards)
    stddev_returns = np.std(eval_episode_rewards)
    env_min, env_max = PROCGEN_RETURN_BOUNDS[args.env_id]
    normalized_mean_returns = (mean_returns - env_min) / (env_max - env_min)
    policy.train()
    return mean_returns, stddev_returns, normalized_mean_returns


def layer_init_normed(layer, norm_dim, scale=1.0):
    with torch.no_grad():
        layer.weight.data *= scale / layer.weight.norm(dim=norm_dim, p=2, keepdim=True)
        layer.bias *= 0
    return layer


def flatten01(arr):
    return arr.reshape((-1, *arr.shape[2:]))


def unflatten01(arr, targetshape):
    return arr.reshape((*targetshape, *arr.shape[1:]))


def flatten_unflatten_test():
    a = torch.rand(400, 30, 100, 100, 5)
    b = flatten01(a)
    c = unflatten01(b, a.shape[:2])
    assert torch.equal(a, c)


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        # scale = (1/3**0.5 * 1/2**0.5)**0.5 # For default IMPALA CNN this is the final scale value in the PPG code
        scale = np.sqrt(scale)
        conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv0 = layer_init_normed(conv0, norm_dim=(1, 2, 3), scale=scale)
        conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = layer_init_normed(conv1, norm_dim=(1, 2, 3), scale=scale)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, scale):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.conv = layer_init_normed(conv, norm_dim=(1, 2, 3), scale=1.0)
        nblocks = 2  # Set to the number of residual blocks
        scale = scale / np.sqrt(nblocks)
        self.res_block0 = ResidualBlock(self._out_channels, scale=scale)
        self.res_block1 = ResidualBlock(self._out_channels, scale=scale)

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
        chans = [16, 32, 32]
        scale = 1 / np.sqrt(len(chans))  # Not fully sure about the logic behind this but its used in PPG code
        for out_channels in chans:
            conv_seq = ConvSequence(shape, out_channels, scale=scale)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)

        encodertop = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        encodertop = layer_init_normed(encodertop, norm_dim=1, scale=1.4)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            encodertop,
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init_normed(nn.Linear(256, envs.single_action_space.n), norm_dim=1, scale=0.1)
        self.critic = layer_init_normed(nn.Linear(256, 1), norm_dim=1, scale=0.1)
        self.aux_critic = layer_init_normed(nn.Linear(256, 1), norm_dim=1, scale=0.1)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden.detach())

    def get_value(self, x):
        return self.critic(self.network(x.permute((0, 3, 1, 2)) / 255.0))  # "bhwc" -> "bchw"

    # PPG logic:
    def get_pi_value_and_aux_value(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return Categorical(logits=self.actor(hidden)), self.critic(hidden.detach()), self.aux_critic(hidden)

    def get_pi(self, x):
        return Categorical(logits=self.actor(self.network(x.permute((0, 3, 1, 2)) / 255.0)))


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.num_phases = int(args.num_iterations // args.n_iteration)
    args.aux_batch_rollouts = int(args.num_envs * args.n_iteration)
    assert args.v_value == 1, "Multiple value epoch (v_value != 1) is not supported yet"
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
            dir=args.logging_dir,
        )

    writer = SummaryWriter(os.path.join(args.logging_dir, f"./runs/{run_name}"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    flatten_unflatten_test()  # Try not to mess with the flatten unflatten logic

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Device:", device)

    sample_envs = gym.vector.AsyncVectorEnv([make_env(args.env_id, args.seed + i) for i in range(args.num_envs)])
    sample_envs = wrap_vecenv(sample_envs)
    single_action_space = sample_envs.single_action_space
    single_observation_space = sample_envs.single_observation_space
    sample_envs.close()

    # Agent setup
    assert isinstance(single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    print("Creating agent")
    agent = ProcgenAgent(
        single_observation_space.shape,
        single_action_space.n,
        base_kwargs={'hidden_size': 256},
        detach_critic=True,
    ).to(device)
    policy_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if args.e_policy == args.e_value:
        value_optimizer = policy_optimizer
    else:
        value_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    aux_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Curriculum setup
    curriculum = None
    if args.curriculum:
        sample_env = openai_gym.make(f"procgen-{args.env_id}-v0")
        sample_env = GymV21CompatibilityV0(env=sample_env)
        sample_env = ProcgenTaskWrapper(sample_env, args.env_id, seed=args.seed)

        # Intialize Curriculum Method
        if args.curriculum_method == "plr":
            print("Using prioritized level replay.")
            evaluator = CleanRLEvaluator(agent, device="cuda", copy_agent=True)
            curriculum = PrioritizedLevelReplay(
                sample_env.task_space,
                sample_env.observation_space,
                num_steps=args.num_steps,
                num_processes=args.num_envs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                task_sampler_kwargs_dict={"strategy": "value_l1"},
                evaluator=evaluator,
                device="cuda",
            )
        elif args.curriculum_method == "simpleplr":
            print("Using simple prioritized level replay.")
            curriculum = DirectPrioritizedLevelReplay(
                sample_env.task_space,
                num_steps=args.num_steps,
                num_processes=args.num_envs,
                device="cpu",
                suppress_usage_warnings=False,
            )
        elif args.curriculum_method == "centralplr":
            print("Using centralized prioritized level replay.")
            curriculum = CentralPrioritizedLevelReplay(
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
        elif args.curriculum_method == "bdr":
            print("Using batched domain randomization.")
            curriculum = BatchedDomainRandomization(args.batch_size, sample_env.task_space)
        elif args.curriculum_method == "constant":
            print("Using constant curriculum.")
            curriculum = Constant(0, sample_env.task_space, require_step_updates=True)
        elif args.curriculum_method == "lp":
            print("Using learning progress.")
            eval_envs = gym.vector.AsyncVectorEnv(
                [make_env(args.env_id, 0, task_wrapper=True, num_levels=1) for _ in range(8)]
            )
            eval_envs = wrap_vecenv(eval_envs)
            evaluator = CleanRLEvaluator(agent, device="cuda", copy_agent=True)
            curriculum = LearningProgress(
                eval_envs, evaluator, sample_env.task_space, eval_interval_steps=25 * args.batch_size
            )
        elif args.curriculum_method == "sq":
            print("Using sequential curriculum.")
            curricula = []
            stopping = []
            for i in range(0, 199, 10):
                curricula.append(list(range(i, i + 10)))
                stopping.append("steps>=500000")
                curricula.append(list(range(i + 10)))
                stopping.append("steps>=500000")
            curriculum = SequentialCurriculum(curricula, stopping[:-1], sample_env.task_space)
        else:
            raise ValueError(f"Unknown curriculum method {args.curriculum_method}")
        curriculum = make_multiprocessing_curriculum(curriculum)

    # env setup
    print("Creating env")
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + i,
                curriculum_components=curriculum.components if args.curriculum else None,
                num_levels=1 if args.curriculum else 0,
            )
            for i in range(args.num_envs)
        ]
    )
    envs = wrap_vecenv(envs)

    # Wait to delete sample_env until after envs is created. For some reason procgen wants to rebuild for each env.
    if args.curriculum:
        del sample_env

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    aux_obs = torch.zeros(
        (args.num_steps, args.aux_batch_rollouts) + envs.single_observation_space.shape, dtype=torch.uint8
    )  # Saves lot system RAM
    aux_returns = torch.zeros((args.num_steps, args.aux_batch_rollouts))
    tasks = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    episode_rewards = deque(maxlen=10)
    completed_episodes = 0

    for phase in range(1, args.num_phases + 1):

        # POLICY PHASE
        for update in range(1, args.n_iteration + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                policy_optimizer.param_groups[0]["lr"] = lrnow
                value_optimizer.param_groups[0]["lr"] = lrnow
                aux_optimizer.param_groups[0]["lr"] = lrnow

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
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                if args.curriculum:
                    tasks[step] = torch.Tensor([i["task"] for i in info])
                done = np.logical_or(term, trunc)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
                completed_episodes += sum(done)

                for item in info:
                    if "episode" in item.keys():
                        episode_rewards.append(item['episode']['r'])
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        if curriculum is not None:
                            curriculum.log_metrics(writer, [], step=global_step, log_n_tasks=5)
                        break

                # Syllabus curriculum update
                if args.curriculum and args.curriculum_method == "centralplr":
                    with torch.no_grad():
                        next_value = agent.get_value(next_obs)
                    current_task = tasks[step]

                    plr_update = {
                        "value": value,
                        "next_value": next_value,
                        "rew": reward,
                        "dones": done,
                        "tasks": current_task,
                    }
                    curriculum.update(plr_update)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
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

            if args.curriculum and args.curriculum_method == "simpleplr":
                a, b = returns.shape
                new_returns = torch.zeros((a + 1, b))
                new_returns[:-1, :] = returns
                new_values = torch.zeros((a + 1, b))
                new_values[:-1, :] = values
                new_values[-1, :] = next_value
                scores = (new_returns - new_values).abs()
                curriculum.update(tasks, scores, dones)

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # PPG code does full batch advantage normalization
            if args.adv_norm_fullbatch:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.e_policy):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, _ = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss

                    policy_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    policy_optimizer.step()

            for epoch in range(args.e_value):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, _, _, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds])

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

                    loss = v_loss * args.vf_coef

                    value_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    value_optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # Evaluate agent
            mean_eval_returns, stddev_eval_returns, normalized_mean_eval_returns = level_replay_evaluate(
                args.env_id, agent, args.num_eval_episodes, device, num_levels=0
            )
            mean_train_returns, stddev_train_returns, normalized_mean_train_returns = level_replay_evaluate(
                args.env_id, agent, args.num_eval_episodes, device, num_levels=200
            )

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", policy_optimizer.param_groups[0]["lr"], global_step)
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

            writer.add_scalar("train_eval/mean_episode_return", mean_train_returns, global_step)
            writer.add_scalar("train_eval/normalized_mean_train_return", normalized_mean_train_returns, global_step)
            writer.add_scalar("train_eval/stddev_train_return", stddev_train_returns, global_step)

            writer.add_scalar("curriculum/completed_episodes", completed_episodes, step)

            # PPG Storage - Rollouts are saved without flattening for sampling full rollouts later:
            storage_slice = slice(args.num_envs * (update - 1), args.num_envs * update)
            aux_obs[:, storage_slice] = obs.cpu().clone().to(torch.uint8)
            aux_returns[:, storage_slice] = returns.cpu().clone()

        # AUXILIARY PHASE
        aux_inds = np.arange(args.aux_batch_rollouts)

        # Build the old policy on the aux buffer before distilling to the network
        aux_pi = torch.zeros((args.num_steps, args.aux_batch_rollouts, envs.single_action_space.n))
        for i, start in enumerate(range(0, args.aux_batch_rollouts, args.num_aux_rollouts)):
            end = start + args.num_aux_rollouts
            aux_minibatch_ind = aux_inds[start:end]
            m_aux_obs = aux_obs[:, aux_minibatch_ind].to(torch.float32).to(device)
            m_obs_shape = m_aux_obs.shape
            m_aux_obs = flatten01(m_aux_obs)
            with torch.no_grad():
                pi_logits = agent.get_pi(m_aux_obs).logits.cpu().clone()
            aux_pi[:, aux_minibatch_ind] = unflatten01(pi_logits, m_obs_shape[:2])
            del m_aux_obs

        for auxiliary_update in range(1, args.e_auxiliary + 1):
            print(f"aux epoch {auxiliary_update}")
            np.random.shuffle(aux_inds)
            for i, start in enumerate(range(0, args.aux_batch_rollouts, args.num_aux_rollouts)):
                end = start + args.num_aux_rollouts
                aux_minibatch_ind = aux_inds[start:end]
                try:
                    m_aux_obs = aux_obs[:, aux_minibatch_ind].to(device)
                    m_obs_shape = m_aux_obs.shape
                    m_aux_obs = flatten01(m_aux_obs)  # Sample full rollouts for PPG instead of random indexes
                    m_aux_returns = aux_returns[:, aux_minibatch_ind].to(torch.float32).to(device)
                    m_aux_returns = flatten01(m_aux_returns)

                    new_pi, new_values, new_aux_values = agent.get_pi_value_and_aux_value(m_aux_obs)

                    new_values = new_values.view(-1)
                    new_aux_values = new_aux_values.view(-1)
                    old_pi_logits = flatten01(aux_pi[:, aux_minibatch_ind]).to(device)
                    old_pi = Categorical(logits=old_pi_logits)
                    kl_loss = td.kl_divergence(old_pi, new_pi).mean()

                    real_value_loss = 0.5 * ((new_values - m_aux_returns) ** 2).mean()
                    aux_value_loss = 0.5 * ((new_aux_values - m_aux_returns) ** 2).mean()
                    joint_loss = aux_value_loss + args.beta_clone * kl_loss

                    loss = (joint_loss + real_value_loss) / args.n_aux_grad_accum
                    loss.backward()

                    if (i + 1) % args.n_aux_grad_accum == 0:
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        aux_optimizer.step()
                        aux_optimizer.zero_grad()  # This cannot be outside, else gradients won't accumulate

                except RuntimeError as e:
                    raise Exception(
                        "if running out of CUDA memory, try a higher --n-aux-grad-accum, which trades more time for less gpu memory"
                    ) from e

                del m_aux_obs, m_aux_returns
        writer.add_scalar("losses/aux/kl_loss", kl_loss.mean().item(), global_step)
        writer.add_scalar("losses/aux/aux_value_loss", aux_value_loss.item(), global_step)
        writer.add_scalar("losses/aux/real_value_loss", real_value_loss.item(), global_step)

    envs.close()
    writer.close()
