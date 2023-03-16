# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is an example self-contained agent running NLE based on MonoBeast.

import argparse
import logging
import math
import os
import pprint
import threading
import time
import timeit
import traceback
from argparse import Namespace
from typing import Callable

import numpy as np

import wandb
from syllabus.core import (MultiProcessingSyncWrapper,
                           make_multiprocessing_curriculum)
from syllabus.curricula import LearningProgressCurriculum
from syllabus.examples import NethackTaskWrapper

# Necessary for multithreading.
os.environ["OMP_NUM_THREADS"] = "1"

try:
    import torch
    from torch import multiprocessing as mp
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    logging.exception(
        "PyTorch not found. Please install the agent dependencies with "
        '`pip install "nle[agent]"`'
    )

import gym  # noqa: E402
import nle  # noqa: F401, E402
from nle import nethack  # noqa: E402
from nle.agent import vtrace  # noqa: E402


def parse_args():
    # yapf: disable
    parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

    parser.add_argument("--env", type=str, default="NetHackScore-v0",
                        help="Gym environment.")
    parser.add_argument("--mode", default="train",
                        choices=["train", "test", "test_render"],
                        help="Training or test mode.")
    parser.add_argument("--profile", action="store_true",
                        help="Profile main process.")
    parser.add_argument("--profile_worker", action="store_true",
                        help="Profile worker process.")

    # Training settings.
    parser.add_argument("--disable_checkpoint", action="store_true",
                        help="Disable saving checkpoint.")
    parser.add_argument("--savedir", default="~/torchbeast/",
                        help="Root dir where experiment data will be saved.")
    parser.add_argument("--num_actors", default=8, type=int, metavar="N",
                        help="Number of actors (default: 4).")
    parser.add_argument("--total_steps", default=100000, type=int, metavar="T",
                        help="Total environment steps to train for.")
    parser.add_argument("--batch_size", default=16, type=int, metavar="B",
                        help="Learner batch size.")
    parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                        help="The unroll length (time dimension).")
    parser.add_argument("--num_buffers", default=None, type=int,
                        metavar="N", help="Number of shared-memory buffers.")
    parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                        metavar="N", help="Number learner threads.")
    parser.add_argument("--disable_cuda", action="store_true",
                        help="Disable CUDA.")
    parser.add_argument("--use_lstm", action="store_true",
                        help="Use LSTM in agent model.")
    parser.add_argument("--save_ttyrec_every", default=1000, type=int,
                        metavar="N", help="Save ttyrec every N episodes.")
    parser.add_argument("--save_video", action="store_true",
                        help="Save and log video during training.")

    # Curriculum Settings
    parser.add_argument("--curriculum", action="store_true",
                        help="Use Syllabus curricula.")

    # Testing settings
    parser.add_argument("--reward_frames", action="store_true",
                        help="Only print reward frames and show inventory.")
    parser.add_argument("--item_frames", action="store_true",
                        help="Only print frames where the agent picks up items and show inventory.")
    parser.add_argument("--message", action="store_true",
                        help="Set to true without the above options to display only the messages.")
    parser.add_argument("--custompath", type=str,
                        help="Set a custom path to draw a tar test file from.")

    # Weights and Biases settings
    parser.add_argument("--exp_name", type=str, default="nle_baseline",
                        help="Set name for wandb experiment.")
    parser.add_argument("--wandb_id", default=1, type=int,
                        help="Set id for wandb experiment.")

    # Loss settings.
    parser.add_argument("--entropy_cost", default=0.0006,
                        type=float, help="Entropy cost/multiplier.")
    parser.add_argument("--baseline_cost", default=0.5,
                        type=float, help="Baseline cost/multiplier.")
    parser.add_argument("--discounting", default=0.99,
                        type=float, help="Discounting factor.")
    parser.add_argument("--reward_clipping", default="abs_one",
                        choices=["abs_one", "none"],
                        help="Reward clipping.")

    # Optimizer settings.
    parser.add_argument("--learning_rate", default=0.00048,
                        type=float, metavar="LR", help="Learning rate.")
    parser.add_argument("--alpha", default=0.99, type=float,
                        help="RMSProp smoothing constant.")
    parser.add_argument("--momentum", default=0, type=float,
                        help="RMSProp momentum.")
    parser.add_argument("--epsilon", default=0.01, type=float,
                        help="RMSProp epsilon.")
    parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                        help="Global gradient norm clip.")
    # yapf: enable
    args = parser.parse_args()
    args.exp_name = f"nethack__{args.exp_name}__{int(time.time())}"
    return args


logging.basicConfig(
    format=("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"),
    level=logging.INFO,
    #filename="monobeast.log"
)


def nested_map(f, n):
    if isinstance(n, tuple) or isinstance(n, list):
        return n.__class__(nested_map(f, sn) for sn in n)
    elif isinstance(n, dict):
        return {k: nested_map(f, v) for k, v in n.items()}
    else:
        return f(n)


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages**2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def create_env(
    name,
    *args,
    chars=False,
    task_queue: mp.SimpleQueue = None,
    complete_queue: mp.SimpleQueue = None,
    step_queue: mp.SimpleQueue = None,
    observation_keys=None,
    **kwargs
):
    if flags.curriculum:
        observation_keys = ("glyphs", "blstats", "message")
        #observation_keys = ("glyphs", "blstats", "inv_glyphs", "inv_strs", "inv_letters", "inv_oclasses",
        #                    "message", "tty_chars", "tty_colors", "tty_cursor", "internal")
        if chars:
            observation_keys += ("chars",)

        env = gym.make(name, *args, observation_keys=observation_keys, penalty_step=0.0, **kwargs)
        env = NethackTaskWrapper(env)
        env = MultiProcessingSyncWrapper(env,
                                         task_queue,
                                         complete_queue,
                                         step_queue=step_queue,
                                         update_on_step=True,
                                         default_task=0,
                                         task_space=env.task_space)
    else:
        env = gym.make(name, *args, observation_keys=observation_keys, **kwargs)
    env = ResettingEnvironment(env)
    return env


def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers,
    initial_agent_state_buffers,
    task_queue: mp.SimpleQueue = None,
    complete_queue: mp.SimpleQueue = None,
    step_queue: mp.SimpleQueue = None,

    video_dir=None
):
    try:
        if flags.profile_worker and actor_index == 0:
            import cProfile
            from pstats import Stats
            pr = cProfile.Profile()
            pr.enable()

        logging.info("Actor %i started.", actor_index)

        gym_env = create_env(
            flags.env,
            savedir=flags.rundir,
            save_ttyrec_every=flags.save_ttyrec_every,
            observation_keys=("glyphs", "blstats"),
            task_queue=task_queue,
            complete_queue=complete_queue,
            step_queue=step_queue,
        )

        env = gym_env
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        agent_output, _ = model(env_output, agent_state)
        env_output = env.step(agent_output["action"])

        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.
            for t in range(flags.unroll_length):
                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                env_output = env.step(agent_output["action"])
                if flags.curriculum and "info" in env_output:
                    info = env_output["info"]

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

            full_queue.put(index)

        if flags.profile_worker and actor_index == 0:
            pr.disable()
            stats = Stats(pr)
            stats.sort_stats('cumtime').print_stats(200)

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers,
    initial_agent_state_buffers,
    lock=threading.Lock(),
):
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }

    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    for m in indices:
        free_queue.put(m)
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    return batch, initial_agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        episode_steps = batch["episode_step"][batch["done"]]
        if flags.curriculum:
            score_returns = batch["score_return"][batch["done"]]
            goal_returns = batch["goal_return"][batch["done"]]
            exp_bonuses = batch["exp_bonus"][batch["done"]]

        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "episode_lengths": tuple(episode_steps.cpu().numpy()),
            "mean_episode_length": torch.mean(episode_steps).item(),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }
        if flags.curriculum:
            stats["mean_score_return"] = torch.mean(score_returns).item()
            stats["mean_goal_return"] = torch.mean(goal_returns).item()
            stats["mean_exp_bonus"] = torch.mean(exp_bonuses).item()

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(flags, observation_space, num_actions, num_overlapping_steps=1):
    size = (flags.unroll_length + num_overlapping_steps,)

    # Get specimens to infer shapes and dtypes.
    samples = {k: torch.from_numpy(v) for k, v in observation_space.sample().items()}

    specs = {
        key: dict(size=size + sample.shape, dtype=sample.dtype)
        for key, sample in samples.items()
    }
    specs.update(
        reward=dict(size=size, dtype=torch.float32),
        done=dict(size=size, dtype=torch.bool),
        episode_return=dict(size=size, dtype=torch.float32),
        episode_step=dict(size=size, dtype=torch.float32),
        policy_logits=dict(size=size + (num_actions,), dtype=torch.float32),
        baseline=dict(size=size, dtype=torch.float32),
        last_action=dict(size=size, dtype=torch.int64),
        action=dict(size=size, dtype=torch.int64),
        score_reward=dict(size=size, dtype=torch.float32),
        score_return=dict(size=size, dtype=torch.float32),
        goal_return=dict(size=size, dtype=torch.float32),
        exp_bonus=dict(size=size, dtype=torch.float32),
        task_complete=dict(size=size, dtype=torch.bool),
    )
    buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def _format_observations(observation):
    observations = {}
    for key in list(observation.keys()):
        entry = observation[key]
        if isinstance(entry, np.ndarray):
            entry = torch.from_numpy(entry)
            entry = entry.view((1, 1) + entry.shape)  # (...) -> (T,B,...).
        observations[key] = entry
    return observations


class ResettingEnvironment:
    """Turns a Gym environment into something that can be step()ed indefinitely."""

    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        #self.score_return = None
        self._copy_gym_properties()
    
    def _copy_gym_properties(self):
        self.action_space = self.gym_env.action_space
        self.observation_space = self.gym_env.observation_space
        self.reward_range = self.gym_env.reward_range
        self.metadata = self.gym_env.metadata
        if flags.curriculum:
            self.task_space = self.gym_env.task_space


    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.float32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        if flags.curriculum:
            pass
            #self.score_return = torch.zeros(1, 1, dtype=torch.float32)

        result = _format_observations(self.gym_env.reset())
        result.update(
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )
        if flags.curriculum:
            pass
            #result.update(
            #    score_return=self.score_return,
            #)

        return result

    def step(self, action):
        observation, reward, done, info = self.gym_env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        if flags.curriculum:
            pass
            #reward_info = info["rewards"]
            #self.score_return += reward_info["score_reward"]
            #score_return = self.score_return

        episode_step = self.episode_step
        episode_return = self.episode_return

        if done:
            observation = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.float32)
            if flags.curriculum:
                pass
                #self.score_return = torch.zeros(1, 1)

        result = _format_observations(observation)

        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        if flags.curriculum:
            result.update(
                reward=reward,
                done=done,
                episode_return=episode_return,
                #score_return=score_return,
                episode_step=episode_step,
                last_action=action,
            )
        else:
            result.update(
                reward=reward,
                done=done,
                episode_return=episode_return,
                episode_step=episode_step,
                last_action=action,
            )

        return result

    def close(self):
        self.gym_env.close()


def parse_logpaths(flags):
    flags.savedir = os.path.expandvars(os.path.expanduser(flags.savedir))

    if flags.exp_name:
        rundir = os.path.join(flags.savedir, f"torchbeast-{flags.exp_name}-{flags.wandb_id}")
    else:
        rundir = os.path.join(flags.savedir, "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S"))

    checkpointpath = os.path.join(rundir, "model.tar")

    # TODO: Check if run name + id exists, and resume
    resume_checkpoint = None
    if os.path.exists(rundir):
        resume_checkpoint = torch.load(checkpointpath)
        # TODO: Make sure this doesn't overwrite anything important,
        # or that we want to be able to change
        # flags = Namespace(**resume_checkpoint["flags"])
    else:
        os.makedirs(rundir)

    logfile = open(os.path.join(rundir, "logs.tsv"), "a", buffering=1)
    logging.info("Logging results to %s", rundir)

    symlink = os.path.join(flags.savedir, "latest")
    try:
        if os.path.islink(symlink):
            os.remove(symlink)
        if not os.path.exists(symlink):
            os.symlink(rundir, symlink)
        logging.info("Symlinked log directory: %s", symlink)
    except OSError:
        raise

    flags.rundir = rundir
    return flags, rundir, checkpointpath, resume_checkpoint, logfile


def train(flags, wandb_run=None):  # pylint: disable=too-many-branches, too-many-statements
    # Set all filepaths using provided arguments
    flags, rundir, checkpointpath, resume_checkpoint, logfile = parse_logpaths(flags)

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    sample_env = create_env(flags.env, observation_keys=("glyphs", "blstats"))
    observation_space = sample_env.observation_space
    action_space = sample_env.action_space
    if flags.curriculum:
        task_space = sample_env.task_space

    model = Net(observation_space, action_space.n, flags.use_lstm, goal=flags.curriculum)
    buffers = create_buffers(flags, observation_space, model.num_actions)

    model.share_memory()

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    # Resume training
    # if resume_checkpoint:
    #     # TODO: Fix resume code
    #     #curriculum = LearningProgressCurriculum(tasks=resume_checkpoint["curriculum"])
    #     if flags.curriculum:
    #         curriculum = LearningProgressCurriculum(task_space=task_space)
    #     model.load_state_dict(resume_checkpoint["model"])
    #     learner_model = Net(observation_space, action_space.n, flags.use_lstm, goal=flags.curriculum).to(
    #         device=flags.device
    #     )
    #     learner_model.load_state_dict(model.state_dict())
    #     optimizer = torch.optim.RMSprop(
    #         learner_model.parameters(),
    #         lr=flags.learning_rate,
    #         momentum=flags.momentum,
    #         eps=flags.epsilon,
    #         alpha=flags.alpha,
    #     )
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    #     scheduler.load_state_dict(resume_checkpoint["scheduler"])
    #     optimizer.load_state_dict(resume_checkpoint['optimizer'])
    #     step = resume_checkpoint["step"]
    #
    # else:
    task_queue, complete_queue, step_queue = None, None, None
    if flags.curriculum:
        curriculum, task_queue, complete_queue, step_queue = make_multiprocessing_curriculum(LearningProgressCurriculum,
                                                                                             task_space,
                                                                                             random_start_tasks=0)

    learner_model = Net(observation_space, action_space.n, flags.use_lstm, goal=flags.curriculum).to(device=flags.device)
    learner_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    step = 0

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    del sample_env  # End this before forking.

    actor_processes = []
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags, i,
                free_queue, full_queue,
                model, buffers,
                initial_agent_state_buffers,
                task_queue, complete_queue, step_queue,
                rundir,
            ),
            name="Actor-%i" % i,
        )
        actor.start()
        actor_processes.append(actor)

    if flags.exp_name:
        wandb.config = {
            "learning_rate": flags.learning_rate,
            "epsilon": flags.epsilon,
            "alpha": flags.alpha,
            "momentum": flags.momentum,
        }

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    stat_keys = ["total_loss", "mean_episode_return", "pg_loss", "baseline_loss", "entropy_loss"]
    logfile.write("# Step\t%s\n" % "\t".join(stat_keys))

    all_stats = []
    stats = {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats, all_stats
        while step < flags.total_steps:
            print(step)
            batch, agent_state = get_batch(flags, free_queue, full_queue, buffers, initial_agent_state_buffers)
            stats = learn(flags, model, learner_model, batch, agent_state, optimizer, scheduler)

            all_stats.append(stats)
            with lock:
                logfile.write("%i\t" % step)
                logfile.write("\t".join(str(stats[k]) for k in stat_keys))
                logfile.write("\n")
                step += T * B

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn,
            name="batch-and-learn-%d" % i,
            args=(i,),
            daemon=True,  # To support KeyboardInterrupt below.
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "flags": vars(flags),
                "step": step
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    recent_sps = [1.0] * 5
    try:
        if flags.curriculum:
            curriculum.log_metrics(step)
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(10)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()
                #wandb.gym.monitor()

            # # Combine stats
            # all_stats_dict = {}
            # if len(all_stats) > 0:
            #     # Iterate through keys and combine based on data type
            #     for key, value in all_stats[0].items():
            #         all_stats_dict[key] = 0
            #         stat_list = []
            #         if isinstance(value, (int, float, np.uint8, np.float32)):
            #             # Average values
            #             for stat_dict in all_stats:
            #                 stat_value = stat_dict.get(key)
            #                 if stat_value is not None and not np.isnan(stat_value) and not math.isnan(stat_value):
            #                     stat_list.append(stat_value)
            #             all_stats_dict[key] = (sum(stat_list) / len(stat_list)) if len(stat_list) > 0 else 0
            #         elif isinstance(value, (list, tuple)):
            #             # Combine lists
            #             for stat_dict in all_stats:
            #                 stat_value = stat_dict.get(key)
            #                 if stat_value is not None and stat_value != () and stat_value != []:
            #                     stat_list += stat_value
            #             all_stats_dict[key] = stat_list
            # all_stats = []

            # # Remove clutter
            # if "episode_returns" in all_stats_dict:
            #     del all_stats_dict["episode_returns"]
            # if "episode_lengths" in all_stats_dict:
            #     del all_stats_dict["episode_lengths"]

            # # Log run data to weights and biases
            # if flags.exp_name:
            #     wandb_stats = all_stats_dict
            #     for key, value in wandb_stats.items():
            #         if isinstance(value, (int, float, np.uint8, np.float32)):
            #             wandb_stats[key] = 0.0 if np.isnan(value) else value
            #     if not flags.curriculum:
            #         wandb_stats["mean_score_return"] = wandb_stats["mean_episode_return"]
            #     wandb_stats["learning_rate"] = scheduler.get_last_lr()[0]
            #     # task_table.add_data(step, str(curriculum.export_task_names()))
            #     wandb.log(wandb_stats, step=step)

            sps = (step - start_step) / (timer() - start_time)

            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            log_str = "Steps %i @ %.1f SPS. Loss %f. %s"        # Stats:\n%s"
            log_args = [step, sps, total_loss, mean_return]     # , pprint.pformat(all_stats_dict)]
            # log_str = "Steps %i @ %.1f SPS. Loss %f. %s"
            # log_args = [step, sps, total_loss, mean_return]
            # if flags.curriculum:
            #     # TODO: Fix this
            #     task_names = curriculum.export_task_names()[:]
            #     log_str += "\nTasks: %s\nL_prog: %s\nP_fast: %s"
            #     log_args.append(task_names)
            #     # Get learning progress metric
            #     lps = curriculum.metric_for_tasks(task_names, metric="lp")
            #     lps = [f"{lp:.4f}" for lp in lps]
            #     log_args.append(lps)
            #     # Get recent estimate of success rates
            #     pfasts = curriculum.metric_for_tasks(task_names, metric="p_fast")
            #     pfasts = [f"{pfast:.4f}" for pfast in pfasts]
            #     log_args.append(pfasts)
            logging.info(log_str, *log_args)
            if flags.curriculum:
                curriculum.log_metrics(step)

            # Stop training if sps remains 0 for too long
            if total_loss != float("inf"):
                for i in reversed(range(1, len(recent_sps))):
                    recent_sps[i] = recent_sps[i-1]
                recent_sps[0] = sps
                if sum(recent_sps) == 0.0:
                    return

    except KeyboardInterrupt:
        logging.warning("Quitting.")
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    logfile.close()


def test(flags, num_episodes=1):
    flags.savedir = os.path.expandvars(os.path.expanduser(flags.savedir))
    print("savedir:" + str(flags.savedir))
    checkpointpath = flags.custompath if flags.custompath else os.path.join(flags.savedir, "latest", "model.tar")

    #gym_env = create_env(flags.env, save_ttyrecs=flags.save_ttyrecs)
    observation_keys = ("glyphs", "blstats")
    observation_keys = ("glyphs", "blstats", "inv_glyphs", "inv_strs", "inv_letters", "inv_oclasses",
                            "message", "tty_chars", "tty_colors", "tty_cursor")
    gym_env = create_env(flags.env, observation_keys=observation_keys)
    env = ResettingEnvironment(gym_env)
    model = Net(gym_env.observation_space, gym_env.action_space.n, flags.use_lstm, goal=flags.curriculum)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    observation = env.initial()
    returns = []

    agent_state = model.initial_state(batch_size=1)
    last_inv = []

    while len(returns) < num_episodes:
        if flags.mode == "test_render":
            if flags.item_frames or flags.reward_frames:
                #The following lines parse the inventory and store it as an array in all_items_txt
                all_items_text = []
                all_items_encoded = observation["inv_strs"][0][0].numpy()
                for x in range(55):
                    if(all_items_encoded[x][0] != 0):
                        all_items_text.append(''.join([str((chr(elem))) for elem in all_items_encoded[x][all_items_encoded[x] != 0]]))
                if observation["reward"].item() > 0 and flags.reward_frames or all_items_text != last_inv and flags.item_frames:
                    env.gym_env.render()
                    print("Reward is %d" % observation["reward"].item())
                    print(all_items_text)
                    last_inv = all_items_text
            elif flags.message:
                #Parse and print the message
                message_encoded = observation["message"][0][0]
                message = ""
                for x in range(256):
                    message += chr(message_encoded[x].item())
                print(message)
            else:
                env.gym_env.render()
        policy_outputs, agent_state = model(observation, agent_state)
        observation = env.step(policy_outputs["action"])
        if observation["done"].item():
            last_inv = []
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
    env.close()
    logging.info(
        "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
    )


class RandomNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm):
        super(RandomNet, self).__init__()
        del observation_shape, use_lstm
        self.num_actions = num_actions
        self.theta = torch.nn.Parameter(torch.zeros(self.num_actions))

    def forward(self, inputs, core_state):
        # print(inputs)
        T, B, *_ = inputs["observation"].shape
        zeros = self.theta * 0
        # set logits to 0
        policy_logits = zeros[None, :].expand(T * B, -1)
        # set baseline to 0
        baseline = policy_logits.sum(dim=1).view(-1, B)

        # sample random action
        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1).view(
            T, B
        )
        policy_logits = policy_logits.view(T, B, self.num_actions)
        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )

    def initial_state(self, batch_size):
        return ()


def _step_to_range(delta, num_steps):
    """
    Range of `num_steps` values separated by distance `delta` centered around zero.
    Given an image with x,y in [-1, 1], return represents the crop range and sampling points on one axis.
    """
    delta_range = delta * torch.arange(-(num_steps // 2), (num_steps + 1) // 2)
    return delta_range


class Crop(nn.Module):
    """Helper class for NetHackNet below."""

    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        self.width = width                  # 79
        self.height = height                # 21
        self.width_target = width_target    # 9
        self.height_target = height_target  # 9

        # Create row-wise sampling grid and repeat across height_target rows
        width_grid = _step_to_range(2 / (self.width - 1), self.width_target)[None, :].expand(self.height_target, -1)
        # Create column-wise sampling grid and repeat across width_target columns
        height_grid = _step_to_range(2 / (self.height - 1), height_target)[:, None].expand(-1, self.width_target)

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.
        Args:
           inputs [B x H x W]
           coordinates [B x 2] x,y coordinates
        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        # Add another axis at 1
        inputs = inputs[:, None, :, :].float()

        # Extract x,y coordinates for each sample
        x = coordinates[:, 0]
        y = coordinates[:, 1]

        # Recenter x and y values around 0 and normalize to range [-1, 1]
        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        # Shifts sampling grid to be centered around x, y
        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        # TODO: only cast to int if original tensor was int
        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
            .squeeze(1)
            .long()
        )


class NetHackNet(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_actions,
        use_lstm,
        embedding_dim=32,
        crop_dim=9,
        num_layers=5,
        goal=False
    ):
        super(NetHackNet, self).__init__()
        self.goal = goal

        self.glyph_shape = observation_shape["glyphs"].shape            # (21, 79)
        self.blstats_size = observation_shape["blstats"].shape[0]       # 27
        if self.goal:
            self.goal_size = observation_shape["goal"].shape[0]        # 785?

        self.num_actions = num_actions                                  # 23
        self.use_lstm = use_lstm

        self.H = self.glyph_shape[0]                                    # 21
        self.W = self.glyph_shape[1]                                    # 79

        self.k_dim = embedding_dim                                      # 32
        self.h_dim = 512

        self.crop_dim = crop_dim                                        # 9

        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)        # 5976, 32

        K = embedding_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = num_layers  # number of convnet layers                      # 5

        in_channels = [K] + [M] * (L - 1)                               # [32, 16, 16, 16, 16]
        out_channels = [M] * (L - 1) + [Y]                              # [16, 16, 16, 16, 8]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        # Create a sequential net of alternating Conv2d and ELU layers
        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        # CNN crop model.
        conv_extract_crop = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        # Create a sequential net of alternating Conv2d and ELU layers
        self.extract_crop_representation = nn.Sequential(
            *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
        )

        # Blstats output dim
        out_dim = self.k_dim                                            # 32
        # Map glyphs output dim
        out_dim += self.H * self.W * Y                                  # 32 + 21 * 79 * 8 = 13304
        # Cropped map glyphs output dim
        out_dim += self.crop_dim**2 * Y                                 # 13304 + 9 * 9 * 8 = 13952
        if self.goal:
            # Goal output dim
            out_dim += int(self.goal_size / 2)                          # 13952 + 392 = 14344

        # Blstats encoding 27 to 32
        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )

        # Goal encoding 27 to 32
        if self.goal:
            goal_dim = int(self.goal_size / 2)
            self.embed_goals = nn.Sequential(
                nn.Linear(self.goal_size, goal_dim),
                nn.ReLU(),
                nn.Linear(goal_dim, goal_dim),
                nn.ReLU(),
            )

        # Fully connected from embedding to policy 13304 to 512
        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        if self.use_lstm:
            # LSTM 512 to 512
            self.core = nn.LSTM(self.h_dim, self.h_dim, num_layers=1)

        # Policy 512 to 23
        self.policy = nn.Linear(self.h_dim, self.num_actions)
        # Baseline 512 to 1
        self.baseline = nn.Linear(self.h_dim, 1)

    def initial_state(self, batch_size=1):
        if self.use_lstm:
            return tuple(
                torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
                for _ in range(2)
            )
        return tuple()

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, env_outputs, core_state):
        # Set up glyphs
        # -- [T x B x H x W]
        glyphs = env_outputs["glyphs"]                                  # [1, 1, 21, 79]
        T, B, *_ = glyphs.shape
        # -- [B' x H x W]
        glyphs = torch.flatten(glyphs, 0, 1)  # Merge time and batch.   # [1, 21, 79]
        # -- [B x H x W]
        glyphs = glyphs.long()

        # Bottom line stats and extract coordinates
        # -- [T x B x F]
        blstats = env_outputs["blstats"]                                # [1, 1, 27]
        # -- [B' x F]
        blstats = blstats.view(T * B, -1).float()                       # [1, 27]
        # -- [B x 2] x,y coordinates
        coordinates = blstats[:, :2]                                    # [1, 2]
        # TODO ???
        # coordinates[:, 0].add_(-1)
        # -- [B x F]
        # TODO: Remove?
        blstats = blstats.view(T * B, -1).float()                       # [1, 27]
        # -- [B x K]
        blstats_emb = self.embed_blstats(blstats)                       # [1, 32]
        assert blstats_emb.shape[0] == T * B
        reps = [blstats_emb]

        # Cropped map glyphs
        # -- [B x H' x W']
        # Crop glyphs observation around player x,y coordinates
        crop = self.crop(glyphs, coordinates)                           # [1, 9, 9]
        # -- [B x H' x W' x K]
        # Embed glyphs
        crop_emb = self._select(self.embed, crop)                       # [1, 9, 9, 32]
        # CNN crop model.
        # -- [B x K x W' x H']
        crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow?           # [1, 32, 9, 9]
        # -- [B x W' x H' x K]
        # Convolutional pass to get representation of cropped region    # [1, 8, 9, 9]
        crop_rep = self.extract_crop_representation(crop_emb)
        # -- [B x K']
        crop_rep = crop_rep.view(T * B, -1)                             # [1, 648]
        assert crop_rep.shape[0] == T * B
        reps.append(crop_rep)

        # -- [B x H x W x K]
        # Full map glyphs
        glyphs_emb = self._select(self.embed, glyphs)                   # [1, 21, 79, 32]
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow?       # [1, 32, 79, 21]
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb)            # [1, 8, 79, 21]
        # -- [B x K']
        glyphs_rep = glyphs_rep.view(T * B, -1)                         # [1, 13272]
        assert glyphs_rep.shape[0] == T * B
        # -- [B x K'']
        reps.append(glyphs_rep)

        # TODO: Goals
        if self.goal:
            # -- [T x B x F]
            goals = env_outputs["goal"]                                # [1, 1, 785]
            # -- [B' x F]
            goals = goals.view(T * B, -1).float()                       # [1, 785]
            # -- [B x F]
            # TODO: Remove?
            goals = goals.view(T * B, -1).float()                       # [1, 785]
            # -- [B x K]
            goals_emb = self.embed_goals(goals)                       # [1, 32]
            assert goals_emb.shape[0] == T * B
            reps.append(goals_emb)

        # [32, 648, 13272]
        st = torch.cat(reps, dim=1)

        # -- [B x K]
        st = self.fc(st)                                                # [1, 512]

        if self.use_lstm:
            core_input = st.view(T, B, -1)
            core_output_list = []
            notdone = (~env_outputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = st

        # -- [B x A]
        policy_logits = self.policy(core_output)
        # -- [B x A]
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )


Net = NetHackNet


def main(flags, wandb=None):
    if flags.mode == "train":
        train(flags, wandb_run=wandb)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parse_args()

    if flags.profile:
        import cProfile
        from pstats import Stats
        pr = cProfile.Profile()
        pr.enable()

    wandb_run = None
    if flags.exp_name:
        wandb_run = wandb.init(
                    project="syllabus",
                    entity="ENTITY",
                    config=flags,
                    save_code=True,
                    name=flags.exp_name,
                    resume="allow",
                    id=f"{flags.exp_name}-{flags.wandb_id}"
                  )
    main(flags, wandb=wandb_run)

    if flags.profile:
        pr.disable()
        stats = Stats(pr)
        stats.sort_stats('cumtime').print_stats(200)
