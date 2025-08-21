# This Syllabus demo script is based on the CleanRL codebase https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Box
from torch.distributions.categorical import Categorical

from syllabus.core import (Curriculum, GymnasiumSyncWrapper, TaskWrapper,
                           make_multiprocessing_curriculum)
from syllabus.task_space import BoxTaskSpace


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 250000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
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

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


args = Args
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)
args.num_iterations = args.total_timesteps // args.batch_size

# TRY NOT TO MODIFY: seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


# SYLLABUS: We need to select a task space that our curriculum learning method will choose tasks from.
# Here we define a simple range of continuous tasks from -0.2 to 0.2
# This is roughly the maximum range of valid CartPole angles
task_space = BoxTaskSpace(Box(low=-0.2, high=0.2, shape=(1,), dtype=np.float32))


# SYLLABUS: We will use a TaskWrapper to allow us to set the initial angle of the pole during each reset()
class CartPoleTaskWrapper(TaskWrapper):
    def __init__(self, env, task_space, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.task_space = task_space

    def reset(self, seed=None, options=None, new_task=None):
        _, info = self.env.reset(seed=seed, options=options)

        # If no task is provided, choose a random one from the task space
        if new_task is None:
            new_task = self.task_space.sample()
        self.task = new_task

        # Change the initial pole angle to the new task
        self.env.unwrapped.state[2] = new_task
        return np.array(self.env.unwrapped.state, dtype=np.float32), info


# SYLLABUS: You can render the environment to see that when we choose the task 0.2
# the pole always starts at the same angle, tilted to the right
render_env = gym.make("CartPole-v1", render_mode="rgb_array")
render_env = CartPoleTaskWrapper(render_env, task_space)
render_env.reset(new_task=0.2)
plt.figure()
plt.ion()
img = plt.imshow(render_env.render())
plt.show()
plt.pause(0.03)

ep_length = 0
for i in range(100):
    obs, rew, term, trunc, info = render_env.step(render_env.action_space.sample())
    img.set_data(render_env.render())
    plt.pause(0.03)
    ep_length += 1

    if term or trunc:
        obs, info = render_env.reset(new_task=0.2)
        ep_length = 0
plt.close()


# SYLLABUS: We need to modify the make_env function to include our task wrapper and synchronization wrapper
def make_env(env_id, idx, task_space, components=None):
    def thunk():
        env = gym.make("CartPole-v1")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = CartPoleTaskWrapper(env, task_space)         # Add task wrapper
        if components is not None:
            assert task_space is not None, "task_space must be provided if components are provided"
            env = GymnasiumSyncWrapper(env, task_space, components)         # Add synchronization wrapper
        return env
    return thunk


# SYLLABUS: We can define a simple curriculum which will start training with a small range
# of possible pole angles, then slowly expand it in stages as the agent improves.
# Specifically we will increase the range 10 times, every time the mean return over the past
# 10 episodes exceeds 300, which is 60% of the maximum possible return in CartPole.
class RangeCurriculum(Curriculum):
    def __init__(self, task_space, stages=10, return_threshold=300.0, **kwargs):
        assert isinstance(task_space, BoxTaskSpace), "RangeCurriculum requires a BoxTaskSpace"
        super().__init__(task_space, **kwargs)
        self.return_threshold = return_threshold
        self.current_stage = 0

        # Get the full range of possible tasks
        self.min_task = self.task_space.gym_space.low[0]
        self.max_task = self.task_space.gym_space.high[0]

        # Set the initial range to the center point and set stepsize per stage
        center = self.min_task + (self.max_task - self.min_task) / 2
        self.range = [center, center]
        self.stepsize = (center - self.min_task) / stages

        # Track episodic returns for past 10 episodes
        self.ten_recent_returns = deque(maxlen=10)

    def sample(self, k):
        # Sample tasks uniformly from the current range
        return [np.random.uniform(self.range[0], self.range[1], size=(k,))]

    def update_on_episode(self, episode_return, length, task, progress, env_id=None):
        self.ten_recent_returns.append(episode_return)

        # Increase the range if the average return over the past 10 episodes exceeds the threshold
        if len(self.ten_recent_returns) == 10 and np.mean(self.ten_recent_returns) > self.return_threshold:
            self.range[0] = max(self.range[0] - self.stepsize, self.min_task)
            self.range[1] = min(self.range[1] + self.stepsize, self.max_task)
            self.ten_recent_returns.clear()
            self.current_stage += 1

    # We're using a simplified version of log_metrics, check the documentation for the real API
    def log_metrics(self):
        return self.current_stage


# SYLLABUS: If you want, you can use some of the curriculum learning methods already included in Syllabus instead.
# from syllabus.curricula import DomainRandomization
# curriculum = DomainRandomization(task_space)
curriculum = RangeCurriculum(task_space)
mp_curriculum = make_multiprocessing_curriculum(curriculum)

# env setup
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, i, task_space, mp_curriculum.components) for i in range(args.num_envs)],
)
assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

# Create Agent
agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)


# SYLLABUS: Evaluation loop
eval_envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, i, task_space) for i in range(args.num_envs)],
)
EVAL_INTERVAL = 50  # Evaluate every 50 batches

# Track a few metrics to plot
eval_results = []
stage_ups = []
last_stage = 0


# SYLLABUS: Evaluate the agent's performance on uniformly sampled tasks
def evaluate(agent, num_episodes=50):
    print(f"Evaluating for {num_episodes} episodes...")
    agent.eval()
    episode_returns = []
    obs, _ = eval_envs.reset(seed=args.seed)
    done = False
    while len(episode_returns) < num_episodes:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        obs, _, terminations, truncations, infos = eval_envs.step(action.cpu().numpy())
        done = np.logical_or(terminations, truncations)
        if "episode" in infos.keys():
            for i in range(len(infos["episode"]["r"])):
                if done[i]:
                    episode_returns.append(infos["episode"]["r"][i])
    agent.train()
    print(f"Evaluation results: {np.mean(episode_returns)}")
    return np.mean(episode_returns)


# SYLLABUS: Lets evaluate the agent before training to see the random policy performance.
print("Running initial evaluation")
result = evaluate(agent)
eval_results.append(result)

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
next_obs, _ = envs.reset(seed=args.seed)
next_obs = torch.Tensor(next_obs).to(device)
next_done = torch.zeros(args.num_envs).to(device)
last_ten_returns = deque(maxlen=10)

for iteration in range(1, args.num_iterations + 1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, args.num_steps):
        global_step += args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
        if "episode" in infos.keys():
            for i in range(len(infos["episode"]["r"])):
                if next_done[i]:
                    last_ten_returns.append(infos["episode"]["r"][i])
                    task = infos['task'][i][0]
                    print(
                        f"global_step={global_step}, episodic_return={infos['episode']['r'][i]} ({np.mean(last_ten_returns)})")
                    # SYLLABUS: We add a bit of optional logging code here to track when the curriculum stage changes
                    current_stage = curriculum.log_metrics()
                    if len(stage_ups) <= 11 and current_stage > last_stage:
                        stage_ups.append(global_step)
                        last_stage = current_stage

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

    # SYLLABUS: Evaluate the agent
    if iteration % EVAL_INTERVAL == 0:
        eval_results.append(evaluate(agent))

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    print(f"Iteration: {iteration}/{args.num_iterations} \t SPS: {int(global_step / (time.time() - start_time))}")
mp_curriculum.stop()
envs.close()

# SYLLABUS: Evaluate the final agent for 250 episodes instead of 50
print("Running final evaluation for 250 episodes...")
final_result = evaluate(agent, num_episodes=250)
eval_results.append(final_result)

# SYLLABUS: Plot the evaluation results
x_axis = [0, 25600, 51200, 76800, 102400, 128000, 153600, 179200, 204800, 230400, 250000]
y_axis = eval_results

plt.figure()
plt.ion()
plt.plot(x_axis, y_axis, marker='o', label='Range Curriculum Evaluation Return')

# plot curriculum stage vs environment steps
for x in stage_ups[1:]:
    plt.axvline(x, color='gray', linestyle='--', label='Curriculum Stage Increase' if x == stage_ups[1] else "")

plt.axhline(y=269, color='red', linestyle='-', label='Domain Randomization Final Performance')

plt.title('Eval Return and Curriculum Stage vs Environment Steps')
plt.xlabel('Environment Steps')
plt.ylabel('Eval Return')
plt.grid(True)
plt.legend()
plt.show()
plt.pause(30.0)
