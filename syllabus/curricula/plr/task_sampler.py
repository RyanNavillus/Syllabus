# Code heavily based on the original Prioritized Level Replay implementation from https://github.com/facebookresearch/level-replay
# If you use this code, please cite the above codebase and original PLR paper: https://arxiv.org/abs/2010.03934

from typing import List

import gymnasium as gym
import numpy as np
import torch
from syllabus.curricula.plr.storage import RolloutStorage
from syllabus.core.evaluator import Evaluator
from syllabus.task_space.task_space import TaskSpace


class TaskSampler:
    """ Task sampler for Prioritized Level Replay (PLR)

    Args:
        tasks (list): List of tasks to sample from
        action_space (gym.spaces.Space): Action space of the environment
        num_actors (int): Number of actors/processes
        strategy (str): Strategy for sampling tasks. One of "value_l1", "gae", "policy_entropy", "least_confidence", "min_margin", "one_step_td_error".
        replay_schedule (str): Schedule for sampling replay levels. One of "fixed" or "proportionate".
        score_transform (str): Transform to apply to task scores. One of "constant", "max", "eps_greedy", "rank", "power", "softmax".
        temperature (float): Temperature for score transform. Increasing temperature makes the sampling distribution more uniform.
        eps (float): Epsilon for eps-greedy score transform.
        rho (float): Proportion of seen tasks before replay sampling is allowed.
        nu (float): Probability of sampling a replay level if using a fixed replay_schedule.
        alpha (float): Linear interpolation weight for score updates. 0.0 means only use old scores, 1.0 means only use new scores.
        staleness_coef (float): Linear interpolation weight for task staleness vs. task score. 0.0 means only use task score, 1.0 means only use staleness.
        staleness_transform (str): Transform to apply to task staleness. One of "constant", "max", "eps_greedy", "rank", "power", "softmax".
        staleness_temperature (float): Temperature for staleness transform. Increasing temperature makes the sampling distribution more uniform.
        eval_envs (List[gym.Env]): List of evaluation environments
        action_value_fn (callable): A function that takes an observation as input and returns an action and value.
    """

    def __init__(
        self,
        tasks: list,
        num_steps: int,
        task_space: TaskSpace = None,
        action_space: gym.spaces.Space = None,
        num_actors: int = 1,
        strategy: str = "value_l1",
        replay_schedule: str = "proportionate",
        score_transform: str = "rank",
        temperature: float = 0.1,
        eps: float = 0.05,
        rho: float = 1.0,
        nu: float = 0.5,
        alpha: float = 1.0,
        num_processes: int = 1,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        staleness_coef: float = 0.1,
        staleness_transform: str = "power",
        staleness_temperature: float = 1.0,

        robust_plr: bool = False,
        eval_envs=None,
        evaluator: Evaluator = None,
        observation_space=None,

    ):
        self.task_space = task_space
        self.action_space = action_space
        self.tasks = tasks
        self.num_tasks = len(self.tasks)
        self.num_steps = num_steps

        self.strategy = strategy
        self.replay_schedule = replay_schedule
        self.score_transform = score_transform
        self.temperature = temperature
        self.eps = eps
        self.rho = rho
        self.nu = nu
        self.alpha = float(alpha)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature
        self.robust_plr = robust_plr
        self.eval_envs = eval_envs
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.evaluator = evaluator
        self.observation_space = observation_space

        self.unseen_task_weights = np.array([1.0] * self.num_tasks)
        self.task_scores = np.array([0.0] * self.num_tasks, dtype=float)
        self.partial_task_scores = np.zeros((num_actors, self.num_tasks), dtype=float)
        self.partial_task_steps = np.zeros((num_actors, self.num_tasks), dtype=np.int64)
        self.task_staleness = np.array([0.0] * self.num_tasks, dtype=float)

        self._robust_rollouts = RolloutStorage(
            self.num_steps,
            self.num_processes,
            self.requires_value_buffers,
            action_space=action_space,
        )

        self.next_task_index = 0  # Only used for sequential strategy

        # Logging metrics
        self._last_score = 0.0

        if not self.requires_value_buffers and self.action_space is None:
            raise ValueError(
                'Must provide action space to PLR if using "policy_entropy", "least_confidence", or "min_margin" strategies'
            )

    def _get_score_function(self):
        if self.strategy == "policy_entropy":
            return self._average_entropy
        elif self.strategy == "least_confidence":
            return self._average_least_confidence
        elif self.strategy == "min_margin":
            return self._average_min_margin
        elif self.strategy == "gae":
            return self._average_gae
        elif self.strategy == "value_l1":
            return self._average_value_l1
        elif self.strategy == "one_step_td_error":
            return self._one_step_td_error
        else:
            raise ValueError(f"Unsupported strategy, {self.strategy}")

    def update_with_rollouts(self, rollouts, actor_id=None):
        if self.strategy == "random":
            return
        score_function = self._get_score_function()
        self._update_with_rollouts(rollouts, score_function, actor_index=actor_id)

    def update_with_episode_data(self, episode_data):
        if self.strategy == "random":
            return
        score_function = self._get_score_function()
        self._update_with_episode_data(episode_data, score_function)

    def update_task_score(self, actor_index, task_idx, score, num_steps):
        score = self._partial_update_task_score(actor_index, task_idx, score, num_steps, done=True)

        self.unseen_task_weights[task_idx] = 0.0  # No longer unseen

        old_score = self.task_scores[task_idx]
        self.task_scores[task_idx] = (1.0 - self.alpha) * old_score + self.alpha * score

    def _partial_update_task_score(self, actor_index, task_idx, score, num_steps, done=False):
        partial_score = self.partial_task_scores[actor_index][task_idx]
        partial_num_steps = self.partial_task_steps[actor_index][task_idx]

        running_num_steps = partial_num_steps + num_steps
        merged_score = partial_score + (score - partial_score) * num_steps / float(running_num_steps)
        if done:
            self.partial_task_scores[actor_index][task_idx] = 0.0  # zero partial score, partial num_steps
            self.partial_task_steps[actor_index][task_idx] = 0
        else:
            self.partial_task_scores[actor_index][task_idx] = merged_score
            self.partial_task_steps[actor_index][task_idx] = running_num_steps

        return merged_score

    def _average_entropy(self, **kwargs):
        episode_logits = kwargs["episode_logits"]
        num_actions = self.action_space.n
        max_entropy = -(1.0 / num_actions) * np.log(1.0 / num_actions) * num_actions

        return (-torch.exp(episode_logits) * episode_logits).sum(-1).mean().item() / max_entropy

    def _average_least_confidence(self, **kwargs):
        episode_logits = kwargs["episode_logits"]
        return (1 - torch.exp(episode_logits.max(-1, keepdim=True)[0])).mean().item()

    def _average_min_margin(self, **kwargs):
        episode_logits = kwargs["episode_logits"]
        top2_confidence = torch.exp(episode_logits.topk(2, dim=-1)[0])
        return 1 - (top2_confidence[:, 0] - top2_confidence[:, 1]).mean().item()

    def _average_gae(self, **kwargs):
        returns = kwargs["returns"]
        value_preds = kwargs["value_preds"]

        advantages = returns - value_preds

        return advantages.mean().item()

    def _average_value_l1(self, **kwargs):
        returns = kwargs["returns"]
        value_preds = kwargs["value_preds"]

        advantages = returns - value_preds

        return advantages.abs().mean().item()

    def _one_step_td_error(self, **kwargs):
        rewards = kwargs["rewards"]
        value_preds = kwargs["value_preds"]

        max_t = len(rewards)
        td_errors = (rewards[:-1] + value_preds[: max_t - 1] - value_preds[1:max_t]).abs()
        assert not torch.isnan(
            td_errors.abs().mean()
        ), f"Got invalid values for 'rewards' or 'value_preds'. Check that reward length: {len(rewards)}"
        return td_errors.abs().mean().item()

    @property
    def requires_value_buffers(self):
        return self.strategy in ["gae", "value_l1", "one_step_td_error"]

    def _update_with_scores(self, rollouts):
        tasks = rollouts.tasks
        scores = rollouts.scores
        done = ~(rollouts.masks > 0)
        num_actors = rollouts.tasks.shape[1]

        for actor_index in range(num_actors):
            done_steps = done[:, actor_index].nonzero()[:self.num_steps, 0]
            start_t = 0

            for t in done_steps:
                if not start_t < self.num_steps:
                    break

                if (t == 0):  # if t is 0, then this done step caused a full update of previous last cycle
                    continue

                task_idx_t = tasks[start_t, actor_index].item()

                score = scores[start_t:t, actor_index].mean().item()
                num_steps = t - start_t
                self.update_task_score(actor_index, task_idx_t, score, num_steps)
                start_t = t.item()
            if start_t < self.num_steps:
                task_idx_t = tasks[start_t, actor_index].item()

                score = scores[start_t:, actor_index].mean().item()
                self._last_score = score
                num_steps = self.num_steps - start_t
                self._partial_update_task_score(actor_index, task_idx_t, score, num_steps)

    def _update_with_rollouts(self, rollouts, score_function, actor_index=None):
        tasks = rollouts.tasks
        if not self.requires_value_buffers:
            policy_logits = rollouts.action_log_dist
        done = ~(rollouts.masks > 0)
        num_actors = rollouts.tasks.shape[1]

        actors = [actor_index] if actor_index is not None else range(num_actors)
        for actor_index in actors:
            done_steps = done[:, actor_index].nonzero()[:self.num_steps, 0]
            start_t = 0

            for t in done_steps:
                if not start_t < self.num_steps:
                    break

                if (t == 0):  # if t is 0, then this done step caused a full update of previous last cycle
                    continue

                # If there is only 1 step, we can't calculate the one-step td error
                if self.strategy == "one_step_td_error" and t - start_t <= 1:
                    continue

                task_idx_t = tasks[start_t, actor_index].item()

                # Store kwargs for score function
                score_function_kwargs = {}
                if self.requires_value_buffers:
                    score_function_kwargs["returns"] = rollouts.returns[start_t:t, actor_index]
                    score_function_kwargs["rewards"] = rollouts.rewards[start_t:t, actor_index]
                    score_function_kwargs["value_preds"] = rollouts.value_preds[start_t:t, actor_index]
                else:
                    episode_logits = policy_logits[start_t:t, actor_index]
                    score_function_kwargs["episode_logits"] = torch.log_softmax(episode_logits, -1)
                score = score_function(**score_function_kwargs)
                num_steps = len(rollouts.tasks[start_t:t, actor_index])
                self.update_task_score(actor_index, task_idx_t, score, num_steps)

                start_t = t.item()
            if start_t < self.num_steps:
                # If there is only 1 step, we can't calculate the one-step td error
                if self.strategy == "one_step_td_error" and start_t == self.num_steps - 1:
                    continue
                task_idx_t = tasks[start_t, actor_index].item()

                # Store kwargs for score function
                score_function_kwargs = {}
                if self.requires_value_buffers:
                    score_function_kwargs["returns"] = rollouts.returns[start_t:, actor_index]
                    score_function_kwargs["rewards"] = rollouts.rewards[start_t:, actor_index]
                    score_function_kwargs["value_preds"] = rollouts.value_preds[start_t:, actor_index]
                else:
                    episode_logits = policy_logits[start_t:, actor_index]
                    score_function_kwargs["episode_logits"] = torch.log_softmax(episode_logits, -1)

                score = score_function(**score_function_kwargs)
                self._last_score = score
                num_steps = len(rollouts.tasks[start_t:, actor_index])
                self._partial_update_task_score(actor_index, task_idx_t, score, num_steps)

    def after_update(self, actor_indices=None):
        # Reset partial updates, since weights have changed, and thus logits are now stale
        actor_indices = range(self.partial_task_scores.shape[0]) if actor_indices is None else actor_indices
        for actor_index in actor_indices:
            for task_idx in range(self.partial_task_scores.shape[1]):
                if self.partial_task_scores[actor_index][task_idx] != 0:
                    self.update_task_score(actor_index, task_idx, 0, 0)
        self.partial_task_scores.fill(0)
        self.partial_task_steps.fill(0)

    def _update_staleness(self, selected_idx):
        if self.staleness_coef > 0:
            self.task_staleness = self.task_staleness + 1
            self.task_staleness[selected_idx] = 0

    def _sample_replay_level(self):
        sample_weights = self.sample_weights()
        if np.isclose(np.sum(sample_weights), 0):
            sample_weights = np.ones_like(sample_weights, dtype=float) / len(sample_weights)

        task_idx = np.random.choice(range(self.num_tasks), 1, p=sample_weights)[0]

        self._update_staleness(task_idx)

        return task_idx

    def _sample_unseen_level(self):
        sample_weights = self.unseen_task_weights / self.unseen_task_weights.sum()
        task_idx = np.random.choice(range(self.num_tasks), 1, p=sample_weights)[0]
        self._update_staleness(task_idx)

        return task_idx

    def _evaluate_unseen_level(self):
        sample_weights = self.unseen_task_weights / self.unseen_task_weights.sum()
        task_idx = np.random.choice(range(self.num_tasks), 1, p=sample_weights)[0]
        task = self.tasks[task_idx]

        episode_data = self.evaluate_task(task)
        self.update_with_episode_data(episode_data)

        self._update_staleness(task_idx)

    def evaluate_task(self, task):
        # TODO: Set task for evaluator envs
        task_encoded = self.task_space.encode(task)
        eval_envs = self.evaluator.create_eval_envs()
        obs, _ = eval_envs.reset(seed=list(range(eval_envs.num_envs)), options={"seed_task": True})
        done = False
        # TODO: Support any number of eval processes
        # TODO: Figure out how to generate roughly 1 episode of data for each task?
        while not done:
            action, value, _ = self.evaluator.get_action_and_value(obs)
            obs, rew, term, trunc, infos = eval_envs.step(action)

            mask = -torch.Tensor(np.logical_or(term, trunc)).unsqueeze(-1)
            self._robust_rollouts.insert(mask, value_preds=value, rewards=torch.Tensor(
                rew).unsqueeze(-1), tasks=torch.Tensor([task_encoded]))

            # Check if the episode is done
            if "final_info" in infos:
                for i, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        print(info["episode"])
                        assert False

        next_value = self.evaluator.get_value(obs)
        self._robust_rollouts.compute_returns(next_value, self.gamma, self.gae_lambda)
        return {
            "tasks": self._robust_rollouts.tasks,
            "masks": self._robust_rollouts.masks,
            "rewards": self._robust_rollouts.rewards,
            "value_preds": self._robust_rollouts.value_preds,
            "returns": self._robust_rollouts.returns,
        }

    def sample(self, strategy=None):
        if not strategy:
            strategy = self.strategy

        if strategy == "random":
            task_idx = np.random.choice(range(self.num_tasks))
            task = self.tasks[task_idx]
            return task

        if strategy == "sequential":
            task_idx = self.next_task_index
            self.next_task_index = (self.next_task_index + 1) % self.num_tasks
            return task_idx

        num_unseen = (self.unseen_task_weights > 0).sum()
        proportion_seen = (self.num_tasks - num_unseen) / self.num_tasks

        if self.replay_schedule == "fixed":
            if proportion_seen >= self.rho:
                # Sample replay level with fixed prob = 1 - nu OR if all levels seen
                if np.random.rand() > self.nu or not proportion_seen < 1.0:
                    return self._sample_replay_level()

            # Otherwise, sample a new level
            if self.robust_plr:
                self._evaluate_unseen_level()
                self._robust_rollouts.after_update()
                self.after_update()
                return self.sample(strategy=strategy)
            else:
                return self._sample_unseen_level()

        elif self.replay_schedule == "proportionate":
            if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
                return self._sample_replay_level()
            else:
                if self.robust_plr:
                    self._evaluate_unseen_level()
                    self._robust_rollouts.after_update()
                    self.after_update()
                    return self.sample(strategy=strategy)
                else:
                    return self._sample_unseen_level()
        else:
            raise NotImplementedError(
                f"Unsupported replay schedule: {self.replay_schedule}. Must be 'fixed' or 'proportionate'.")

    def _update_with_episode_data(self, episode_data, score_function):
        tasks = episode_data["tasks"]
        if not self.requires_value_buffers:
            policy_logits = episode_data.action_log_dist
        done = ~(episode_data["masks"] > 0)

        total_steps, num_actors = tasks.shape[:2]

        for actor_index in range(num_actors):
            # TODO: Track episode_length in episode_data and only update up to end of episode. Or write single episode update funtion
            done_steps = done[:, actor_index].nonzero()[:total_steps, 0]
            start_t = 0

            for t in done_steps:
                if not start_t < total_steps:
                    break

                if (t == 0):  # if t is 0, then this done step caused a full update of previous last cycle
                    continue

                # If there is only 1 step, we can't calculate the one-step td error
                if self.strategy == "one_step_td_error" and t - start_t <= 1:
                    continue

                task_idx_t = tasks[start_t, actor_index].item()

                # Store kwargs for score function
                score_function_kwargs = {}
                if self.requires_value_buffers:
                    score_function_kwargs["returns"] = episode_data["returns"][start_t:t, actor_index]
                    score_function_kwargs["rewards"] = episode_data["rewards"][start_t:t, actor_index]
                    score_function_kwargs["value_preds"] = episode_data["value_preds"][start_t:t, actor_index]
                else:
                    episode_logits = policy_logits[start_t:t, actor_index]
                    score_function_kwargs["episode_logits"] = torch.log_softmax(episode_logits, -1)
                score = score_function(**score_function_kwargs)
                num_steps = len(episode_data["tasks"][start_t:t, actor_index])
                self.update_task_score(actor_index, task_idx_t, score, num_steps)

                start_t = t.item()
            if start_t < total_steps:
                # If there is only 1 step, we can't calculate the one-step td error
                if self.strategy == "one_step_td_error" and start_t == total_steps - 1:
                    continue
                task_idx_t = tasks[start_t, actor_index].item()

                # Store kwargs for score function
                score_function_kwargs = {}
                if self.requires_value_buffers:
                    score_function_kwargs["returns"] = episode_data["returns"][start_t:, actor_index]
                    score_function_kwargs["rewards"] = episode_data["rewards"][start_t:, actor_index]
                    score_function_kwargs["value_preds"] = episode_data["value_preds"][start_t:, actor_index]
                else:
                    episode_logits = policy_logits[start_t:, actor_index]
                    score_function_kwargs["episode_logits"] = torch.log_softmax(episode_logits, -1)

                score = score_function(**score_function_kwargs)
                self._last_score = score
                num_steps = len(episode_data["tasks"][start_t:, actor_index])
                self._partial_update_task_score(actor_index, task_idx_t, score, num_steps)

    def sample_weights(self):
        weights = self._score_transform(self.score_transform, self.temperature, self.task_scores)
        weights = weights * (1 - self.unseen_task_weights)  # zero out unseen levels
        z = np.sum(weights)
        if z > 0:
            weights /= z

        staleness_weights = 0
        if self.staleness_coef > 0:
            staleness_weights = self._score_transform(
                self.staleness_transform,
                self.staleness_temperature,
                self.task_staleness,
            )
            staleness_weights = staleness_weights * (1 - self.unseen_task_weights)
            z = np.sum(staleness_weights)
            if z > 0:
                staleness_weights /= z
            weights = (1 - self.staleness_coef) * weights + self.staleness_coef * staleness_weights
        return weights

    def _score_transform(self, transform, temperature, scores):
        if transform == "constant":
            weights = np.ones_like(scores)
        if transform == "max":
            weights = np.zeros_like(scores)
            scores = scores[:]
            scores[self.unseen_task_weights > 0] = -float("inf")  # only argmax over seen levels
            argmax = np.random.choice(np.flatnonzero(np.isclose(scores, scores.max())))
            weights[argmax] = 1.0
        elif transform == "eps_greedy":
            weights = np.zeros_like(scores)
            weights[scores.argmax()] = 1.0 - self.eps
            weights += self.eps / self.num_tasks
        elif transform == "rank":
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1 / ranks ** (1.0 / temperature)
        elif transform == "power":
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores) + eps) ** (1.0 / temperature)
        elif transform == "softmax":
            weights = np.exp(np.array(scores) / temperature)

        return weights

    def metrics(self):
        return {
            "task_scores": self.task_scores,
            "unseen_task_weights": self.unseen_task_weights,
            "task_staleness": self.task_staleness,
            "proportion_seen": (self.num_tasks - (self.unseen_task_weights > 0).sum()) / self.num_tasks,
            "score": self._last_score,
        }
