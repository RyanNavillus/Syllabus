# Code heavily based on the original Prioritized Level Replay implementation from https://github.com/facebookresearch/level-replay
# If you use this code, please cite the above codebase and original PLR paper: https://arxiv.org/abs/2010.03934
import gymnasium as gym
import numpy as np
import torch
from collections import defaultdict, deque
from syllabus.core.evaluator import Evaluator
from syllabus.task_space import TaskSpace

INT32_MAX = 2147483647
np.seterr(all='raise')


class TaskSampler:
    """ Task sampler for Prioritized Level Replay (PLR)

    Args:
        tasks (list): List of tasks to sample from
        action_space (gym.spaces.Space): Action space of the environment
        num_actors (int): Number of actors/processes
        strategy (str): Strategy for sampling tasks. Some possible values include
            "random", "sequential", "policy_entropy", "least_confidence", 
            "min_margin", "gae", "value_l1", "one_step_td_error", "signed_value_loss",
            "positive_value_loss", "grounded_signed_value_loss", "grounded_positive_value_loss",
            "alt_advantage_abs", "uniform", "off".
        replay_schedule (str): Schedule for sampling replay levels. One of "fixed" or "proportionate".
        score_transform (str): Transform to apply to task scores. One of "constant", "max", "eps_greedy", 
            "rank", "power", "softmax", etc.
        temperature (float): Temperature for score transform. Increasing temperature makes the sampling
            distribution more uniform.
        eps (float): Epsilon for eps-greedy score transform.
        rho (float): Proportion of seen tasks before replay sampling is allowed.
        nu (float): Probability of sampling a replay level if using a fixed replay_schedule.
        alpha (float): Linear interpolation weight for score updates (0.0 uses old scores only, 1.0 uses new).
        staleness_coef (float): Linear interpolation weight for task staleness vs. task score.
        staleness_transform (str): Transform to apply to task staleness (e.g. "power").
        staleness_temperature (float): Temperature for staleness transform.
        max_score_coef (float): Interpolation weight for combining max_score and mean_score.
        sample_full_distribution (bool): If True, treat the task space as unbounded and manage a buffer.
        task_buffer_size (int): Size of that buffer if sample_full_distribution is True.
        task_buffer_priority (str): Criterion (e.g. "replay_support") for picking replacement in the buffer.
        use_dense_rewards (bool): If True, uses dense rewards in certain grounded strategies.
        gamma (float): Discount factor for one-step TD-error calculations.
    """

    def __init__(
        self,
        tasks: list,
        num_steps: int,
        action_space: gym.spaces.Space = None,
        task_space: TaskSpace = None,
        num_actors: int = 1,
        strategy: str = "value_l1",
        replay_schedule: str = "proportionate",
        score_transform: str = "rank",
        temperature: float = 0.1,
        eps: float = 0.05,
        rho: float = 1.0,
        nu: float = 0.5,
        alpha: float = 1.0,
        staleness_coef: float = 0.1,
        staleness_transform: str = "power",
        staleness_temperature: float = 1.0,
        max_score_coef=0.0,
        sample_full_distribution=False,
        task_buffer_size=0,
        task_buffer_priority="replay_support",
        use_dense_rewards=False,
        gamma=0.999,
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
        # TODO: Add space for eval actors
        self.num_actors = num_actors

        self.strategy = strategy
        self.replay_schedule = replay_schedule
        self.score_transform = score_transform
        self.temperature = temperature
        self.eps = eps
        self.rho = rho
        self.nu = nu
        self.alpha = float(alpha)
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature
        self.max_score_coef = max_score_coef
        self.sample_full_distribution = sample_full_distribution
        self.task_buffer_size = task_buffer_size
        self.task_buffer_priority = task_buffer_priority
        self.use_dense_rewards = use_dense_rewards
        self.gamma = gamma
        self.score_function = self._get_score_function()

        self._init_task_index(tasks if tasks else [])
        self.num_tasks = len(self.tasks)

        N = self.task_buffer_size if self.sample_full_distribution else self.num_tasks
        self.unseen_task_weights = np.array([1.0] * N)
        self.task_scores = np.array([0.0] * N, dtype=np.float32)
        self.partial_task_scores = np.zeros((num_actors, N), dtype=np.float32)
        self.partial_task_max_scores = np.ones((num_actors, N), dtype=np.float32) * float("-inf")
        self.partial_task_steps = np.zeros((num_actors, N), dtype=np.int32)
        self.task_staleness = np.array([0.0] * N, dtype=np.float32)

        self.running_sample_count = 0
        self.next_task_index = 0  # Only used for sequential strategy

        self.track_solvable = False
        self.grounded_values = None
        if self.strategy.startswith("grounded"):
            self.grounded_values = np.array([np.NINF] * N, dtype=np.float32)

        if self.sample_full_distribution:
            self.task2actor = defaultdict(set)
            self.working_task_buffer_size = 0
            self.staging_task_set = set()
            self.working_task_set = set()
            self.task2timestamp_buffer = {}
            self.partial_task_scores_buffer = [defaultdict(float) for _ in range(num_actors)]
            self.partial_task_max_scores_buffer = [defaultdict(lambda: float("-inf")) for _ in range(num_actors)]
            self.partial_task_steps_buffer = [defaultdict(int) for _ in range(num_actors)]

        self._last_score = 0.0

        # Offline evaluation
        self.offline_queue = []
        self.evaluator = evaluator
        self.eval_envs = eval_envs

    def _init_task_index(self, tasks):
        if tasks:
            self.tasks = np.array(tasks, dtype=np.int64)
            self.task2index = {t: i for i, t in enumerate(tasks)}
        else:
            self.tasks = np.zeros(self.task_buffer_size, dtype=np.int64) - 1
            self.task2index = {}

    def _init_solvable_tracking(self):
        self.track_solvable = True
        self.staging_task2solvable = {}
        n = self.task_buffer_size if self.sample_full_distribution else self.num_tasks
        self.task_solvable = np.ones(n, dtype=bool)

    @property
    def _proportion_filled(self):
        if self.sample_full_distribution:
            return self.working_task_buffer_size / self.task_buffer_size if self.task_buffer_size > 0 else 0.0
        else:
            num_unseen = (self.unseen_task_weights > 0).sum()
            proportion_seen = (len(self.tasks) - num_unseen) / len(self.tasks)
            return proportion_seen

    @property
    def requires_value_buffers(self):
        return self.strategy in [
            "gae",
            "value_l1",
            "signed_value_loss",
            "positive_value_loss",
            "grounded_signed_value_loss",
            "grounded_positive_value_loss",
            "one_step_td_error",
            "alt_advantage_abs",
        ]

    
    def _get_score_function(self):
        if self.strategy in ["random", "off"]:
            return

        if self.strategy == "uniform":
            score_function = self._uniform
        elif self.strategy == "policy_entropy":
            score_function = self._average_entropy
        elif self.strategy == "least_confidence":
            score_function = self._average_least_confidence
        elif self.strategy == "min_margin":
            score_function = self._average_min_margin
        elif self.strategy == "gae":
            score_function = self._average_gae
        elif self.strategy == "value_l1":
            score_function = self._average_value_l1
        elif self.strategy == "signed_value_loss":
            score_function = self._average_signed_value_loss
        elif self.strategy == "positive_value_loss":
            score_function = self._average_positive_value_loss
        elif self.strategy == "grounded_signed_value_loss":
            score_function = self._average_grounded_signed_value_loss
        elif self.strategy == "grounded_positive_value_loss":
            score_function = self._average_grounded_positive_value_loss
        elif self.strategy == "one_step_td_error":
            score_function = self._one_step_td_error
        elif self.strategy == "alt_advantage_abs":
            score_function = self._average_alt_advantage_abs
        else:
            raise ValueError(f"Unsupported strategy, {self.strategy}")
        return score_function

    def update_with_rollouts(self, rollouts, actor_id=None, external_scores=None):
        self._update_with_rollouts(rollouts, actor_index=actor_id, external_scores=external_scores)

        # Evaluate random tasks
        self.num_eval_envs = self.eval_envs.num_envs if self.eval_envs is not None else 0
        while len(self.offline_queue) > self.num_eval_envs:
            tasks = self.offline_queue[:self.num_eval_envs]
            self._evaluate_tasks(tasks)
            self.offline_queue = self.offline_queue[self.num_eval_envs:]


    def update_task_score(self, actor_index, task, score, max_score, num_steps, running_mean=True):
        if self.sample_full_distribution and task in self.staging_task_set:
            score_out, task_idx = self._partial_update_task_score_buffer(
                actor_index, task, score, num_steps, done=True, running_mean=running_mean
            )
        else:
            score_out, task_idx = self._partial_update_task_score(
                actor_index, task, score, max_score, num_steps, done=True, running_mean=running_mean
            )
        return score_out, task_idx

    def _partial_update_task_score(self, actor_index, task, score, max_score, num_steps, done=False, running_mean=True):
        task_idx = self.task2index.get(task, -1)
        old_partial_score = self.partial_task_scores[actor_index][task_idx]
        old_partial_max = self.partial_task_max_scores[actor_index][task_idx]
        old_steps = self.partial_task_steps[actor_index][task_idx]

        new_steps = old_steps + num_steps
        merged_score = old_partial_score + (score - old_partial_score) * num_steps
        if running_mean:
            merged_score /= float(new_steps)

        merged_max = max(old_partial_max, max_score)

        if done:
            self.partial_task_scores[actor_index][task_idx] = 0.0
            self.partial_task_max_scores[actor_index][task_idx] = float("-inf")
            self.partial_task_steps[actor_index][task_idx] = 0
            self.unseen_task_weights[task_idx] = 0.0
            old_score = self.task_scores[task_idx]
            total_score = self.max_score_coef * merged_max + (1.0 - self.max_score_coef) * merged_score
            self.task_scores[task_idx] = (1.0 - self.alpha) * old_score + self.alpha * total_score
            self._last_score = total_score
        else:
            self.partial_task_scores[actor_index][task_idx] = merged_score
            self.partial_task_max_scores[actor_index][task_idx] = merged_max
            self.partial_task_steps[actor_index][task_idx] = new_steps

        return merged_score, task_idx

    def _partial_update_task_score_buffer(self, actor_index, task, score, num_steps, done=False, running_mean=True):
        task_idx = -1
        self.task2actor[task].add(actor_index)
        old_partial_score = self.partial_task_scores_buffer[actor_index].get(task, 0.0)
        old_steps = self.partial_task_steps_buffer[actor_index].get(task, 0)

        new_steps = old_steps + num_steps
        merged_score = old_partial_score + (score - old_partial_score) * num_steps
        if running_mean:
            merged_score /= float(new_steps)

        if done:
            task_idx = self._next_buffer_index
            if self.task_scores[task_idx] <= merged_score or self.unseen_task_weights[task_idx] > 0:
                self.unseen_task_weights[task_idx] = 0.0
                if self.tasks[task_idx] in self.working_task_set:
                    self.working_task_set.remove(self.tasks[task_idx])
                self.working_task_set.add(task)
                self.tasks[task_idx] = task
                self.task2index[task] = task_idx
                self.task_scores[task_idx] = merged_score
                self.partial_task_scores[:, task_idx] = 0.0
                self.partial_task_steps[:, task_idx] = 0
                self.task_staleness[task_idx] = self.running_sample_count - self.task2timestamp_buffer[task]
                self.working_task_buffer_size = min(self.working_task_buffer_size + 1, self.task_buffer_size)
                if self.track_solvable:
                    self.task_solvable[task_idx] = self.staging_task2solvable.get(task, True)
            else:
                task_idx = None

            for a in self.task2actor[task]:
                self.partial_task_scores_buffer[a].pop(task, None)
                self.partial_task_steps_buffer[a].pop(task, None)
                self.partial_task_max_scores_buffer[a].pop(task, None)
            del self.task2timestamp_buffer[task]
            del self.task2actor[task]
            self.staging_task_set.remove(task)
            if self.track_solvable:
                del self.staging_task2solvable[task]
        else:
            self.partial_task_scores_buffer[actor_index][task] = merged_score
            self.partial_task_steps_buffer[actor_index][task] = new_steps

        return merged_score, task_idx

    def _uniform(self, **kwargs):
        return 1.0, 1.0

    def _average_entropy(self, **kwargs):
        episode_logits = kwargs["episode_logits"]
        num_actions = self.action_space.n
        max_entropy = -(1.0 / num_actions) * np.log(1.0 / num_actions) * num_actions
        score_tensor = (-torch.exp(episode_logits) * episode_logits).sum(-1) / max_entropy
        return score_tensor.mean().item(), score_tensor.max().item()

    def _average_least_confidence(self, **kwargs):
        episode_logits = kwargs["episode_logits"]
        score_tensor = 1.0 - torch.exp(episode_logits.max(-1, keepdim=True)[0])
        return score_tensor.mean().item(), score_tensor.max().item()

    def _average_min_margin(self, **kwargs):
        episode_logits = kwargs["episode_logits"]
        top2_confidence = torch.exp(episode_logits.topk(2, dim=-1)[0])
        gap = top2_confidence[:, 0] - top2_confidence[:, 1]
        mean_score = 1.0 - gap.mean().item()
        max_score = 1.0 - gap.min().item()
        return mean_score, max_score

    def _average_gae(self, **kwargs):
        returns = kwargs["returns"]
        value_preds = kwargs["value_preds"]
        advantages = returns - value_preds
        return advantages.mean().item(), advantages.max().item()

    def _average_value_l1(self, **kwargs):
        returns = kwargs["returns"]
        value_preds = kwargs["value_preds"]
        abs_adv = (returns - value_preds).abs()
        return abs_adv.mean().item(), abs_adv.max().item()

    def _average_signed_value_loss(self, **kwargs):
        returns = kwargs["returns"]
        value_preds = kwargs["value_preds"]
        advantages = returns - value_preds
        return advantages.mean().item(), advantages.max().item()

    def _average_positive_value_loss(self, **kwargs):
        returns = kwargs["returns"]
        value_preds = kwargs["value_preds"]
        clipped_adv = (returns - value_preds).clamp(0)
        return clipped_adv.mean().item(), clipped_adv.max().item()

    def _average_grounded_signed_value_loss(self, **kwargs):
        # TODO: Make this work when called from offline eval
        task = kwargs["task"]
        actor_idx = kwargs["actor_index"]
        done = kwargs["done"]
        value_preds = kwargs["value_preds"]
        grounded_val = kwargs.get("grounded_value", None)
        if self.sample_full_distribution and task in self.partial_task_steps_buffer[actor_idx]:
            partial_steps = self.partial_task_steps_buffer[actor_idx][task]
        else:
            task_idx = self.task2index.get(task, None)
            partial_steps = self.partial_task_steps[actor_idx][task_idx] if task_idx is not None else 0

        new_steps = len(kwargs["episode_logits"])
        total_steps = partial_steps + new_steps

        if done and grounded_val is not None:
            if self.use_dense_rewards:
                adv = grounded_val - value_preds[0]
            else:
                adv = grounded_val - value_preds
            mean_score = (total_steps / new_steps) * adv.mean().item()
            max_score = adv.max().item()
        else:
            mean_score, max_score = 0.0, 0.0
        return mean_score, max_score

    def _average_external_score(self, **kwargs):
        done = kwargs["done"]
        external_scores = kwargs["external_scores"]
        if done:
            ms = external_scores.item()
            return ms, ms
        return 0.0, 0.0

    def _average_grounded_positive_value_loss(self, **kwargs):
        # TODO: Make this work when called from offline eval
        task = kwargs["task"]
        actor_idx = kwargs["actor_index"]
        done = kwargs["done"]
        value_preds = kwargs["value_preds"]
        grounded_val = kwargs.get("grounded_value", None)
        if self.sample_full_distribution and task in self.partial_task_steps_buffer[actor_idx]:
            partial_steps = self.partial_task_steps_buffer[actor_idx][task]
        else:
            task_idx = self.task2index.get(task, None)
            partial_steps = self.partial_task_steps[actor_idx][task_idx] if task_idx is not None else 0

        new_steps = len(kwargs["value_preds"])
        total_steps = partial_steps + new_steps
        if done and grounded_val is not None:
            if self.use_dense_rewards:
                adv = grounded_val - value_preds[0]
            else:
                adv = grounded_val - value_preds
            adv = adv.clamp(0)
            mean_score = (total_steps / new_steps) * adv.mean().item()
            max_score = adv.max().item()
        else:
            mean_score, max_score = 0.0, 0.0
        return mean_score, max_score

    def _one_step_td_error(self, **kwargs):
        rewards = kwargs["rewards"]
        value_preds = kwargs["value_preds"]
        max_t = len(rewards)
        if max_t > 1:
            td_errors = (rewards[:-1] + self.gamma * value_preds[1:max_t] - value_preds[: max_t - 1]).abs()
        else:
            td_errors = (rewards[0] - value_preds[0]).abs()
        return td_errors.mean().item(), td_errors.max().item()

    def _average_alt_advantage_abs(self, **kwargs):
        returns = kwargs["alt_returns"]
        value_preds = kwargs["value_preds"]
        abs_adv = (returns - value_preds).abs()
        return abs_adv.mean().item(), abs_adv.max().item()

    @property
    def _next_buffer_index(self):
        if self._proportion_filled < 1.0:
            return self.working_task_buffer_size
        else:
            if self.task_buffer_priority == "replay_support":
                return self.sample_weights().argmin()
            return self.task_scores.argmin()

    @property
    def _has_working_task_buffer(self):
        return (not self.sample_full_distribution) or (
            self.sample_full_distribution and self.task_buffer_size > 0
        )

    def _update_with_rollouts(self, rollouts, actor_index=None, external_scores=None):
        if not self._has_working_task_buffer:
            return
        tasks = rollouts.tasks
        if not self.requires_value_buffers:
            policy_logits = rollouts.action_log_dist
        done = ~(rollouts.masks > 0)
        num_actors = rollouts.tasks.shape[1]

        score_function = self.score_function if external_scores is None else self._average_external_score

        actors = [actor_index] if actor_index is not None else range(num_actors)
        for actor_index in actors:
            done_steps = done[:, actor_index].nonzero()[:self.num_steps, 0]
            start_t = 0

            for t in done_steps:
                if not start_t < self.num_steps:
                    break
                if t == 0:
                    continue

                task_t = tasks[start_t, actor_index].item()
                kwargs_ = {
                    "actor_index": actor_index,
                    "done": True,
                    "task": task_t,
                }
                if not self.requires_value_buffers:
                    ep_logits = policy_logits[start_t:t, actor_index]
                    kwargs_["episode_logits"] = torch.log_softmax(ep_logits, -1)

                if external_scores is not None:
                    kwargs_["external_scores"] = external_scores[actor_index]

                if self.requires_value_buffers:
                    kwargs_["returns"] = rollouts.returns[start_t:t, actor_index]
                    kwargs_["rewards"] = rollouts.rewards[start_t:t, actor_index]
                    if self.strategy == "alt_advantage_abs":
                        kwargs_["alt_returns"] = rollouts.alt_returns[start_t:t, actor_index]

                    # if rollouts.use_popart:
                    #     kwargs_["value_preds"] = rollouts.denorm_value_preds[start_t:t, actor_index]
                    # else:
                    kwargs_["value_preds"] = rollouts.value_preds[start_t:t, actor_index]

                    if self.grounded_values is not None:
                        task_idx_ = self.task2index.get(task_t, None)
                        ret_ = rollouts.rewards[start_t:t].sum(0)[actor_index]
                        if task_idx_ is not None:
                            gv_ = max(self.grounded_values[task_idx_], ret_)
                        else:
                            gv_ = ret_
                        kwargs_["grounded_value"] = gv_

                score, max_score = score_function(**kwargs_)
                num_steps = len(rollouts.tasks[start_t:t, actor_index])
                _, final_task_idx = self.update_task_score(
                    actor_index, task_t, score, max_score, num_steps, running_mean=(external_scores is not None)
                )
                if (
                    self.grounded_values is not None
                    and final_task_idx is not None
                    and "grounded_value" in kwargs_
                    and kwargs_["grounded_value"] is not None
                ):
                    self.grounded_values[final_task_idx] = kwargs_["grounded_value"]
                start_t = t.item()

            if start_t < self.num_steps:
                task_t = tasks[start_t, actor_index].item()
                kwargs_ = {
                    "actor_index": actor_index,
                    "done": False,
                    "task": task_t,
                }
                if not self.requires_value_buffers:
                    ep_logits = policy_logits[start_t:, actor_index]
                    kwargs_["episode_logits"] = torch.log_softmax(ep_logits, -1)

                if external_scores is not None:
                    kwargs_["external_scores"] = external_scores[actor_index]

                if self.requires_value_buffers:
                    kwargs_["returns"] = rollouts.returns[start_t:, actor_index]
                    kwargs_["rewards"] = rollouts.rewards[start_t:, actor_index]
                    if self.strategy == "alt_advantage_abs":
                        kwargs_["alt_returns"] = rollouts.alt_returns[start_t:, actor_index]

                    # if rollouts.use_popart:
                    #     kwargs_["value_preds"] = rollouts.denorm_value_preds[start_t:, actor_index]
                    # else:
                    kwargs_["value_preds"] = rollouts.value_preds[start_t:, actor_index]

                score, max_score = score_function(**kwargs_)
                self._last_score = score
                num_steps = len(rollouts.tasks[start_t:, actor_index])
                if self.sample_full_distribution and task_t in self.staging_task_set:
                    self._partial_update_task_score_buffer(
                        actor_index, task_t, score, num_steps, running_mean=(external_scores is not None))
                else:
                    self._partial_update_task_score(actor_index, task_t, score, max_score,
                                                    num_steps, running_mean=(external_scores is not None))

    def after_update(self, actor_indices=None):
        if not self._has_working_task_buffer:
            return
        actor_indices = range(self.partial_task_scores.shape[0]) if actor_indices is None else actor_indices

        for actor_index in actor_indices:
            for task_idx in range(self.partial_task_scores.shape[1]):
                if self.partial_task_scores[actor_index][task_idx] != 0.0:
                    self.update_task_score(
                        actor_index,
                        self.tasks[task_idx],
                        0.0,
                        float("-inf"),
                        0,
                    )
        self.partial_task_scores.fill(0.0)
        self.partial_task_steps.fill(0.0)

        if self.sample_full_distribution:
            for actor_index in actor_indices:
                staging_list = list(self.partial_task_scores_buffer[actor_index].keys())
                for t_ in staging_list:
                    if self.partial_task_scores_buffer[actor_index][t_] > 0.0:
                        self.update_task_score(actor_index, t_, 0.0, float("-inf"), 0)

    def _update_staleness(self, selected_idx):
        if self.staleness_coef > 0:
            self.task_staleness = self.task_staleness + 1
            self.task_staleness[selected_idx] = 0

    def sample_replay_decision(self):
        proportion_seen = self._proportion_filled
        if self.sample_full_distribution:
            if self.task_buffer_size > 0:
                if self.replay_schedule == "fixed":
                    if proportion_seen >= self.rho and np.random.rand() < self.nu:
                        return True
                    return False
                else:
                    if proportion_seen >= self.rho and np.random.rand() < min(proportion_seen, self.nu):
                        return True
                    return False
            return False
        elif self.replay_schedule == "fixed":
            # Sample random level until we have seen enough tasks
            if proportion_seen >= self.rho:
                # Sample replay level with fixed replay_prob OR if all levels seen
                if np.random.rand() < self.nu or proportion_seen >= 1.0:
                    return True
            return False
        else:
            if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
                return True
            return False

    def observe_external_unseen_sample(self, tasks, solvable=None):
        for i, t_ in enumerate(tasks):
            self.running_sample_count += 1
            if not (t_ in self.staging_task_set or t_ in self.working_task_set):
                self.task2timestamp_buffer[t_] = self.running_sample_count
                self.staging_task_set.add(t_)
                if solvable is not None:
                    if not self.track_solvable:
                        self._init_solvable_tracking()
                    self.staging_task2solvable[t_] = solvable[i]
            else:
                task_idx = self.task2index.get(t_, None)
                if task_idx is not None:
                    self._update_staleness(task_idx)

    def _sample_replay_level(self, update_staleness=True):
        sample_weights = self.sample_weights()
        total = np.sum(sample_weights)
        if np.isclose(total, 0):
            sample_weights = np.ones_like(self.tasks, dtype=np.float32) / len(self.tasks)
            sample_weights *= (1 - self.unseen_task_weights)
            sample_weights /= np.sum(sample_weights)
        elif total != 1.0:
            sample_weights /= total
        task_idx = np.random.choice(range(len(self.tasks)), 1, p=sample_weights)[0]
        if update_staleness:
            self._update_staleness(task_idx)
        return int(self.tasks[task_idx])

    def _sample_unseen_level(self):
        if self.sample_full_distribution:
            t_val = int(np.random.randint(1, INT32_MAX))
            while t_val in self.staging_task_set or t_val in self.working_task_set:
                t_val = int(np.random.randint(1, INT32_MAX))
            self.task2timestamp_buffer[t_val] = self.running_sample_count
            self.staging_task_set.add(t_val)
            return t_val
        else:
            sample_weights = self.unseen_task_weights / self.unseen_task_weights.sum()
            task_idx = np.random.choice(range(len(self.tasks)), 1, p=sample_weights)[0]
            self._update_staleness(task_idx)
            return int(self.tasks[task_idx])

    def compute_gae_returns(self, 
                            returns_buffer,
                            next_value, 
                            gamma, 
                            gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        value_preds = self.value_preds

        if self.use_proper_time_limits:
            # Get truncated value preds
            self._compute_truncated_value_preds()
            value_preds = self.truncated_value_preds

        if self.use_popart:
            self.denorm_value_preds = self.model.popart.denormalize(value_preds) # denormalize all value predictions
            value_preds = self.denorm_value_preds

        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + \
                gamma*value_preds[step + 1]*self.masks[step + 1] - value_preds[step]

            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + value_preds[step]

    def compute_discounted_returns(self,
                                   rewards,
                                   value_preds,
                                   masks,
                                   returns,
                                   returns_buffer, 
                                   next_value,
                                   gamma):
        value_preds[-1] = next_value
        value_preds = value_preds

        # if self.use_proper_time_limits:    
        #     self._compute_truncated_value_preds()
        #     value_preds = self.truncated_value_preds

        # if self.use_popart:
        #     self.denorm_value_preds = self.model.popart.denormalize(value_preds) # denormalize all value predictions

        returns[-1] = value_preds[-1]

        for step in reversed(range(rewards.size(0))):
            returns_buffer[step] = returns_buffer[step + 1] * \
                gamma * masks[step + 1] + rewards[step]

    def compute_returns(self,
                        rewards,
                        value_preds,
                        masks,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda):
        returns = np.zeros_like(value_preds)
        if use_gae:
            return self.compute_gae_returns(
                rewards, value_preds, masks, returns, next_value, gamma, gae_lambda)
        else:
            return self.compute_discounted_returns(
                rewards, value_preds, masks, returns, next_value, gamma)

    def _evaluate_tasks(self, tasks):
        # TODO: Set task for evaluator envs
        tasks_encoded = [self.task_space.encode(task) for task in tasks]
        print(f"Evaluating tasks: {tasks_encoded}")
        obs, _ = self.eval_envs.reset(seed=tasks_encoded, options={"seed_task": True})
        done = False
        # TODO: Support any number of eval processes
        # TODO: Figure out how to generate roughly 1 episode of data for each task?
        rewards = []
        value_preds = []
        masks = []
        tasks = []
        while not done:
            action, value, lstm_states, _ = self.evaluator.get_action_and_value(obs)
            obs, rew, term, trunc, infos = self.eval_envs.step(action.cpu().numpy())

            mask = -torch.Tensor(np.logical_or(term, trunc)).unsqueeze(-1)
            rewards.append(torch.Tensor(
                rew).unsqueeze(-1))
            value_preds.append(value)
            masks.append(mask)
            tasks.append(torch.Tensor(tasks_encoded))
            print(infos)

            # Check if the episode is done
            if "episode" in infos:
                for i, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        print(info["episode"])
                        assert False

        next_value = self.evaluator.get_value(obs)
        returns = self.compute_returns(rewards, value_preds, masks, next_value, self.gamma, self.gae_lambda)
        
        rewards = torch.cat(rewards, dim=0)
        value_preds = torch.cat(value_preds, dim=0)
        masks = torch.cat(masks, dim=0)
        tasks = torch.cat(tasks, dim=0)

        # Iterate over eval actor indices
        for actor_index in range(self.num_actors, self.num_actors + self.eval_envs.num_envs):
            episode_data = {
                "tasks": tasks[:, actor_index],
                "masks": masks[:, actor_index],
                "rewards": rewards[:, actor_index],
                "value_preds": value_preds[:, actor_index],
                "returns": returns[:, actor_index],
            }
            self._update_with_rollouts(
                episode_data, actor_index=actor_index,
            )



    def sample(self, strategy=None):
        if strategy == "full_distribution":
            raise ValueError("One-off sampling via full_distribution strategy is not supported.")
        self.running_sample_count += 1

        if not strategy:
            strategy = self.strategy

        if not self.sample_full_distribution:
            if strategy == "random":
                task_idx = np.random.choice(range(self.num_tasks))
                return int(self.tasks[task_idx])
            if strategy == "sequential":
                task_idx = self.next_task_index
                self.next_task_index = (self.next_task_index + 1) % self.num_tasks
                return int(self.tasks[task_idx])

        replay_decision = self.sample_replay_decision()

        # If we have seen enough tasks to sample a replay level, stop training on them and only evaluate
        if self._proportion_filled >= self.rho:
            # Add random levels to an evaluation queue until we sample a replay level
            while not replay_decision:
                self.offline_queue.append(self._sample_unseen_level())
                replay_decision = self.sample_replay_decision()
            return self._sample_replay_level()
        else:
            return self._sample_unseen_level()

    def sample_weights(self):
        weights = self._score_transform(self.score_transform, self.temperature, self.task_scores)
        weights *= (1 - self.unseen_task_weights)
        z = np.sum(weights)
        if z > 0:
            weights /= z
        else:
            weights = np.ones_like(weights, dtype=np.float32) / len(weights)
            weights *= (1 - self.unseen_task_weights)
            weights /= np.sum(weights)

        if self.staleness_coef > 0:
            staleness_w = self._score_transform(
                self.staleness_transform, self.staleness_temperature, self.task_staleness)
            staleness_w *= (1 - self.unseen_task_weights)
            z_s = np.sum(staleness_w)
            if z_s > 0:
                staleness_w /= z_s
            else:
                staleness_w = (1.0 / len(staleness_w)) * (1 - self.unseen_task_weights)
            weights = (1.0 - self.staleness_coef) * weights + self.staleness_coef * staleness_w
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        return weights

    def _score_transform(self, transform, temperature, scores):
        if transform == "constant":
            weights = np.ones_like(scores)
        elif transform == "max":
            weights = np.zeros_like(scores)
            scores_ = scores[:]
            scores_[self.unseen_task_weights > 0] = -float("inf")
            argmax = np.random.choice(np.flatnonzero(np.isclose(scores_, scores_.max())))
            weights[argmax] = 1.0
        elif transform == "eps_greedy":
            weights = np.zeros_like(scores)
            weights[scores.argmax()] = 1.0 - self.eps
            weights += self.eps / len(self.tasks)
        elif transform == "rank":
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1 / (ranks ** (1.0 / temperature))
        elif transform == "power":
            eps_ = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores).clip(0) + eps_) ** (1.0 / temperature)
        elif transform == "softmax":
            weights = np.exp(np.array(scores) / temperature)
        elif transform == "match":
            w_ = np.array([(1.0 - s) * s for s in scores])
            weights = w_ ** (1.0 / temperature)
        elif transform == "match_rank":
            w_ = np.array([(1.0 - s) * s for s in scores])
            temp = np.flip(w_.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1 / (ranks ** (1.0 / temperature))
        else:
            weights = np.ones_like(scores)
        return weights

    def metrics(self):
        """ Return sampling metrics for logging. """
        n = self.task_buffer_size if self.sample_full_distribution else self.num_tasks
        proportion_seen = (n - (self.unseen_task_weights > 0).sum()) / float(n) if n > 0 else 0.0
        return {
            "task_scores": self.task_scores,
            "unseen_task_weights": self.unseen_task_weights,
            "task_staleness": self.task_staleness,
            "proportion_seen": proportion_seen,
            "score": self._last_score,
        }

    @property
    def solvable_mass(self):
        if self.track_solvable:
            sw = self.sample_weights()
            return np.sum(sw[self.task_solvable])
        return 1.0

    @property
    def max_score(self):
        return max(self.task_scores)
