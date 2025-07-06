""" Task wrapper for NLE that can change tasks at reset using the NLE's task definition format. """
import re
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from nle import nethack
from nle.env import base
from nle.env.base import ASCII_SPACE, ASCII_ESC, SKIP_EXCEPTIONS
from nle.env.tasks import TASK_ACTIONS, NetHackChallenge, NetHackGold
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

from syllabus.core import TaskWrapper
from syllabus.task_space import DiscreteTaskSpace


EXTENDED_TASK_ACTIONS = (
    *TASK_ACTIONS,
    nethack.Command.CAST,
    nethack.Command.QUAFF,
    nethack.Command.PRAY,
    nethack.Command.PICKUP,
    nethack.Command.PAY,
)

SKIP_EXCEPTIONS = (*SKIP_EXCEPTIONS, b"drink")


class NetHackScore(base.NLE):
    """Environment for "score" task.

    The task is an augmentation of the standard NLE task. The return function is
    defined as:
    :math:`\text{score}_t - \text{score}_{t-1} + \text{TP}`,
    where the :math:`\text{TP}` is a time penalty that grows with the amount of
    environment steps that do not change the state (such as navigating menus).

    Args:
        penalty_mode (str): name of the mode for calculating the time step
            penalty. Can be ``constant``, ``exp``, ``square``, ``linear``, or
            ``always``. Defaults to ``constant``.
        penalty_step (float): constant applied to amount of frozen steps.
            Defaults to -0.01.
        penalty_time (float): constant applied to amount of frozen steps.
            Defaults to -0.0.

    """

    def __init__(
        self,
        *args,
        penalty_mode="constant",
        penalty_step: float = -0.01,
        penalty_time: float = -0.0,
        **kwargs,
    ):
        self.penalty_mode = penalty_mode
        self.penalty_step = penalty_step
        self.penalty_time = penalty_time

        self._frozen_steps = 0

        self.dungeon_explored = {}

        actions = kwargs.pop("actions", TASK_ACTIONS)
        super().__init__(*args, actions=actions, **kwargs)

    def _get_time_penalty(self, last_observation, observation):
        blstats_old = last_observation[self._blstats_index]
        blstats_new = observation[self._blstats_index]

        old_time = blstats_old[nethack.NLE_BL_TIME]
        new_time = blstats_new[nethack.NLE_BL_TIME]

        if old_time == new_time:
            self._frozen_steps += 1
        else:
            self._frozen_steps = 0

        penalty = 0
        if self.penalty_mode == "constant":
            if self._frozen_steps > 0:
                penalty += self.penalty_step
        elif self.penalty_mode == "exp":
            penalty += 2**self._frozen_steps * self.penalty_step
        elif self.penalty_mode == "square":
            penalty += self._frozen_steps**2 * self.penalty_step
        elif self.penalty_mode == "linear":
            penalty += self._frozen_steps * self.penalty_step
        elif self.penalty_mode == "always":
            penalty += self.penalty_step
        else:  # default
            raise ValueError("Unknown penalty_mode '%s'" % self.penalty_mode)
        penalty += (new_time - old_time) * self.penalty_time
        return penalty

    def _reward_fn(self, last_observation, action, observation, end_status):
        """Score delta, but with added a state loop penalty."""
        score_diff = super()._reward_fn(
            last_observation, action, observation, end_status
        )
        time_penalty = self._get_time_penalty(last_observation, observation)
        return score_diff + time_penalty

    def _perform_known_steps(self, observation, done, exceptions=True):
        while not done:

            if observation[self._internal_index][3]:  # xwaitforspace
                # Make sure to include information about going down the stairs.
                previous_msg = observation[self._message_index].copy()
                msg_str = bytes(previous_msg)
                observation, done = self.nethack.step(ASCII_SPACE)
                if b"You descend the stairs." in msg_str:
                    observation = (
                        *observation[: self._message_index],
                        previous_msg,
                        *observation[self._message_index + 1 :],
                    )
                continue

            internal = observation[self._internal_index]
            in_yn_function = internal[1]
            in_getline = internal[2]

            if in_getline:  # Game asking for a line of text. We don't do that.
                observation, done = self.nethack.step(ASCII_ESC)
                continue

            if in_yn_function:  # Game asking for a single character.
                # Note: No auto-yes to final questions thanks to the disclose option.
                if exceptions:
                    # This causes an annoying unnecessary copy...
                    msg = bytes(observation[self._message_index])

                    # Do not skip some questions to allow agent to select
                    # stuff to eat, attack, and to select directions.
                    # Also do not skip if all allowed or the allowed message appears.
                    if self._allow_all_yn_questions or any(
                        el in msg for el in SKIP_EXCEPTIONS
                    ):
                        break

                # Otherwise, auto-decline.
                observation, done = self.nethack.step(ASCII_ESC)
            break

        return observation, done

    def get_scout_score(self, last_observation):
        glyphs = last_observation[self._glyph_index]
        blstats = last_observation[self._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != nethack.GLYPH_CMAP_OFF)
        self.dungeon_explored[key] = explored
        total_explored = 0
        for key, value in self.dungeon_explored.items():
            total_explored += value
        return total_explored

    def step(self, action: int):
        """Steps the environment.

        Args:
            action (int): action integer as defined by ``self.action_space``.

        Returns:
            (dict, float, bool, dict): a tuple containing
                - (*dict*): an observation of the state; this will contain the keys
                  specified by ``self.observation_space``.
                - (*float*): a reward; see ``self._reward_fn`` to see how it is
                  specified.
                - (*bool*): True if the state is terminal, False otherwise.
                - (*dict*): a dictionary of extra information (such as
                  `end_status`, i.e. a status info -- death, task win, etc. --
                  for the terminal state).
        """
        # Careful: By default we re-use Numpy arrays, so copy before!
        last_observation = tuple(a.copy() for a in self.last_observation)

        # Fix the eating action such that it is possible to eat all items
        last_msg = bytes(last_observation[self._message_index]).decode("utf-8")
        if "What do you want to eat" in last_msg:
            pattern = r"\[([a-zA-Z]+)"
            match = re.search(pattern, last_msg)
            if match and self.actions[action] == ord("y"):
                # Action 'y' for 'yes' will lead to eating any random item in the inventory
                action = ord(match.group(1)[0])
            else:
                # Otherwise escape
                action = ASCII_SPACE
        else:
            action = self.actions[action]

        observation, done = self.nethack.step(action)
        is_game_over = observation[self._program_state_index][0] == 1
        if is_game_over or not self._allow_all_modes:
            observation, done = self._perform_known_steps(
                observation, done, exceptions=True
            )

        self._steps += 1

        self.last_observation = observation

        if self._check_abort(observation):
            end_status = self.StepStatus.ABORTED
        else:
            end_status = self._is_episode_end(observation)
        end_status = self.StepStatus(done or end_status)

        reward = float(
            self._reward_fn(last_observation, action, observation, end_status)
        )

        if end_status and not done:
            # Try to end the game nicely.
            self._quit_game(observation, done)
            done = True

        info = {}
        info["end_status"] = end_status
        info["is_ascended"] = self.nethack.how_done() == nethack.ASCENDED
        info["dlvl"] = last_observation[self._blstats_index][12]
        info["gold"] = last_observation[self._blstats_index][13]
        info["xlvl"] = last_observation[self._blstats_index][18]
        info["scout"] = self.get_scout_score(last_observation)

        return self._get_observation(observation), reward, done, info


class NetHackExtendedActionEnv(base.NLE):

    def __init__(self, *args, no_progress_timeout: int = 10_000, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_progress_timeout = no_progress_timeout

    def _perform_known_steps(self, observation, done, action=None, exceptions=True):
        while not done:
            message_str = bytes(observation[self._message_index]).decode("utf-8")

            # Macro-action for casting the spell
            if action == nethack.Command.CAST:
                observation, done = self.nethack.step(ord("a"))

            # Enhance skills whenever possible
            elif "You feel more confident in" in message_str:
                self.enhance = True

            if observation[self._internal_index][3]:  # xwaitforspace
                if "You feel more confident in" in message_str:
                    self.enhance = True

                # Make sure to include information about going down the stairs.
                previous_msg = observation[self._message_index].copy()
                msg_str = bytes(previous_msg)
                observation, done = self.nethack.step(ASCII_SPACE)
                action = ASCII_SPACE
                if b"You descend the stairs." in msg_str:
                    observation = (
                        *observation[: self._message_index],
                        previous_msg,
                        *observation[self._message_index + 1 :],
                    )
                continue

            internal = observation[self._internal_index]
            in_yn_function = internal[1]
            in_getline = internal[2]

            if in_getline:  # Game asking for a line of text. We don't do that.
                observation, done = self.nethack.step(ASCII_ESC)
                action = ASCII_ESC
                continue

            if in_yn_function:  # Game asking for a single character.
                # Note: No auto-yes to final questions thanks to the disclose option.
                if exceptions:
                    # This causes an annoying unnecessary copy...
                    msg = bytes(observation[self._message_index])

                    # Do not skip some questions to allow agent to select
                    # stuff to eat, attack, and to select directions.
                    # Also do not skip if all allowed or the allowed message appears.
                    if self._allow_all_yn_questions or any(
                        el in msg for el in SKIP_EXCEPTIONS
                    ):
                        break

                # Otherwise, auto-decline.
                observation, done = self.nethack.step(ASCII_ESC)
                action = ASCII_ESC

            if self.enhance:
                observation, done = self.nethack.step(nethack.Command.ENHANCE)
                observation, done = self.nethack.step(ord("a"))
                self.enhance = False
            break

        return observation, done

    def get_scout_score(self, last_observation):
        glyphs = last_observation[self._glyph_index]
        blstats = last_observation[self._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != nethack.GLYPH_CMAP_OFF)
        self.dungeon_explored[key] = explored
        total_explored = 0
        for key, value in self.dungeon_explored.items():
            total_explored += value
        return total_explored

    def reset(self, *args, **kwargs):
        self.dungeon_explored = {}
        self.enhance = False
        self.num_kills = 0
        return super().reset(*args, **kwargs)

    def step(self, action: int):
        """Steps the environment.

        Args:
            action (int): action integer as defined by ``self.action_space``.

        Returns:
            (dict, float, bool, dict): a tuple containing
                - (*dict*): an observation of the state; this will contain the keys
                  specified by ``self.observation_space``.
                - (*float*): a reward; see ``self._reward_fn`` to see how it is
                  specified.
                - (*bool*): True if the state is terminal, False otherwise.
                - (*dict*): a dictionary of extra information (such as
                  `end_status`, i.e. a status info -- death, task win, etc. --
                  for the terminal state).
        """
        # Careful: By default we re-use Numpy arrays, so copy before!
        last_observation = tuple(a.copy() for a in self.last_observation)

        # Fix the eating action such that it is possible to eat all items
        last_msg = bytes(last_observation[self._message_index]).decode("utf-8")

        if "you kill " in last_msg.lower():
            self.num_kills += 1

        if "What do you want to eat" in last_msg:
            pattern = r"\[([a-zA-Z]+)"
            match = re.search(pattern, last_msg)
            if match and self.actions[action] == ord("y"):
                # Action 'y' for 'yes' will lead to eating any random item in the inventory
                action = ord(match.group(1)[0])
            else:
                # Otherwise escape
                action = ASCII_SPACE
        elif "What do you want to drink" in last_msg:
            pattern = r"\[([a-zA-Z]+)"
            match = re.search(pattern, last_msg)
            if match and self.actions[action] == ord("y"):
                action = ord(match.group(1)[0])
            else:
                action = ASCII_SPACE
        else:
            action = self.actions[action]

        observation, done = self.nethack.step(action)
        is_game_over = observation[self._program_state_index][0] == 1
        # perform known steps
        if is_game_over or not self._allow_all_modes:
            observation, done = self._perform_known_steps(
                observation, done, action=action, exceptions=True
            )

        self._steps += 1

        self.last_observation = observation

        if self._check_abort(observation):
            end_status = self.StepStatus.ABORTED
        else:
            end_status = self._is_episode_end(observation)
        end_status = self.StepStatus(done or end_status)

        reward = float(
            self._reward_fn(last_observation, action, observation, end_status)
        )

        if end_status and not done:
            # Try to end the game nicely.
            self._quit_game(observation, done)
            done = True

        info = {}
        info["end_status"] = end_status
        info["is_ascended"] = self.nethack.how_done() == nethack.ASCENDED
        info["dlvl"] = last_observation[self._blstats_index][12]
        info["gold"] = last_observation[self._blstats_index][13]
        info["xlvl"] = last_observation[self._blstats_index][18]
        info["scout"] = self.get_scout_score(last_observation)
        info["kill_counts"] = self.num_kills

        observation = self._get_observation(observation)
        return observation, reward, done, info

    def _check_abort(self, observation):
        """Check if time has stopped and no observations has changed long enough
        to trigger an abort."""

        turns = observation[self._blstats_index][nethack.NLE_BL_TIME]
        if self._turns == turns:
            self._no_progress_count += 1
        else:
            self._turns = turns
            self._no_progress_count = 0
        return (
            self._steps >= self._max_episode_steps
            or self._no_progress_count >= self.no_progress_timeout
        )
    

class NetHackScoreExtendedActions(NetHackExtendedActionEnv, NetHackScore):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.actions = EXTENDED_TASK_ACTIONS
        self.action_space = gym.spaces.Discrete(len(self.actions))


class NetHackSeed(NetHackScore):
    """Environment for the NetHack Challenge.

    The task is an augmentation of the standard NLE task. This is the NLE Score Task
    but with some subtle differences:
    * the action space is fixed to include the full keyboard
    * menus and "<More>" tokens are not skipped
    * starting character is randomly assigned
    """

    def __init__(
        self,
        *args,
        character="@",
        allow_all_yn_questions=True,
        allow_all_modes=True,
        penalty_mode="constant",
        penalty_step: float = -0.00,
        penalty_time: float = -0.0,
        max_episode_steps: int = 1e6,
        observation_keys=(
            "glyphs",
            "chars",
            "colors",
            "specials",
            "blstats",
            "message",
            "inv_glyphs",
            "inv_strs",
            "inv_letters",
            "inv_oclasses",
            "tty_chars",
            "tty_colors",
            "tty_cursor",
            "misc",
        ),
        no_progress_timeout: int = 10_000,
        **kwargs,
    ):
        actions = nethack.ACTIONS
        kwargs["wizard"] = False
        super().__init__(
            *args,
            actions=actions,
            character=character,
            allow_all_yn_questions=allow_all_yn_questions,
            allow_all_modes=allow_all_modes,
            penalty_mode=penalty_mode,
            penalty_step=penalty_step,
            penalty_time=penalty_time,
            max_episode_steps=max_episode_steps,
            observation_keys=observation_keys,
            **kwargs,
        )
        # If the in-game turn count doesn't change for 10_000 steps, we abort
        self.no_progress_timeout = no_progress_timeout

    def reset(self, *args, **kwargs):
        self._turns = None
        self._no_progress_count = 0
        return super().reset(*args, **kwargs)

    def _check_abort(self, observation):
        """Check if time has stopped and no observations has changed long enough
        to trigger an abort."""

        turns = observation[self._blstats_index][nethack.NLE_BL_TIME]
        if self._turns == turns:
            self._no_progress_count += 1
        else:
            self._turns = turns
            self._no_progress_count = 0
        return (
            self._steps >= self._max_episode_steps
            or self._no_progress_count >= self.no_progress_timeout
        )


class NetHackDescend(NetHackScore):
    """Environment for "staircase" task.

    This task requires the agent to get on top of a staircase down (>).
    The reward function is :math:`I + \text{TP}`, where :math:`I` is 1 if the
    task is successful, and 0 otherwise, and :math:`\text{TP}` is the time step
    function as defined by `NetHackScore`.
    """

    def reset(self, wizkit_items=None):
        self.max_dungeon_level = 1
        return super().reset(wizkit_items=wizkit_items)

    def _is_episode_end(self, observation):
        return self.StepStatus.RUNNING

    def _reward_fn(self, last_observation, action, observation, end_status):
        del action, end_status
        time_penalty = self._get_time_penalty(last_observation, observation)
        dungeon_level = observation[self._blstats_index][12]
        if dungeon_level > self.max_dungeon_level:
            reward = 100
            self.max_dungeon_level = dungeon_level
        else:
            reward = 0
        return reward + time_penalty


class NetHackCollect(NetHackGold):
    """Environment for "staircase" task.

    This task requires the agent to get on top of a staircase down (>).
    The reward function is :math:`I + \text{TP}`, where :math:`I` is 1 if the
    task is successful, and 0 otherwise, and :math:`\text{TP}` is the time step
    function as defined by `NetHackScore`.
    """

    def __init__(self, *args, **kwargs):
        actions = kwargs.pop("actions", TASK_ACTIONS + (nethack.Command.PICKUP, nethack.Command.DROP))
        super().__init__(*args, actions=actions, **kwargs)

    def reset(self, wizkit_items=None):
        observation = super().reset(wizkit_items=wizkit_items)
        inventory = observation["inv_glyphs"]
        self.collected_items = set(inventory)
        self._inv_glphys_index = self._observation_keys.index("inv_glyphs")
        return observation

    def _is_episode_end(self, observation):
        return self.StepStatus.RUNNING

    def _reward_fn(self, last_observation, action, observation, end_status):
        gold_reward = min(10, super()._reward_fn(last_observation, action, observation, end_status))
        inventory = observation[self._inv_glphys_index]
        item_reward = 0
        for item in inventory:
            if item not in self.collected_items:
                self.collected_items.add(item)
                item_reward += 10
        return item_reward + gold_reward


class NetHackSatiate(NetHackScore):
    """Environment for the "eat" task.

    The task is similar to the one defined by `NetHackScore`, but the reward
    uses positive changes in the character's hunger level (e.g. by consuming
    comestibles or monster corpses), rather than the score.
    """

    def _reward_fn(self, last_observation, action, observation, end_status):
        """Difference between previous hunger and new hunger."""
        del end_status  # Unused
        del action  # Unused

        if not self.nethack.in_normal_game():
            # Before game started and after it ended blstats are zero.
            return 0.0

        old_internal = last_observation[self._internal_index]
        internal = observation[self._internal_index]
        old_blstats = last_observation[self._blstats_index]

        old_uhunger = old_internal[7]
        uhunger = internal[7]
        is_satiated = old_blstats[21] == 0

        if is_satiated:
            # If the agent is satiated, we don't want to reward it for eating
            reward = 0
        else:
            # Give a reward for eating, but cap it at 10
            reward = min(10, uhunger - old_uhunger)

        time_penalty = self._get_time_penalty(last_observation, observation)

        return reward + time_penalty


class NetHackScoutClipped(NetHackScore):
    """Environment for the "scout" task.

    The task is similar to the one defined by `NetHackScore`, but the score is
    defined by the changes in glyphs discovered by the agent.
    """

    def reset(self, *args, **kwargs):
        self.dungeon_explored = {}
        return super().reset(*args, **kwargs)

    def _reward_fn(self, last_observation, action, observation, end_status):
        del end_status  # Unused
        del action  # Unused

        if not self.nethack.in_normal_game():
            # Before game started and after it ended blstats are zero.
            return 0.0

        reward = 0
        glyphs = observation[self._glyph_index]
        blstats = observation[self._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != nethack.GLYPH_CMAP_OFF)
        explored_old = 0
        if key in self.dungeon_explored:
            explored_old = self.dungeon_explored[key]
        reward = min(5, explored - explored_old)
        self.dungeon_explored[key] = explored
        time_penalty = self._get_time_penalty(last_observation, observation)
        return reward + time_penalty


class NethackTaskWrapper(TaskWrapper):
    """
    This wrapper allows you to change the task of an NLE environment.

    This wrapper was designed to meet two goals.
        1. Allow us to change the task of the NLE environment at the start of an episode
        2. Allow us to use the predefined NLE task definitions without copying/modifying their code.
           This makes it easier to integrate with other work on nethack tasks or curricula.

    Each task is defined as a subclass of the NLE, so you need to cast and reinitialize the
    environment to change its task. This wrapper manipulates the __class__ property to achieve this,
    but does so in a safe way. Specifically, we ensure that the instance variables needed for each
    task are available and reset at the start of the episode regardless of which task is active.
    """

    def __init__(
        self,
        env: gym.Env,
        additional_tasks: List[base.NLE] = None,
        use_default_tasks: bool = True,
        env_kwargs: Dict[str, Any] = {},
        wrappers: List[Tuple[gym.Wrapper, List[Any], Dict[str, Any]]] = None,
        seed: int = None,
    ):
        super().__init__(env)
        self.env = env
        self.task = NetHackScore
        self._init_kwargs = env_kwargs
        if self.env.__class__ == NetHackChallenge:
            self._no_progress_timeout = self._init_kwargs.pop("no_progress_timeout", 150)

        # This is set to False during reset
        self.done = True

        # Add nethack tasks provided by the base NLE
        task_list: List[base.NLE] = []
        if use_default_tasks:
            task_list = [
                NetHackScore,
                NetHackDescend,
                NetHackCollect,
                NetHackSatiate,
                NetHackScoutClipped,
            ]

        # Add in custom nethack tasks
        if additional_tasks:
            for task in additional_tasks:
                assert isinstance(task, base.NLE), "Env must subclass the base NLE"
                task_list.append(task)

        self.task_list = task_list
        self.task_space = DiscreteTaskSpace(len(task_list), task_list)

        # Add goal space to observation
        # self.observation_space = copy.deepcopy(self.env.observation_space)
        # self.observation_space["goal"] = spaces.MultiBinary(len(self.task_list))

        # Task completion metrics
        self.episode_return = 0

        # TODO: Deal with wrappers
        self._nethack_env = self.env
        while self._nethack_env.__class__ not in self.task_list and self._nethack_env.__class__ != NetHackChallenge:
            if self._nethack_env.__class__ == GymV21CompatibilityV0:
                self._nethack_env = self._nethack_env.gym_env
            else:
                self._nethack_env = self._nethack_env.env

        # Initialize missing instance variables
        self._nethack_env.oracle_glyph = None
        if seed is not None:
            self.seed(seed)

    def seed(self, seed):
        self.env.env.seed(core=seed, disp=seed)

    def _task_name(self, task):
        return task.__name__

    def reset(self, new_task=None, **kwargs):
        """
        Resets the environment along with all available tasks, and change the current task.

        This ensures that all instance variables are reset, not just the ones for the current task.
        We do this efficiently by keeping track of which reset functions have already been called,
        since very few tasks override reset. If new_task is provided, we change the task before
        calling the final reset.
        """
        # Change task if new one is provided
        if new_task is None:
            new_task = kwargs.get("options", None)
        kwargs.pop("options", None)

        if new_task is not None:
            self.change_task(new_task)

        self.done = False
        self.episode_return = 0

        obs, info = self.env.reset(**kwargs)
        obs["prev_action"] = 0
        obs["tty_cursor"] = self.task_space.encode(self.task)

        return self.observation(obs), info

    def change_task(self, new_task: int):
        """
        Change task by directly editing environment class.

        Ignores requests for unknown tasks or task changes outside of a reset.
        """
        # Ignore new task if mid episode
        if self.task.__init__ != new_task.__init__ and not self.done:
            print(f"Given task {self._task_name(new_task)} needs to be reinitialized.\
                  Ignoring request to change task and keeping {self.task.__name__}")
            return

        # Ignore if task is unknown
        if new_task not in self.task_list:
            print(f"Given task {new_task} not in task list.\
                  Ignoring request to change task and keeping {self.env.__class__.__name__}")
            return

        # Update current task
        self.task = new_task
        self._nethack_env.__class__ = new_task

        # If task requires reinitialization
        # if type(self._nethack_env).__init__ != NetHackScore.__init__:
        #     self._nethack_env.__init__(actions=nethack.ACTIONS, **self._init_kwargs)

    def _encode_goal(self):
        goal_encoding = np.zeros(len(self.task_list))
        index = self.task_list.index(self.task)
        goal_encoding[index] = 1
        return goal_encoding

    def observation(self, observation):
        """
        Parses current inventory and new items gained this timestep from the observation.
        Returns a modified observation.
        """
        # Add goal to observation
        # observation['goal'] = self._encode_goal()
        return observation

    def _task_completion(self, obs, rew, term, trunc, info):
        # TODO: Add real task completion metrics
        completion = 0.0
        if self.task == 0:
            completion = self.episode_return / 1000
        elif self.task == 1:
            completion = self.episode_return
        elif self.task == 2:
            completion = self.episode_return
        elif self.task == 3:
            completion = self.episode_return
        elif self.task == 4:
            completion = self.episode_return / 1000
        elif self.task == 5:
            completion = self.episode_return / 10
        elif self.task == 6:
            completion = self.episode_return / 100

        return min(max(completion, 0.0), 1.0)

    def step(self, action):
        """
        Step through environment and update task completion.
        """
        obs, rew, term, trunc, info = self.env.step(action)
        # self.episode_return += rew
        self.done = term or trunc
        info["task_completion"] = self._task_completion(obs, rew, term, trunc, info)
        obs["prev_action"] = action
        obs["tty_cursor"] = self.task_space.encode(self.task)
        return self.observation(obs), rew, term, trunc, info


class NethackSeedWrapper(TaskWrapper):
    """
    This wrapper allows you to change the task of an NLE environment.

    This wrapper was designed to meet two goals.
        1. Allow us to change the task of the NLE environment at the start of an episode
        2. Allow us to use the predefined NLE task definitions without copying/modifying their code.
           This makes it easier to integrate with other work on nethack tasks or curricula.

    Each task is defined as a subclass of the NLE, so you need to cast and reinitialize the
    environment to change its task. This wrapper manipulates the __class__ property to achieve this,
    but does so in a safe way. Specifically, we ensure that the instance variables needed for each
    task are available and reset at the start of the episode regardless of which task is active.
    """

    def __init__(
        self,
        env: gym.Env,
        seed: int = 0,
        num_seeds: int = 200,
    ):
        super().__init__(env)
        self.env = env
        self.task_space = DiscreteTaskSpace(num_seeds)

        # Task completion metrics
        self.episode_return = 0
        self.task = seed

        if seed is not None:
            self.seed(seed)

    def seed(self, seed):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        env.seed(core=seed, disp=seed)

    def _task_name(self, task):
        return task.__name__

    def reset(self, new_task=None, **kwargs):
        """
        Resets the environment along with all available tasks, and change the current task.

        This ensures that all instance variables are reset, not just the ones for the current task.
        We do this efficiently by keeping track of which reset functions have already been called,
        since very few tasks override reset. If new_task is provided, we change the task before
        calling the final reset.
        """
        obs, info = super().reset(new_task=new_task, **kwargs)
        if isinstance(obs, dict):
            obs["prev_action"] = 0
        return self.observation(obs), info

    def change_task(self, new_task: int):
        """
        Change task by setting the seed.
        """
        # Ignore new task if mid episode
        self.task = new_task
        self.seed(new_task)

    def observation(self, observation):
        """
        Parses current inventory and new items gained this timestep from the observation.
        Returns a modified observation.
        """
        if isinstance(observation, dict):
            encoded_task = self.task_space.encode(self.task)
            observation["tty_cursor"] = encoded_task if encoded_task is not None else -1

        return observation

    def step(self, action):
        """
        Step through environment and update task completion.
        """
        obs, rew, term, trunc, info = super().step(action)
        if isinstance(obs, dict):
            obs["prev_action"] = action
        return obs, rew, term, trunc, info
