""" Task wrapper for NLE that can change tasks at reset using the NLE's task definition format. """
import os
import time
from typing import Any, Dict, List, Tuple

import cv2
import gymnasium as gym
import numpy as np
import render_utils
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from nle import nethack
from nle.env import base
from nle.env.tasks import (NetHackChallenge, NetHackEat, NetHackGold,
                           NetHackOracle, NetHackScore, NetHackScout,
                           NetHackStaircase, NetHackStaircasePet)
from numba import njit
from PIL import Image, ImageDraw, ImageFont
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

from syllabus.core import TaskWrapper
from syllabus.task_space import TaskSpace


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
        wrappers: List[Tuple[gym.Wrapper, List[Any], Dict[str, Any]]] = None
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
                NetHackStaircase,
                NetHackStaircasePet,
                NetHackOracle,
                NetHackGold,
                NetHackEat,
                NetHackScout,
            ]

        # Add in custom nethack tasks
        if additional_tasks:
            for task in additional_tasks:
                assert isinstance(task, base.NLE), "Env must subclass the base NLE"
                task_list.append(task)

        self.task_list = task_list
        gym_space = gym.spaces.Discrete(len(self.task_list))
        self.task_space = TaskSpace(gym_space, task_list)

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
        new_task = np.random.choice(self.task_list)
        if new_task is not None:
            self.change_task(new_task)

        self.done = False
        self.episode_return = 0

        return self.observation(self.env.reset(**kwargs))

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
        obs, rew, term, trunc, info = step_api_compatibility(self.env.step(action), output_truncation_bool=True)
        # self.episode_return += rew
        self.done = term or trunc
        info["task_completion"] = self._task_completion(obs, rew, term, trunc, info)
        return self.observation(obs), rew, term, trunc, info


SMALL_FONT_PATH = os.path.abspath("syllabus/examples/utils/Hack-Regular.ttf")

# Mapping of 0-15 colors used.
# Taken from bottom image here. It seems about right
# https://i.stack.imgur.com/UQVe5.png
COLORS = [
    "#000000",
    "#800000",
    "#008000",
    "#808000",
    "#000080",
    "#800080",
    "#008080",
    "#808080",  # - flipped these ones around
    "#C0C0C0",  # | the gray-out dull stuff
    "#FF0000",
    "#00FF00",
    "#FFFF00",
    "#0000FF",
    "#FF00FF",
    "#00FFFF",
    "#FFFFFF",
]


@njit
def _tile_characters_to_image(
    out_image,
    chars,
    colors,
    output_height_chars,
    output_width_chars,
    char_array,
    offset_h,
    offset_w,
):
    """
    Build an image using cached images of characters in char_array to out_image
    """
    char_height = char_array.shape[3]
    char_width = char_array.shape[4]
    for h in range(output_height_chars):
        h_char = h + offset_h
        # Stuff outside boundaries is not visible, so
        # just leave it black
        if h_char < 0 or h_char >= chars.shape[0]:
            continue
        for w in range(output_width_chars):
            w_char = w + offset_w
            if w_char < 0 or w_char >= chars.shape[1]:
                continue
            char = chars[h_char, w_char]
            color = colors[h_char, w_char]
            h_pixel = h * char_height
            w_pixel = w * char_width
            out_image[
                :, h_pixel : h_pixel + char_height, w_pixel : w_pixel + char_width
            ] = char_array[char, color]


def _initialize_char_array(font_size, rescale_font_size):
    """Draw all characters in PIL and cache them in numpy arrays

    if rescale_font_size is given, assume it is (width, height)

    Returns a np array of (num_chars, num_colors, char_height, char_width, 3)
    """
    try:
        font = ImageFont.truetype(SMALL_FONT_PATH, font_size)
    except OSError as e:
        raise ValueError("Change SMALL_FONT_PATH to point to syllabus/examples/utils/Hack-Regular.ttf") from e

    dummy_text = "".join(
        [(chr(i) if chr(i).isprintable() else " ") for i in range(256)]
    )
    _, _, image_width, image_height = font.getbbox(dummy_text)
    # Above can not be trusted (or its siblings)....
    image_width = int(np.ceil(image_width / 256) * 256)

    char_width = rescale_font_size[0]
    char_height = rescale_font_size[1]

    char_array = np.zeros((256, 16, char_height, char_width, 3), dtype=np.uint8)
    image = Image.new("RGB", (image_width, image_height))
    image_draw = ImageDraw.Draw(image)
    for color_index in range(16):
        image_draw.rectangle((0, 0, image_width, image_height), fill=(0, 0, 0))
        image_draw.text((0, 0), dummy_text, fill=COLORS[color_index], spacing=0)

        arr = np.array(image).copy()
        arrs = np.array_split(arr, 256, axis=1)
        for char_index in range(256):
            char = arrs[char_index]
            if rescale_font_size:
                char = cv2.resize(char, rescale_font_size, interpolation=cv2.INTER_AREA)
            char_array[char_index, color_index] = char
    return char_array


class RenderCharImagesWithNumpyWrapper(gym.Wrapper):
    """
    Render characters as images, using PIL to render characters like we humans see on screen
    but then some caching and numpy stuff to speed up things.

    To speed things up, crop image around the player.
    """

    def __init__(
        self,
        env,
        font_size=9,
        crop_size=12,
        rescale_font_size=(6, 6),
        blstats_cursor=False,
    ):
        super().__init__(env)
        self.char_array = _initialize_char_array(font_size, rescale_font_size)
        self.char_height = self.char_array.shape[2]
        self.char_width = self.char_array.shape[3]
        # Transpose for CHW
        self.char_array = self.char_array.transpose(0, 1, 4, 2, 3)

        self.crop_size = crop_size
        self.blstats_cursor = blstats_cursor

        self.half_crop_size = crop_size // 2
        self.output_height_chars = crop_size
        self.output_width_chars = crop_size
        self.chw_image_shape = (
            3,
            self.output_height_chars * self.char_height,
            self.output_width_chars * self.char_width,
        )

        obs_spaces = {
            "screen_image": gym.spaces.Box(
                low=0, high=255, shape=self.chw_image_shape, dtype=np.uint8
            )
        }
        obs_spaces.update(
            [
                (k, self.env.observation_space[k])
                for k in self.env.observation_space
                if k not in ["tty_chars", "tty_colors"]
            ]
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _render_text_to_image(self, obs):
        chars = obs["tty_chars"]
        colors = obs["tty_colors"]
        offset_w = 0
        offset_h = 0
        if self.crop_size:
            # Center around player
            if self.blstats_cursor:
                center_x, center_y = obs["blstats"][:2]
            else:
                center_y, center_x = obs["tty_cursor"]
            offset_h = center_y - self.half_crop_size
            offset_w = center_x - self.half_crop_size

        out_image = np.zeros(self.chw_image_shape, dtype=np.uint8)

        _tile_characters_to_image(
            out_image=out_image,
            chars=chars,
            colors=colors,
            output_height_chars=self.output_height_chars,
            output_width_chars=self.output_width_chars,
            char_array=self.char_array,
            offset_h=offset_h,
            offset_w=offset_w,
        )

        obs["screen_image"] = out_image
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._render_text_to_image(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self._render_text_to_image(obs)
        return obs


class RenderCharImagesWithNumpyWrapperV2(gym.Wrapper):
    """
    Same as V1, but simpler and faster.
    """

    def __init__(
        self,
        env,
        font_size=9,
        crop_size=12,
        rescale_font_size=(6, 6),
    ):
        super().__init__(env)
        self.char_array = _initialize_char_array(font_size, rescale_font_size)
        self.char_height = self.char_array.shape[2]
        self.char_width = self.char_array.shape[3]
        # Transpose for CHW
        self.char_array = self.char_array.transpose(0, 1, 4, 2, 3)
        self.char_array = np.ascontiguousarray(self.char_array)
        self.crop_size = crop_size

        crop_rows = crop_size or nethack.nethack.TERMINAL_SHAPE[0]
        crop_cols = crop_size or nethack.nethack.TERMINAL_SHAPE[1]

        self.chw_image_shape = (
            3,
            crop_rows * self.char_height,
            crop_cols * self.char_width,
        )

        obs_spaces = {
            "screen_image": gym.spaces.Box(
                low=0, high=255, shape=self.chw_image_shape, dtype=np.uint8
            )
        }
        obs_spaces.update(
            [
                (k, self.env.observation_space[k])
                for k in self.env.observation_space
                # if k not in ["tty_chars", "tty_colors"]
            ]
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _populate_obs(self, obs):
        screen = np.zeros(self.chw_image_shape, order="C", dtype=np.uint8)
        render_utils.render_crop(
            obs["tty_chars"],
            obs["tty_colors"],
            obs["tty_cursor"],
            self.char_array,
            screen,
            crop_size=self.crop_size,
        )
        obs["screen_image"] = screen

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._populate_obs(obs)
        return obs, reward, term, trunc, info

    def reset(self):
        obs, info = self.env.reset()
        self._populate_obs(obs)
        return obs, info


if __name__ == "__main__":
    def run_episode(env, task: str = None, verbose=1):
        env.reset(new_task=task)
        task_name = type(env.unwrapped).__name__
        term = trunc = False
        ep_rew = 0
        while not (term or trunc):
            action = env.action_space.sample()
            _, rew, term, trunc, _ = env.step(action)
            ep_rew += rew
        if verbose:
            print(f"Episodic reward for {task_name}: {ep_rew}")

    print("Testing NethackTaskWrapper")
    N_EPISODES = 100

    # Initialize NLE
    nethack_env = NetHackScore()
    nethack_env = GymV21CompatibilityV0(env=nethack_env)

    nethack_task_env = NethackTaskWrapper(nethack_env)

    task_list = [
        NetHackScore,
        NetHackStaircase,
        NetHackStaircasePet,
        NetHackOracle,
        NetHackGold,
        NetHackEat,
        NetHackScout,
    ]

    start_time = time.time()

    for _ in range(N_EPISODES):
        run_episode(nethack_task_env, verbose=0)

    end_time = time.time()
    print(f"Run time same task: {end_time - start_time}")
    start_time = time.time()

    for i in range(N_EPISODES):
        nethack_task = task_list[i % 7]
        run_episode(nethack_task_env, task=nethack_task, verbose=0)

    end_time = time.time()
    print(f"Run time swapping tasks: {end_time - start_time}")
