# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gym
from gym_minigrid.register import register

from custom_envs.obstructedmaze_fixedgrid import ObstructedMazeEnvFixedGrid


ALL_SUBENVS = [
	'MiniGrid-ObstructedMaze-1Dl-fixed_grid-v0',
	'MiniGrid-ObstructedMaze-1Dlh-fixed_grid-v0',
	'MiniGrid-ObstructedMaze-1Dlhb-fixed_grid-v0',
	'MiniGrid-ObstructedMaze-2Dl-fixed_grid-v0',
	'MiniGrid-ObstructedMaze-2Dlh-fixed_grid-v0',
	'MiniGrid-ObstructedMaze-2Dlhb-fixed_grid-v0',
	'MiniGrid-ObstructedMaze-1Q-fixed_grid-v0',
	'MiniGrid-ObstructedMaze-2Q-fixed_grid-v0',
	'MiniGrid-ObstructedMaze-Full-fixed_grid-v0'
]

TILE_PIXELS = 32


class ObstructedMazeGamut(gym.Env):
	def __init__(self, distribution='easy', max_difficulty=None, seed=1337):

		self.distribution = distribution
		if distribution == 'easy':
			self.max_difficulty = 3
		elif distribution == 'medium':
			self.max_difficulty = 6
		elif distribution == 'hard':
			self.max_difficulty = 9
		else:
			raise ValueError(f'Unsupported distribution {distribution}.')

		if max_difficulty is not None:
			self.max_difficulty = max_difficulty

		self.subenvs = []
		for env_name in ALL_SUBENVS[:self.max_difficulty]:
			self.subenvs.append(gym.make(env_name))

		self.num_subenvs = len(self.subenvs)

		self.seed(seed)
		self.reset()

	@property
	def actions(self):
		return self.env.actions

	@property
	def agent_view_size(self):
		return self.env.agent_view_size

	@property
	def reward_range(self):
		return self.env.reward_range

	@property
	def window(self):
		return self.env.window

	@property
	def width(self):
		return self.env.width
	
	@property
	def height(self):
		return self.env.height

	@property
	def grid(self):
		return self.env.grid
	
	@property
	def max_steps(self):
		return self.env.max_steps

	@property
	def see_through_walls(self):
		return self.env.see_through_walls

	@property
	def agent_pos(self):
		return self.env.agent_pos

	@property
	def agent_dir(self):
		return self.env.agent_dir

	@property
	def step_count(self):
		return self.env.step_count

	@property
	def carrying(self):
		return self.env.carrying

	@property
	def observation_space(self):
		return self.env.observation_space

	@property
	def action_space(self):
		return self.env.action_space

	@property
	def steps_remaining(self):
		return self.env.steps_remaining

	def __str__(self):
		return self.env.__str__()

	def reset(self):
		return self.env.reset()

	def seed(self, seed=1337):
		env_index = seed % self.num_subenvs
		self.env = self.subenvs[env_index]
		self.env.seed(seed)

	def hash(self, size=16):
		return self.env.hash(size)

	def relative_coords(self, x, y):
		return self.env.relative_coords(x, y)

	def in_view(self, x, y):
		return self.env.in_view(x, y)

	def agent_sees(self, x, y):
		return self.env.agent_sees(x, y)

	def step(self, action):
		return self.env.step(action)

	def gen_obs_grid(self):
		return self.env.gen_obs_grid()

	def gen_obs(self):
		return self.env.gen_obs()

	def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
		return self.env.get_obs_render(obs, tile_size)

	def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
		return self.env.render(mode, close, highlight, tile_size)

	def close(self):
		return self.env.close()


class ObstructedMazeGamut_Easy(ObstructedMazeGamut):
    def __init__(self, seed=1337):
        super().__init__(distribution='easy', seed=seed)

class ObstructedMazeGamut_Medium(ObstructedMazeGamut):
    def __init__(self, seed=1337):
        super().__init__(distribution='medium', seed=seed)

class ObstructedMazeGamut_Hard(ObstructedMazeGamut):
    def __init__(self, seed=1337):
        super().__init__(distribution='hard', seed=seed)


register(
    id="MiniGrid-ObstructedMazeGamut-Easy-v0",
    entry_point=f"{__name__}:ObstructedMazeGamut_Easy"
)

register(
    id="MiniGrid-ObstructedMazeGamut-Medium-v0",
    entry_point=f"{__name__}:ObstructedMazeGamut_Medium"
)

register(
    id="MiniGrid-ObstructedMazeGamut-Hard-v0",
    entry_point=f"{__name__}:ObstructedMazeGamut_Hard"
)
