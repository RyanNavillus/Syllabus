import glob
import os
import shutil
import tempfile

import gym
import numpy as np
import torch
import yaml

# from gym.utils.play import play
from griddly import GymWrapper as GriddlyGymWrapper
from griddly import gd
from griddly.util.action_space import MultiAgentActionSpace

try:
    import PIL.Image
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "To safe GIF files of trajectories, please install Pillow:"
        " pip install Pillow"
    )


class Lasertag(GriddlyGymWrapper):
    def __init__(
        self,
        *args,
        n_agents=5,  # max number of agents, can be less than this
        agent_view_size=5,  # agent view size
        record_video=False,  # recording video during evaluation
        video_filename="",  # where to record the video
        zero_sum=True,  # Dead agents receive a reward of -1
        mask_actions=False,
        survival_mode=False,  # Only reward the agent for not dying
        **kwargs,
    ):
        self.step_count = 0
        self.n_agents = n_agents
        self.agent_view_size = agent_view_size
        self.zero_sum = zero_sum
        self.mask_actions = mask_actions
        self.survival_mode = survival_mode

        self.record_video = record_video
        self.recording_started = False
        self.video_filename = video_filename

        yaml_filename = "gdy/lasertag_wall_general_zs.yaml"

        yaml_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), yaml_filename
        )
        with open(yaml_path, "r") as stream:
            yaml_dict = yaml.safe_load(stream)
            # Fixing the number of agents
            yaml_dict["Environment"]["Player"]["Count"] = self.n_agents
            # Fixing the agent view size
            yaml_dict["Environment"]["Player"]["Observer"]["Height"] = agent_view_size
            yaml_dict["Environment"]["Player"]["Observer"]["Width"] = agent_view_size
            yaml_dict["Environment"]["Player"]["Observer"]["OffsetY"] = 0
            yaml_dict["Environment"]["Player"]["Observer"]["OffsetY"] = (
                agent_view_size // 2
            )
            self.yaml_string = yaml.dump(
                yaml_dict, default_flow_style=False, sort_keys=False
            )

        kwargs["yaml_string"] = self.yaml_string
        kwargs["player_observer_type"] = kwargs.pop(
            "player_observer_type", gd.ObserverType.VECTOR
        )
        kwargs["global_observer_type"] = kwargs.pop(
            "global_observer_type", gd.ObserverType.BLOCK_2D
        )
        kwargs["max_steps"] = kwargs.pop("max_steps", 200)

        super().__init__(*args, **kwargs)

        self.action_map = {
            0: [0, 0],  # no-op
            1: [0, 1],  # left
            2: [0, 2],  # move
            3: [0, 3],  # right
            4: [1, 1],  # shoot
        }
        self.n_actions = len(self.action_map)

    @property
    def observation_space(self):
        obs_dict = {}
        self.image_obs_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                self.n_agents,
                3,
                self.agent_view_size,
                self.agent_view_size,
            ),
            dtype="uint8",
        )
        obs_dict["image"] = self.image_obs_space

        if self.mask_actions:
            self.avail_act_obs_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.n_agents, self.n_actions),
                dtype="bool",
            )
            obs_dict["avail_actions"] = self.avail_act_obs_space

        return gym.spaces.Dict(obs_dict)

    @property
    def action_space(self):
        return MultiAgentActionSpace(
            [gym.spaces.Discrete(self.n_actions) for _ in range(self.n_agents)]
        )

    def get_active_agents(self):
        state = self.get_state()
        active_agents = set()

        for object in state["Objects"]:
            if object["Name"] == "agent":
                active_agents.add(object["PlayerId"] - 1)

        return active_agents

    def step(self, actions):
        if torch.is_tensor(actions):
            actions = actions.numpy()
        assert len(actions) == self.n_agents

        if self.mask_actions:
            for a in range(self.n_agents):
                # Make sure only available actions are chosen
                avail_actions = self.get_avail_agent_actions(a)
                assert avail_actions[actions[a]] == 1

        actions = [self.action_map[a] for a in actions]
        obs, reward, done, info = super().step(actions)
        self.step_count += 1

        if self.record_video:
            frame = self.render(mode="rgb_array")
            image = PIL.Image.fromarray(frame)
            image.save(os.path.join(self.tmpdir, f"e_s_{self.step_count}.png"))

        # Terminate if less than two agents
        self.active_agents = self.get_active_agents()
        if len(self.active_agents) < 2:
            done = True
            info["solved"] = True
        elif done:
            info["solved"] = False

        if done and self.record_video and self.recording_started:
            gif_path = self.video_filename
            if gif_path == "":
                gif_path = "lasertag.gif"
            elif gif_path.endswith("mp4"):
                gif_path = gif_path[:-4] + ".gif"
            # Make the GIF and delete the temporary directory
            png_files = glob.glob(os.path.join(self.tmpdir, "e_s_*.png"))
            png_files.sort(key=os.path.getmtime)

            img, *imgs = [PIL.Image.open(f) for f in png_files]
            img.save(
                fp=gif_path,
                format="GIF",
                append_images=imgs,
                save_all=True,
                duration=60,
                loop=0,
            )
            shutil.rmtree(self.tmpdir)

            print("Saving replay GIF at {}".format(os.path.abspath(gif_path)))

        # dead agents don't get rewarded

        if not self.zero_sum:
            for a in range(self.n_agents):
                if reward[a] != 0 and a not in self.active_agents:
                    reward[a] = 0

        # Survival mode: re-compute all rewards
        if self.survival_mode:
            reward = [0] * self.n_agents
            if len(self.active_agents) == 1:
                active_agent_index = next(iter(self.active_agents))
                reward[active_agent_index] = 1

        return self.update_obs(obs), reward, done, info

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        if agent_id not in self.active_agents:  # agent is dead
            # only no_op allowed
            return [1] + [0] * (self.n_actions - 1)
        else:
            # everything other than no_op allowed
            return [0] + [1] * (self.n_actions - 1)

    def reset(self, *args, level=None, **kwargs):
        self.step_count = 0
        if level is None:
            obs = super().reset(*args, **kwargs)
        else:
            assert isinstance(level, np.ndarray)
            assert len(level.shape) == 2
            shape_0 = level.shape[0]
            shape_1 = level.shape[1]

            # Wallify from four sides
            lvl = np.full((shape_0 + 2, shape_1 + 2), "w", dtype="U5")
            lvl[1 : shape_0 + 1, 1 : shape_1 + 1] = level

            # translate into a string
            level_list = ["\t".join(lvl[i]) + "\n" for i in range(lvl.shape[0])]
            obs = super().reset(*args, level_string="".join(level_list), **kwargs)

        self.active_agents = self.get_active_agents()

        if self.record_video and not self.recording_started:
            frame = self.render(mode="rgb_array")
            # creating gifs not videos
            self.tmpdir = tempfile.mkdtemp()
            image = PIL.Image.fromarray(frame)
            image.save(os.path.join(self.tmpdir, f"e_s_{self.step_count}.png"))
            self.recording_started = True

        return self.update_obs(obs)

    def update_obs(self, obs):
        obs_dict = {}
        obs_dict["image"] = np.stack(obs, axis=0)
        if self.mask_actions:
            obs_dict["avail_actions"] = np.stack(self.get_avail_actions(), axis=0)
        # Zero out obs of dead agents
        for a in range(self.player_count):
            if a not in self.active_agents:
                obs_dict["image"][a, :, :, :] = 0

        return obs_dict

    def seed(self, seed=None):
        # super().seed(seed=seed)
        np.random.seed(seed)
        return seed

    def render(self, mode=None):
        return super().render(observer="global", mode="rgb_array")
