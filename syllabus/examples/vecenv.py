import gym
import torch
import numpy as np
import time
from collections import deque


class VecEnv:
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_env(self, actions, reset_random=False):
        if reset_random:
            self.step_env_reset_random_async(actions)
        else:
            self.step_env_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


class VecEnvWrapper(VecEnv):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        VecEnv.__init__(self, num_envs=venv.num_envs,
                        observation_space=observation_space or venv.observation_space,
                        action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def reset(self):
        pass

    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.venv, name)


class VecEnvObservationWrapper(VecEnvWrapper):
    def process(self, obs):
        pass

    def reset(self):
        obs = self.venv.reset()
        return self.process(obs)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return self.process(obs), rews, dones, infos


class VecExtractDictObs(VecEnvObservationWrapper):
    def __init__(self, venv, key):
        self.key = key
        super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

    def process(self, obs):
        return obs[self.key]


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, use_tf=False):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def reset_agent(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset_agent()
        return self._obfilt(obs)

    def reset_random(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset_random()
        return self._obfilt(obs)

    def reset_alp_gmm(self, level):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset_alp_gmm(level)
        return self._obfilt(obs)


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename=None, keep_buf=0, info_keywords=()):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        self.results_writer = None
        self.info_keywords = info_keywords
        self.keep_buf = keep_buf
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                for k in self.info_keywords:
                    epinfo[k] = info[k]
                info['episode'] = epinfo
                if self.keep_buf:
                    self.epret_buf.append(ret)
                    self.eplen_buf.append(eplen)
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(epinfo)
                newinfos[i] = info
        return obs, rews, dones, newinfos


class RunningMeanStd():
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImageProcgen(TransposeObs):
    def __init__(self, env=None, op=[0, 3, 2, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImageProcgen, self).__init__(env)
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[2], obs_shape[1], obs_shape[0]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        if ob.shape[0] == 1:
            ob = ob[0]
        return ob.transpose(self.op[0], self.op[1], self.op[2], self.op[3])


class VecPyTorchProcgen(VecEnvWrapper):
    def __init__(self, venv, device, level_sampler=None):
        """
        Environment wrapper that returns tensors (for obs and reward)
        """
        super(VecPyTorchProcgen, self).__init__(venv)
        self.device = device

        self.level_sampler = level_sampler

        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [3, 64, 64],
            dtype=self.observation_space.dtype)

    @property
    def raw_venv(self):
        rvenv = self.venv
        while hasattr(rvenv, 'venv'):
            rvenv = rvenv.venv
        return rvenv

    def reset(self):
        if self.level_sampler:
            seeds = torch.zeros(self.venv.num_envs, dtype=torch.int)
            for e in range(self.venv.num_envs):
                seed = self.level_sampler.sample('sequential')
                seeds[e] = seed
                self.venv.seed(seed, e)

        obs = self.venv.reset()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.

        if self.level_sampler:
            return obs, seeds
        else:
            return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        # actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        # print(f"stepping {info[0]['level_seed']}, done: {done}")

        # reset environment here
        if self.level_sampler:
            for e in done.nonzero()[0]:
                seed = self.level_sampler.sample()
                self.venv.seed(seed, e)     # seed resets the corresponding level

            # NB: This reset call propagates upwards through all VecEnvWrappers
            obs = self.raw_venv.observe()['rgb']    # Note reset does not reset game instances, but only returns latest observations

        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()

        return obs, reward, done, info
