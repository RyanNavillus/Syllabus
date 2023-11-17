import gym
import procgen  # noqa: F401
import random
import numpy as np
from procgen import ProcgenEnv
from syllabus.tests import evaluate_random_policy
from syllabus.examples.task_wrappers import ProcgenTaskWrapper
import multiprocessing as mp
from syllabus.core import MultiProcessingSyncWrapper
from syllabus.examples.vecenv import VecNormalize, VecMonitor, VecExtractDictObs


N_EPISODES = 10
N_ENVS = 4
import os
seeds = [int.from_bytes(os.urandom(N_ENVS), byteorder="little") for _ in range(10)]
print(seeds)

def wrap_vecenv(vecenv):
    vecenv.is_vector_env = True
    # if args.capture_video:
    #     vecenv = gym.wrappers.RecordVideo(vecenv, f"videos/{run_name}")
    # vecenv = gym.wrappers.NormalizeReward(vecenv, gamma=args.gamma)
    # vecenv = gym.wrappers.TransformReward(vecenv, lambda reward: np.clip(reward, -10, 10))
    vecenv = VecMonitor(venv=vecenv, filename=None, keep_buf=100)
    vecenv = VecNormalize(venv=vecenv, ob=False, ret=True)
    return vecenv

def make_env(seed=None, start_level=0, num_levels=1, rand_seed=0):
    def thunk():
        env_id = "bigfish"
        env = gym.make(f"procgen-{env_id}-v0", start_level=start_level, num_levels=num_levels, rand_seed=rand_seed, distribution_mode="easy")
        env = ProcgenTaskWrapper(env, env_id, seed)
        # Seed environment
        # gym.utils.seeding.np_random(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env
    return thunk

# env = make_env(seed=0)
# env.seed(0)
# obs = env.reset()
# print(np.mean(obs))
# env.close()

print("\nProcgenEnv 1")
envs = ProcgenEnv(num_envs=N_ENVS, env_name="bigfish", start_level=0, num_levels=1, distribution_mode="easy")
action = np.asarray([envs.action_space.sample() for _ in range(N_ENVS)])
for e in range(N_ENVS):
    envs.seed(1, e)
obs_1_procgen = envs.reset()
obs_2_procgen, _, _, _ = envs.step(action)
envs.close()

print("\nProcgenEnv 2")
envs = ProcgenEnv(num_envs=N_ENVS, env_name="bigfish", start_level=0, num_levels=1, distribution_mode="easy")
for e in range(N_ENVS):
    envs.seed(1, e)
obs_3_procgen = envs.reset()
obs_4_procgen, _, _, _ = envs.step(action)
envs.close()
assert np.allclose(np.array(obs_1_procgen["rgb"], dtype=float), np.array(obs_3_procgen["rgb"], dtype=float))
assert np.allclose(np.array(obs_2_procgen["rgb"], dtype=float), np.array(obs_4_procgen["rgb"], dtype=float))
assert np.allclose(obs_1_procgen["rgb"][0,:,:,:], obs_1_procgen["rgb"][1,:,:,:],), "Initial observations are different for first and second environment"

print("\nVecEnv 1")
envs = gym.vector.SyncVectorEnv(
    [
        make_env(1)
        for i in range(N_ENVS)
    ]
)
envs.is_vector_env = True
# envs.seed([0]*N_ENVS)
obs_1_vector = envs.reset()
obs_2_vector, _, _, _ = envs.step(action)
envs.close()

print("\nVecEnv 2")
envs = gym.vector.SyncVectorEnv(
    [
        make_env(1)
        for i in range(N_ENVS)
    ]
)
envs.is_vector_env = True
# envs.seed([0]*N_ENVS)
obs_3_vector = envs.reset()
obs_4_vector, _, _, _ = envs.step(action)
envs.close()
assert np.allclose(np.array(obs_1_vector, dtype=float), np.array(obs_3_vector, dtype=float))
assert np.allclose(np.array(obs_2_vector, dtype=float), np.array(obs_4_vector, dtype=float))
assert np.allclose(obs_1_vector[0,:,:,:], obs_1_vector[1,:,:,:]), "Initial observations are different for first and second environment."


# Passes only when N_ENVS = 1
for env in range(N_ENVS):
    print("Env {}".format(env))
    assert np.allclose(obs_1_procgen["rgb"].transpose(0,3,1,2)[env,:,:,:] / 255.0, obs_1_vector[env,:,:,:]), "Initial observations are different"
    assert np.allclose(obs_2_procgen["rgb"].transpose(0,3,1,2)[env,:,:,:] / 255.0, obs_2_vector[env,:,:,:]), "First step observations are different"


print("\nProcgenEnv 3")
envs = ProcgenEnv(num_envs=N_ENVS, env_name="bigfish", start_level=0, num_levels=1, distribution_mode="easy")
for e in range(N_ENVS):
    envs.seed(e, e)
obs_1_procgen = envs.reset()

print("\nVecEnv 3")
vec_envs = gym.vector.SyncVectorEnv(
    [
        make_env(i, num_levels=1)
        for i in range(N_ENVS)
    ]
)
vec_envs.is_vector_env = True
# envs.seed([0]*N_ENVS)
obs_1_vector = vec_envs.reset()


# Complete episode
print("\nComparing episode")
done = False
for i in range(1000):
    action = np.asarray([envs.action_space.sample() for _ in range(N_ENVS)])
    obs_2_procgen, _, dones_procgen, _ = envs.step(action)
    obs_2_vector, _, dones_vector, _ = vec_envs.step(action)
    for env in range(N_ENVS):
        assert np.allclose(obs_2_procgen["rgb"].transpose(0,3,1,2)[env,:,:,:] / 255.0, obs_2_vector[env,:,:,:]), "First step observations are different"
    # print(np.allclose(obs_2_vector[0,:,:,:], obs_2_vector[1,:,:,:]))
    # print(np.allclose(obs_2_procgen["rgb"].transpose(0,3,1,2)[0,:,:,:], obs_2_procgen["rgb"].transpose(0,3,1,2)[1,:,:,:],))
    # seed = random.randint(0,200)
    # for e in dones_procgen.nonzero()[0]:
    #     print(dones_procgen.nonzero()[0])
    #     print("reset_procgen")
    #     envs.seed(seed, e)
    # for e in dones_vector.nonzero()[0]:
    #     print("reset_vector")
    #     vec_envs.envs[e].seed(seed)
envs.close()
vec_envs.close()

print("\nProcgenEnv 4")
envs = ProcgenEnv(num_envs=N_ENVS, env_name="bigfish", start_level=0, num_levels=1, rand_seed=0, distribution_mode="easy")
envs = wrap_vecenv(envs)
for e in range(N_ENVS):
    envs.seed(0, e)
obs_1_procgen = envs.reset()
assert np.allclose(obs_1_procgen["rgb"][0,:,:,:], obs_1_procgen["rgb"][1,:,:,:],), "Observations are different for first and second environment"

print("\nVecEnv 4")
vec_envs = gym.vector.SyncVectorEnv(
    [
        make_env(0, num_levels=1, rand_seed=0)
        for i in range(N_ENVS)
    ]
)
vec_envs = wrap_vecenv(vec_envs)

vec_envs.is_vector_env = True
# envs.seed([0]*N_ENVS)
obs_1_vector = vec_envs.reset()
assert np.allclose(obs_1_vector[0,:,:,:], obs_1_vector[1,:,:,:]), "Observations are different for first and second environment."

for env in range(N_ENVS):
    print("Env {}".format(env))
    assert np.allclose(obs_1_procgen["rgb"].transpose(0,3,1,2)[env,:,:,:] / 255.0, obs_1_vector[env,:,:,:]), "Initial observations are different"

# Complete episode
print("\nComparing episode")
done = False
for i in range(500):
    # print(i)
    a = envs.action_space.sample()
    action = np.asarray([a for _ in range(N_ENVS)])
    obs_2_procgen, rews_procgen, dones_procgen, _ = envs.step(action)
    obs_2_vector, rews_vector, dones_vector, _ = vec_envs.step(action)
    for env in range(N_ENVS):
        assert np.allclose(obs_2_procgen["rgb"].transpose(0,3,1,2)[env,:,:,:] / 255.0, obs_2_vector[env,:,:,:]), "First step observations are different"
        assert np.allclose(rews_procgen[env], rews_vector[env]), "Rewards are different"

    assert np.allclose(obs_2_vector[0,:,:,:], obs_2_vector[1,:,:,:])
    assert np.allclose(obs_2_procgen["rgb"].transpose(0,3,1,2)[0,:,:,:], obs_2_procgen["rgb"].transpose(0,3,1,2)[1,:,:,:],)

    seed = random.randint(0,200)
    for e in dones_procgen.nonzero()[0]:
        # print(dones_procgen.nonzero()[0])
        # print("reset_procgen")
        envs.seed(seed, e)
    for e in dones_vector.nonzero()[0]:
        # print("reset_vector")
        vec_envs.envs[e].seed(seed)

envs.close()
vec_envs.close()