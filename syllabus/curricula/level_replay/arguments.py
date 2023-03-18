# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/rraileanu/auto-drac/blob/master/ucb_rl2_meta/arguments.py

import argparse

import torch

parser = argparse.ArgumentParser(description='RL')

# PPO Arguments. 
parser.add_argument(
    '--lr', 
    type=float, 
    default=5e-4, 
    help='learning rate')
parser.add_argument(
    '--eps',
    type=float,
    default=1e-5,
    help='RMSprop optimizer epsilon')
parser.add_argument(
    '--alpha',
    type=float,
    default=0.99,
    help='RMSprop optimizer apha')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.999,
    help='discount factor for rewards')
parser.add_argument(
    '--gae_lambda',
    type=float,
    default=0.95,
    help='gae lambda parameter')
parser.add_argument(
    '--entropy_coef',
    type=float,
    default=0.01,
    help='entropy term coefficient')
parser.add_argument(
    '--value_loss_coef',
    type=float,
    default=0.5,
    help='value loss coefficient (default: 0.5)')
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.5,
    help='max norm of gradients)')
parser.add_argument(
    '--no_ret_normalization',
    action='store_true',
    help='Whether to use unnormalized returns')
parser.add_argument(
    '--seed', 
    type=int, 
    default=1, 
    help='random seed')
parser.add_argument(
    '--num_processes',
    type=int,
    default=64,
    help='how many training CPU processes to use')
parser.add_argument(
    '--num_steps',
    type=int,
    default=256,
    help='number of forward steps in A2C')
parser.add_argument(
    '--ppo_epoch',
    type=int,
    default=3,
    help='number of ppo epochs')
parser.add_argument(
    '--num_mini_batch',
    type=int,
    default=8,
    help='number of batches for ppo')
parser.add_argument(
    '--clip_param',
    type=float,
    default=0.2,
    help='ppo clip parameter')
parser.add_argument(
    '--num_env_steps',
    type=int,
    default=25e6,
    help='number of environment steps to train')
parser.add_argument(
    '--env_name',
    type=str,
    default='bigfish',
    help='environment to train on')
parser.add_argument(
    '--xpid',
    default='latest',
    help='name for the run - prefix to log files')
parser.add_argument(
    '--log_dir',
    default='~/logs/ppo/',
    help='directory to save agent logs')
parser.add_argument(
    '--no_cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--hidden_size',
    type=int,
    default=256,
    help='state embedding dimension')
parser.add_argument(
    '--arch',
    type=str,
    default='large',
    choices=['small', 'large'],
    help='agent architecture')

# Procgen arguments.
parser.add_argument(
    '--distribution_mode',
    default='easy',
    help='distribution of envs for procgen')
parser.add_argument(
    '--paint_vel_info',
    action='store_true',
    help='Paint velocity vector at top of frames.')
parser.add_argument(
    '--num_train_seeds',
    type=int,
    default=200,
    help='number of Procgen levels to use for training')
parser.add_argument(
    '--start_level',
    type=int,
    default=0,
    help='start level id for sampling Procgen levels')
parser.add_argument(
    "--num_test_seeds", 
    type=int,
    default=10,
    help="Number of test seeds")
parser.add_argument(
    "--final_num_test_seeds", 
    type=int,
    default=1000,
    help="Number of test seeds")
parser.add_argument(
    '--seed_path',
    type=str,
    default=None,
    help='Path to file containing specific training seeds')
parser.add_argument(
    "--full_train_distribution",
    action='store_true',
    help="Train on the full distribution")

# Level Replay arguments.
parser.add_argument(
    "--level_replay_score_transform",
    type=str, 
    default='softmax', 
    choices=['constant', 'max', 'eps_greedy', 'rank', 'power', 'softmax'], 
    help="Level replay scoring strategy")
parser.add_argument(
    "--level_replay_temperature", 
    type=float,
    default=1.0,
    help="Level replay scoring strategy")
parser.add_argument(
    "--level_replay_strategy", 
    type=str,
    default='random',
    choices=['off', 'random', 'sequential', 'policy_entropy', 'least_confidence', 'min_margin', 'gae', 'value_l1', 'one_step_td_error'],
    help="Level replay scoring strategy")
parser.add_argument(
    "--level_replay_eps", 
    type=float,
    default=0.05,
    help="Level replay epsilon for eps-greedy sampling")
parser.add_argument(
    "--level_replay_schedule",
    type=str,
    default='proportionate',
    help="Level replay schedule for sampling seen levels")
parser.add_argument(
    "--level_replay_rho",
    type=float, 
    default=1.0,
    help="Minimum size of replay set relative to total number of levels before sampling replays.")
parser.add_argument(
    "--level_replay_nu", 
    type=float,
    default=0.5,
    help="Probability of sampling a new level instead of a replay level.")
parser.add_argument(
    "--level_replay_alpha",
    type=float, 
    default=1.0,
    help="Level score EWA smoothing factor")
parser.add_argument(
    "--staleness_coef",
    type=float, 
    default=0.0,
    help="Staleness weighing")
parser.add_argument(
    "--staleness_transform",
    type=str, 
    default='power',
    choices=['max', 'eps_greedy', 'rank', 'power', 'softmax'], 
    help="Staleness normalization transform")
parser.add_argument(
    "--staleness_temperature",
    type=float, 
    default=1.0,
    help="Staleness normalization temperature")

# Logging arguments
parser.add_argument(
    "--verbose", 
    action="store_true",
    help="Whether to print logs")
parser.add_argument(
    '--log_interval',
    type=int,
    default=1,
    help='log interval, one log per n updates')
parser.add_argument(
    "--save_interval", 
    type=int, 
    default=60,
    help="Save model every this many minutes.")
parser.add_argument(
    "--weight_log_interval", 
    type=int, 
    default=1,
    help="Save level weights every this many updates")
parser.add_argument(
    "--disable_checkpoint", 
    action="store_true",
    help="Disable saving checkpoint.")