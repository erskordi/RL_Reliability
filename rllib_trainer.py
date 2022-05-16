import argparse
from json import load
import os
import pickle
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

import ray
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray import tune

from data_prep import DataPrep
from env import CMAPSSEnv
from VAE_dense import VAE

##### Command line arguments #####
parser = argparse.ArgumentParser(description="Build RL agent")
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--tune-log-level", type=str, default="INFO")
parser.add_argument("--env-logging", action="store_true")
parser.add_argument("--restore", type=str, default=None)
args = parser.parse_args()

##### RLlib setup #####
ray.init()

##### Load and configure problem instance #####
file_path = "CMAPSSData/train_FD002.txt"
num_settings = 3
num_sensors = 21
num_units = 40
prev_step_units = 200
step = "RL"

neurons = [64, 32, 16, 8]

# Data prep
data = DataPrep(file=file_path,
                num_settings=num_settings, 
                num_sensors=num_sensors, 
                num_units=num_units, 
                prev_step_units=prev_step_units,
                step=step,
                normalization_type="01")

df = data.ReadData()

# List of engine lifetimes
engine_lives = df.groupby(df['Unit']).size()
engine_lives = engine_lives.tolist()
num_engines = len(engine_lives)

# Load options
#vae = VAE(latent_dim=1,image_size=25)

with open('/Users/erotokritosskordilis/git-repos/RL_Reliability/decoder.pkl', 'rb') as f:
    decoder = pickle.load(f)

env_config = {
    "df": df,
    "timestep": 0,
    "obs_size": num_settings+num_sensors+1,
    "engines": num_engines,
    "engine_lives": engine_lives, 
    "model": decoder,
}

env_name = "CMAPSS_env"
register_env(env_name, lambda config: CMAPSSEnv(**env_config))


##### Run TUNE experiments #####
tune.run(
    "PPO",
    name=env_name,
    checkpoint_freq=3,
    checkpoint_at_end=True,
    checkpoint_score_attr="episode_reward_mean",
    keep_checkpoints_num=50,
    stop={"training_iteration": 10000},
    restore=args.restore,
    config={
        "env": env_name,
        "num_workers": args.num_cpus,
        "num_gpus": args.num_gpus,
        "log_level": args.tune_log_level,
        #"train_batch_size": np.sum(engine_lives),
        "ignore_worker_failures": True,
    }
)