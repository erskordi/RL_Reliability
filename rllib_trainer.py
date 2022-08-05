import argparse
from json import load
import os
import pickle
import sys

import gym
import numpy as np
import pandas as pd
import tensorflow as tf

import ray
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray import tune, serve

from data_prep import DataPrep
from env import CMAPSSEnv
from tf_serve_models import TFEncoderDecoderModel
from VAE_dense import *


##### Command line arguments #####
parser = argparse.ArgumentParser(description="Build RL agent")
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--num-frames", type=int, default=20)
parser.add_argument("--tune-log-level", type=str, default="INFO")
parser.add_argument("--env-logging", action="store_true")
parser.add_argument("--restore", type=str, default=None)
args = parser.parse_args()

##### RLlib setup #####
ray.init()

##### Load and configure problem instance #####
const = Config()

# Data prep
data = DataPrep(file = const.file_path,
                num_settings = const.num_settings,
                num_sensors = const.num_sensors,
                num_units = const.num_units[1],
                prev_step_units = const.prev_step_units[1],
                step = const.step[1],
                normalization_type="01")

df = data.ReadData()


# List of engine lifetimes
engine_lives = df.groupby(df['Unit']).size()
engine_lives = engine_lives.tolist()
num_engines = len(engine_lives)

# Load options
serve.start()
TFEncoderDecoderModel.deploy(['./saved_models/encoder','./saved_models/decoder'])

const = Config()
neurons = const.VAE_neurons

# Environment types
env_types = ["batch", "intertemporal"]

##########################################
env_config = {
    "df": df,
    "timestep": 0,
    "obs_size": const.num_settings+const.num_sensors+1,
    "engines": num_engines,
    "engine_lives": engine_lives, 
    "models": [None, None],
    "env_type": env_types[1],
}

#print("env_config: ", env_config)

env_name = "CMAPSS_env"
env_wrapper_config = gym.wrappers.FrameStack(CMAPSSEnv(**env_config), const.num_frames)
register_env(env_name, lambda config: env_wrapper_config)


##### Run TUNE experiments #####
tune.run(
    "PPO",
    name=env_name,
    checkpoint_freq=3,
    checkpoint_at_end=True,
    checkpoint_score_attr="episode_reward_mean",
    keep_checkpoints_num=50,
    stop={"training_iteration": 100},
    restore=args.restore,
    config={
        "env": env_name,
        "num_workers": args.num_cpus,
        "num_gpus": args.num_gpus,
        "log_level": args.tune_log_level,
        "rollout_fragment_length": 4000 // args.num_cpus,
        #"horizon": np.sum(engine_lives),
        "ignore_worker_failures": True,
        "model":{
            "fcnet_hiddens": const.policy_neurons,
            "fcnet_activation": "relu",
            "free_log_std": True,
        }
    }
)