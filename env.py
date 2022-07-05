import bisect
import copy
import gym
import math
import numpy as np
import pickle
import requests
import ray
import tensorflow as tf

from ray import serve
from tensorflow.python.ops.gen_array_ops import shape
from sklearn.metrics import mean_squared_error

from data_prep import DataPrep
from tf_serve_models import TFEncoderDecoderModel
from VAE_dense import *


class CMAPSSEnv(gym.Env):
    def __init__(self,
                 df,
                 timestep,
                 obs_size=24,
                 engines=100,
                 engine_lives=[],
                 models=[None, None],
                 env_type="batch") -> None:
        super().__init__()
        self.obs_size = obs_size

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_size,)) # observations
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(2,)) # latent x -> based on mu, sigma coming from sampling layer; here we want the policy to retrieve that value

        self.df = df

        self.num_engines = engines
        self.engine_lives = engine_lives
        self.timestep = timestep

        # Load trained models
        self.models = models

        self.env_type = env_type
        

    def reset(self):

        if self.env_type == "batch":
            self.timestep = 0
            init_state = self.df.iloc[self.timestep,1:].tolist()
            init_state = np.array(init_state)
            
            return init_state 
        else:
            self.timestep = np.random.randint(sum(self.engine_lives))
            init_state = self.df.iloc[self.timestep,1:].tolist()
            init_state = np.array(init_state)
            
            return init_state 

    def step(self, action):

        #action = copy.copy(action)
        action = action.tolist()

        done = False

        """ Actual observation (from data) """
        actual_obs = self.df.iloc[self.timestep,1:].tolist()
                
        actuals = requests.get(
            "http://localhost:8000/saved_models", 
            json={"array": [[actual_obs], [action]]
                } 
            )
        
        """ Actual latent x from encoder """
        actual_latent_x = actuals.json()['predictions'][0]

        """ Estimated state|action """
        new_state = actuals.json()['predictions'][1][0]

        reconstructed = requests.get(
            "http://localhost:8000/saved_models", 
            json={"array": [[actual_obs], actual_latent_x]
                } #np.random.uniform(0,1,1).tolist()
            )

        """ Reconstructed state given true X"""    
        #_ = reconstructed.json()['predictions'][0][0]
        reconstructed_state = reconstructed.json()['predictions'][1][0]

        #print(new_state, type(new_state), reconstructed_state, type(reconstructed_state))
        
        """New state excluding the RUL estimate (not part of agent training)"""
        reward = self._reward(new_state, reconstructed_state, actual_latent_x[0], action)

        new_state = np.array(new_state)
        
        self.timestep += 1

        if self.env_type == "batch":            
            if self.timestep == np.sum(self.engine_lives):
                done = True
            
            return new_state, reward, done, {}
        else:
            if self.df['NormTime'].iloc[self.timestep] == float(0.0):
                done = True

            return new_state, reward, done, {}

    def render(self) -> None:
        env_snapshot = {
            "Current observation": self.df.iloc[self.timestep,1:].tolist(),
            "New state": list(self.decoder_model.predict(action, verbose=0)[0]),
        }
        print(env_snapshot)

    def _reward(self, est_state, rec_state, latent_x, act, latent=False):
        """
        Reward function can be also augmented for minimizing the error between the actual latent state as derived
        from the encoder with the action proposed from the policy network. Here we don't use that, but it can be 
        seamlessly added.
        """
        if latent:
            return -mean_squared_error(est_state, rec_state, squared=True) - mean_squared_error(act, latent_x, squared=True)
        else:
            return -mean_squared_error(est_state, rec_state, squared=True)


if __name__ == "__main__":
    import time

    serve.start()
    TFEncoderDecoderModel.deploy(['./saved_models/encoder','./saved_models/decoder'])
    
    const = Config()
    neurons = const.VAE_neurons

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

    env = CMAPSSEnv(**env_config)
    env = gym.wrappers.FrameStack(env, const.num_frames)

    total_cost = 0
    

    for _ in range(1):
        
        init_state = env.reset()
        cntr = 0

        """"""
        s = bisect.bisect_left(np.cumsum(engine_lives), env.timestep)
        steps_to_go = abs(env.timestep - np.cumsum(engine_lives[:s+1])[-1]) - 1
        current_step = abs(engine_lives[s] - steps_to_go)
        #print(f'Current step: {current_step},\
        #    System: {s}, System life: {engine_lives[s]}, Steps until failure: {steps_to_go}')
        

        while True:
            action = env.action_space.sample()
            obs, rew, done, _ = env.step(action)
            total_cost += rew
            cntr += 1
            print(rew, done)
            time.sleep(1)
            if done:
                break
        print(total_cost)