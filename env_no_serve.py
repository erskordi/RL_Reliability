import bisect
import copy
import gym
import numpy as np
import pickle
import requests
import tensorflow as tf

from tensorflow.python.ops.gen_array_ops import shape
from sklearn.metrics import mean_squared_error

from data_prep import DataPrep
from VAE_dense import *


class CMAPSSEnv(gym.Env):
    def __init__(self,
                 df,
                 timestep,
                 obs_size=24,
                 engines=100,
                 engine_lives=[],
                 decoder_model=None,
                 env_type="batch") -> None:
        super().__init__()
        self.obs_size = obs_size

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_size,)) # observations + RUL
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,)) # latent x -> based on mu, sigma coming from sampling layer; here we want the policy to retrieve that value

        self.df = df

        self.num_engines = engines
        self.engine_lives = engine_lives
        self.timestep = timestep

        # Load trained models
        self.decoder_model = decoder_model
        
        
        self.env_type = env_type
        

    def reset(self):

        if self.env_type == "batch":
            self.timestep = 0
            init_state = self.df.iloc[self.timestep,1:].tolist()
            
            return init_state 
        else:
            self.timestep = np.cumsum(self.engine_lives[:len(self.engine_lives)-1])[-1]#np.random.randint(sum(self.engine_lives))
            init_state = self.df.iloc[self.timestep,1:].tolist()
        
            return init_state 

    def step(self, action):
        
        action = action.copy()
        
        if self.env_type == "batch":
            
            self.timestep += 1
        
            done = False
            #encoded_data = self.encoder.predict(obs)
            new_state = list(self.decoder_model.predict(action)[0])
            actual_obs = self.df.iloc[self.timestep,1:].tolist()
            
            reward = self._reward(actual_obs, new_state)
            
            if self.timestep == len(self.df)-1:
                done = True
            
            return new_state, reward, done, {}
        else:
            
            self.timestep += 1
            
            s = bisect.bisect_left(np.cumsum(self.engine_lives), self.timestep)
            steps_to_go = abs(self.timestep - np.cumsum(self.engine_lives[:s+1])[-1])
            
            done = False
            #encoded_data = self.encoder.predict(obs)
            new_state = list(self.decoder_model.predict(action)[0])
            actual_obs = self.df.iloc[self.timestep,1:].tolist()
            
            reward = self._reward(actual_obs, new_state) / steps_to_go if steps_to_go != 0 else self._reward(actual_obs, new_state)
            
            if self.df['NormTime'].iloc[self.timestep-1] == float(0.0):
                done = True
            
            return new_state, reward, done, {}
        

    def render(self) -> None:
        pass

    def _reward(self, y_true, y_pred):
        return -mean_squared_error(y_pred, y_true)


if __name__ == "__main__":
    
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

    # Load decoder
    decoder = tf.keras.models.load_model('./saved_models/decoder')
    
    # Environment types
    env_types = ["batch", "intertemporal"]

    ##########################################
    env_config = {
        "df": df,
        "timestep": 0,
        "obs_size": const.num_settings+const.num_sensors+1,
        "engines": num_engines,
        "engine_lives": engine_lives, 
        "decoder_model": decoder,
        "env_type": env_types[0],
    }

    print("env_config: ", env_config)

    env = CMAPSSEnv(**env_config)
    

    total_cost = 0
    

    for _ in range(1):
        
        env.reset()
        cntr = 0
        s = bisect.bisect_left(np.cumsum(engine_lives), env.timestep)
        steps_to_go = abs(env.timestep - np.cumsum(engine_lives[:s+1])[-1])
        current_step = abs(engine_lives[s] - steps_to_go)
        print(f'Current step: {current_step},\
            System: {s}, System life: {engine_lives[s]}, Steps until failure: {steps_to_go}')
        
        while True:
            action = env.action_space.sample()
            obs, rew, done, _ = env.step(action)
            total_cost += rew
            cntr += 1
            print(current_step+cntr, rew, done)
            if done:
                break
        print(total_cost)