import bisect
import gym
import numpy as np
import pickle
import requests
import tensorflow as tf

from tensorflow.python.ops.gen_array_ops import shape
from sklearn.metrics import mean_squared_error

from data_prep import DataPrep
from tf_decoder_model import TFDecoderModel
from VAE_dense import *


class CMAPSSEnv(gym.Env):
    def __init__(self,
                 df,
                 timestep,
                 obs_size=24,
                 engines=100,
                 engine_lives=[],
                 decoder_model=None) -> None:
        super().__init__()
        self.obs_size = obs_size

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_size,)) # observations + RUL
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,)) # latent x -> based on mu, sigma coming from sampling layer; here we want the policy to retrieve that value

        self.df = df

        self.num_engines = engines
        self.engine_lives = engine_lives
        self.timestep = timestep

        # Load trained models
        self.model = decoder_model
        

    def reset(self):

        self.timestep = np.random.randint(sum(self.engine_lives))
        init_state = self.df.iloc[self.timestep,1:].to_numpy()
        #print(f'Initial state: {init_state}, dimensions: {init_state.shape}')
        
        return init_state 

    def step(self, action):
                
        resp = requests.get(
            "http://localhost:8000/saved_models", json={"array": action.tolist()}
        )
        new_state = resp.json()['prediction'][0]
        
        reward = self._reward(self.df.iloc[self.timestep,1:], new_state)
        
        if self.df['NormTime'].iloc[self.timestep] == float(0.0):
            done = True
        
        self.timestep += 1
        
        return new_state, reward, done, {}

    def render(self) -> None:
        pass

    def _reward(self, y_true, y_pred):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(y_true, y_pred))
                )
        return -reconstruction_loss


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
    with open('./decoder.pkl', 'rb') as f:
        decoder = pickle.load(f)

    ##########################################
    env_config = {
        "df": df,
        "timestep": 0,
        "obs_size": const.num_settings+const.num_sensors+1,
        "engines": num_engines,
        "engine_lives": engine_lives, 
        "decoder_model": decoder,
    }

    print("env_config: ", env_config)

    env = CMAPSSEnv(**env_config)
    

    total_cost = 0
    

    for _ in range(1):
        
        done = False
        env.reset()
        print(env.timestep)
        cntr = 0
        s = bisect.bisect_left(np.cumsum(engine_lives), env.timestep)
        steps_to_go = abs(env.timestep - np.cumsum(engine_lives[:s+1])[-1])
        current_step = abs(engine_lives[s] - steps_to_go)
        print(f'Current step: {current_step}, \
            System: {s}, System life: {engine_lives[s]}, Steps until failure: {steps_to_go}')
        
        while not done:
            action = env.action_space.sample()
            obs, rew, done, _ = env.step(action)
            total_cost += rew
            cntr += 1
            print(cntr, rew, done)
        print(total_cost)